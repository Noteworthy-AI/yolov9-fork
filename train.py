import os
import math
import time
import glob
import torch
import random
import numpy as np
import torch.nn as nn
import torch.distributed as dist

from tqdm import tqdm
from pathlib import Path
from copy import deepcopy
from datetime import datetime
from dataclasses import asdict
from PIL import Image
from torch.optim import lr_scheduler

from eval import evaluate
from models.yolo import Model
from utils.plots import plot_images
from utils.object_det_eval import compute_fitness
from utils.loss_tal import ComputeLoss as ComputeLossGELAN
from utils.loss_tal_dual import ComputeLoss as ComputeLossPGI
from utils.dataloaders import create_dataloader
from utils.autobatch import check_train_batch_size
from utils.torch_utils import EarlyStopping, ModelEMA, de_parallel, smart_DDP, smart_optimizer, smart_resume, torch_distributed_zero_first
from utils.general import TQDM_BAR_FORMAT, check_amp, check_img_size, colorstr, init_seeds, intersect_dicts, \
    labels_to_class_weights, labels_to_image_weights, one_cycle, one_flat_cycle, strip_optimizer


def train(cfg, device, wandb_logger, mldb_logger):
    # TODO: switch from print to logger?
    print("YOLOv9 Training Run")
    print("Initializing configs & logging")

    train_cfg = cfg.training
    nc = cfg.get_n_classes()  # number of classes
    names = cfg.dset.class_list  # class names
    # Directories & filepaths
    save_dir = cfg.get_local_output_dir(exist_ok=True)
    results_file = os.path.join(save_dir, "results.txt")
    weights_dir = os.path.join(save_dir, "weights")
    best_weight_pth = os.path.join(weights_dir, "best.pt")
    last_weight_pth = os.path.join(weights_dir, "last.pt")
    os.makedirs(weights_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, "train_plot_ims"), exist_ok=True)

    local_w_pth = cfg.get_local_weights_path()

    train_path = cfg.dset.get_local_path("train")
    val_path = cfg.dset.get_local_path("val")

    # Configure
    plots = not train_cfg.evolve  # create plots
    cuda = device.type != 'cpu'
    init_seeds(2 + train_cfg.global_rank)

    # Initialize YOLOv9 Model
    print("Initializing model & optimizer")
    ckpt = torch.load(local_w_pth, map_location='cpu')  # load checkpoint to CPU to avoid CUDA memory leak
    model_cfg = cfg.get_model_cfg()
    model = Model(model_cfg, ch=3, nc=nc, anchors=train_cfg.anchors).to(device)
    model.hyp = asdict(train_cfg)  # attach hyperparameters to model
    model.names = names
    exclude = ['anchor'] if not cfg.resume.enabled else []  # exclude keys
    state_dict = ckpt['model'].float().state_dict()  # to FP32
    state_dict = intersect_dicts(state_dict, model.state_dict(), exclude=exclude)  # intersect
    model.load_state_dict(state_dict, strict=False)      # Load state dictionary
    amp = check_amp(model)
    print(f"- transferred {len(state_dict)}/{len(model.state_dict())} items from {local_w_pth}")

    # Freeze model params
    freeze = cfg.model.freeze
    freeze = [f'model.{x}.' for x in (freeze if len(freeze) > 1 else range(freeze[0]))]
    for k, v in model.named_parameters():
        v.requires_grad = True  # train all layers
        if any(x in k for x in freeze):
            print(f"- freezing {k}")
            v.requires_grad = False

    # Initialize W&B logging
    with torch_distributed_zero_first(train_cfg.local_rank):
        if train_cfg.global_rank in {-1, 0}:
            last_epoch = ckpt.get('epoch') if cfg.resume.enabled else None
            wandb_logger.init(last_epoch)
            cfg_s3_uri = mldb_logger.log_configs(return_s3_uri=True)
            wandb_logger.save_s3_artifact(cfg_s3_uri, cfg.model_name, [], artifact_type="config")
            train_val_split = {
                "train": [fn for fn in os.listdir(os.path.join(train_path, "images")) if fn.endswith(".jpg")],
                "val": [fn for fn in os.listdir(os.path.join(val_path, "images")) if fn.endswith(".jpg")]
            }
            wandb_logger.save_json_artifact(train_val_split, cfg.model_name, [], "dataset")

    # Image size
    gs = max(int(model.stride.max()), 32)  # grid size (max stride)
    train_cfg.img_size = check_img_size(train_cfg.img_size, gs, floor=gs * 2)  # verify imgsz is gs-multiple

    # Batch size
    if train_cfg.global_rank == -1 and train_cfg.auto_batchsize:  # single-GPU only, estimate best batch size
        batch_size = check_train_batch_size(model, train_cfg.img_size, amp)
        total_batch_size = train_cfg.total_batch_size

    else:
        batch_size = train_cfg.total_batch_size // train_cfg.world_size
        total_batch_size = train_cfg.total_batch_size

    print(f"Training with a total batch size of {total_batch_size} ({batch_size} for {train_cfg.world_size} devices)")

    # Loss Function
    model_arch_type = cfg.get_model_arch_type()
    if model_arch_type == "gelan":
        compute_loss = ComputeLossGELAN(model)  # Loss function for GELAN architecture
    elif model_arch_type == "pgi":
        compute_loss = ComputeLossPGI(model)  # Loss function for full YOLOv9 (GELAN+PGI) architecture
    else:
        raise Exception("Invalid model architecture type")

    # Optimizer
    nbs = 64  # nominal batch size
    accumulate = max(round(nbs / total_batch_size), 1)  # accumulate loss before optimizing
    train_cfg.weight_decay *= total_batch_size * accumulate / nbs  # scale weight_decay
    optimizer = smart_optimizer(model, train_cfg.optimizer, train_cfg.lr0, train_cfg.momentum, train_cfg.weight_decay)

    # Model Exponential Moving Average (EMA)
    ema = ModelEMA(model) if train_cfg.global_rank in {-1, 0} else None

    # Resume & initialize early stopping TODO: verify this, not sure if this is correct
    early_stopping, stop = EarlyStopping(patience=train_cfg.patience), False
    start_epoch = 0
    if cfg.resume.enabled:
        best_fitness, start_epoch = smart_resume(ckpt, optimizer, ema, train_cfg.epochs)
        early_stopping.best_fitness = best_fitness
        early_stopping.best_epoch = start_epoch
    del ckpt, state_dict

    # Learning Rate Scheduler & Grad Scaler
    if train_cfg.lr_mode == "cos":
        lf = one_cycle(1, train_cfg.lrf, train_cfg.epochs)  # cosine 1->hyp['lrf']
    elif train_cfg.lr_mode == "flat_cos":
        lf = one_flat_cycle(1, train_cfg.lrf, train_cfg.epochs)  # flat cosine 1->hyp['lrf']
    elif train_cfg.lr_mode == "fixed":
        lf = lambda x: 1.0
    elif train_cfg.lr_mode == "linear":
        lf = lambda x: (1 - x / train_cfg.epochs) * (1.0 - train_cfg.lrf) + train_cfg.lrf # linear
    else:
        raise Exception("Invalid LR mode")

    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    scheduler.last_epoch = start_epoch - 1  # do not move
    scaler = torch.cuda.amp.GradScaler(enabled=amp)

    # DP mode
    if cuda and train_cfg.global_rank == -1 and torch.cuda.device_count() > 1:
        print("Using DP mode, not DDP")
        model = torch.nn.DataParallel(model)

    # SyncBatchNorm
    if train_cfg.sync_bn and cuda and train_cfg.global_rank != -1:
        print("Using SyncBatchNorm for DDP")
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)

    # Dataloaders TODO: run experiments on "close mosaic" & "min_items"
    print("Initializing data loaders")
    train_cfg.close_mosaic = 0
    train_cfg.min_items = 0
    train_loader, dataset = create_dataloader(train_cfg, train_path, gs, batch_size, (nc == 1), augment=True,
                                              prefix=colorstr('train: '), shuffle=True)
    class_weights = labels_to_class_weights(dataset.labels, nc).to(device) * nc  # attach class weights

    with torch_distributed_zero_first(train_cfg.local_rank):  # Val loader (for process 0)
        val_loader, _ = create_dataloader(train_cfg, val_path, gs, batch_size, (nc == 1), pad=0.5, prefix=colorstr('val: '))

        if not cfg.resume.enabled:  # TODO: autoanchor???
            model.half().float()  # pre-reduce anchor precision

    # DDP mode
    if cuda and train_cfg.global_rank != -1:
        model = smart_DDP(model, train_cfg)
        device = model.device
    else:
        model.device = next(model.parameters()).device
        model.device_type = next(model.parameters()).device
        model.device_ids = next(model.parameters()).device

    # Start training
    t0 = time.time()
    nb = len(train_loader)  # number of batches
    nw = max(round(train_cfg.warmup_epochs * nb), 100)  # number of warmup iterations, max(3 epochs, 100 iterations)

    last_opt_step = -1
    map95s = np.zeros(nc)  # mAP per class
    print(f"Using {train_loader.num_workers * train_cfg.world_size} workers across {train_cfg.world_size} devices")
    print(f'Starting training for {train_cfg.epochs} epochs...')
    print(f"Model device = {model.device} device type = {model.device_type}, device_ids = {model.device_ids}")
    print_batch_dims = True

    # Main Training Loop
    for epoch in range(start_epoch, train_cfg.epochs):
        model.train()

        # Update image weights (optional, single-GPU only)
        if train_cfg.image_weights:
            cw = model.class_weights.cpu().numpy() * (1 - map95s) ** 2 / nc  # class weights
            iw = labels_to_image_weights(dataset.labels, nc=nc, class_weights=cw)  # image weights
            dataset.indices = random.choices(range(dataset.n), weights=iw, k=dataset.n)  # rand weighted idx
        if epoch == (train_cfg.epochs - train_cfg.close_mosaic):
            print("Closing dataloader mosaic")
            dataset.mosaic = False

        # TODO: update mosaic borders?
        optimizer.zero_grad()
        mloss = torch.zeros(3, device=device)  # mean losses
        if train_cfg.global_rank != -1:
            train_loader.sampler.set_epoch(epoch)

        # Initialize progress bar & metric logging
        progress_header = ('%11s' * 7) % ('Epoch', 'GPU_mem', 'box_loss', 'cls_loss', 'dfl_loss', 'Instances', 'Size')
        progress_str = ""
        pbar = enumerate(train_loader)
        if train_cfg.global_rank in {-1, 0}:
            pbar = tqdm(pbar, total=nb, bar_format=TQDM_BAR_FORMAT)  # progress bar
            metrics_header = ('%11s' * 7) % ( 'Pre', 'Rec', 'mAP@0.5', 'mAP@0.95', 'box_loss', 'obj_loss', 'cls_loss')
            if not cfg.resume.enabled:
                with open(results_file, 'a') as f:
                    f.write(progress_header + metrics_header)

        # Main epoch loop
        for i, (imgs, targets, paths, _) in pbar:
            ni = i + nb * epoch  # number integrated batches (since train start)
            imgs = imgs.to(device, non_blocking=True).float() / 255  # uint8 to float32, 0-255 to 0.0-1.0

            if print_batch_dims:
                print(f"Training on tensors with shape {list(imgs.shape)}")
                print_batch_dims=False

            # Warmup
            if ni <= nw:
                xi = [0, nw]  # x interp
                # compute_loss.gr = np.interp(ni, xi, [0.0, 1.0])  # iou loss ratio (obj_loss = 1.0 or iou)
                accumulate = max(1, np.interp(ni, xi, [1, nbs / total_batch_size]).round())
                for j, x in enumerate(optimizer.param_groups):
                    # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                    x['lr'] = np.interp(ni, xi, [train_cfg.warmup_bias_lr if j == 0 else 0.0, x['initial_lr'] * lf(epoch)])
                    if 'momentum' in x:
                        x['momentum'] = np.interp(ni, xi, [train_cfg.warmup_momentum, train_cfg.momentum])

            # Multi-scale
            if train_cfg.multi_scale:
                sz = random.randrange(train_cfg.img_size * 0.5, train_cfg.img_size * 1.5 + gs) // gs * gs  # size
                sf = sz / max(imgs.shape[2:])  # scale factor
                if sf != 1:
                    ns = [math.ceil(x * sf / gs) * gs for x in imgs.shape[2:]]  # new shape (stretched to gs-multiple)
                    imgs = nn.functional.interpolate(imgs, size=ns, mode='bilinear', align_corners=False)

            # Forward pass
            with torch.cuda.amp.autocast(amp):
                pred = model(imgs)  # forward
                loss, loss_items = compute_loss(pred, targets.to(device))  # loss scaled by batch_size
                if train_cfg.global_rank != -1:
                    loss *= train_cfg.world_size  # gradient averaged between devices in DDP mode
                if train_cfg.quad:
                    loss *= 4.

            # Backward
            scaler.scale(loss).backward()

            # Optimize - https://pytorch.org/docs/master/notes/amp_examples.html
            if ni - last_opt_step >= accumulate:
                scaler.unscale_(optimizer)  # unscale gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)  # clip gradients
                scaler.step(optimizer)  # optimizer.step
                scaler.update()
                optimizer.zero_grad()
                if ema:
                    ema.update(model)
                last_opt_step = ni

            if train_cfg.global_rank in {-1, 0}:
                # TQDM progress bar log
                mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
                mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'  # (GB)
                ep_str = f'{epoch}/{train_cfg.epochs - 1}'
                progress_str = ('%11s' * 2 + '%11.4g' * 5) % (ep_str, mem, *mloss, targets.shape[0], imgs.shape[-1])
                pbar.set_description(progress_str)

                # W&B log training image examples
                if plots and ni < 10:
                    f = os.path.join(save_dir, "train_plot_ims", 'train_batch{}.jpg'.format(ni)) # filename
                    im_thread = plot_images(imgs, targets, paths, f)
                    threadjoin_status = im_thread.join()  # prevent hanging thread
                elif plots and ni == 10 and wandb_logger.wandb:
                    plotted_ims = [wandb_logger.wandb.Image(Image.open(str(fp)), caption=os.path.basename(fp))
                                   for fp in glob.glob(os.path.join(save_dir, "train_plot_ims", 'train*.jpg'))
                                   if os.path.exists(fp)]
                    wandb_logger.log({"train_vis/example_batches": plotted_ims}, log_type=wandb_logger.tracked_logs.AlwaysLogType)
        # end batch ------------------------------------------------------------------------------------------------

        # Scheduler
        scheduler.step()

        with torch_distributed_zero_first(train_cfg.local_rank):
            if train_cfg.global_rank in {-1, 0}:
                # Validation Loop
                print("Validation for epoch {}".format(epoch))

                ema.update_attr(model, include=['yaml', 'nc', 'hyp', 'names', 'stride', 'class_weights'])
                final_epoch = (epoch + 1 == train_cfg.epochs) or early_stopping.possible_stop
                wandb_logger.current_epoch = epoch
                val_out = evaluate(model=ema.ema, dataloader=val_loader, eval_mode="val", loss_fn=compute_loss,
                                   half_precision=amp, wandb_logger=wandb_logger, plot_save_dir=save_dir)

                val_mets, val_losses, val_cls_names, wandb_logger = val_out

                # Calculate fitness score for best model epoch
                fitness = compute_fitness(val_mets)  # weighted combination of [P, R, mAP@.5, mAP@.5-.95]
                stop = early_stopping(epoch=epoch, fitness=fitness)  # early stop check
                wandb_logger.current_epoch_is_best = (early_stopping.best_epoch == epoch)
                val_met_dict = {
                    'val_metrics/fitness': fitness,
                    'val_metrics/mean_recall': (val_mets["rec"] * val_mets["support"]).sum() / val_mets["support"].sum(),
                    'val_metrics/mean_precision': (val_mets["pre"] * val_mets["support"]).sum() / val_mets["support"].sum(),
                    'val_metrics/mAP_0.5': (val_mets["ap50"] * val_mets["support"]).sum() / val_mets["support"].sum(),
                    'val_metrics/mAP_0.95': (val_mets["ap95"] * val_mets["support"]).sum() / val_mets["support"].sum(),
                    'val_metrics/box_loss': val_losses[0],
                    'val_metrics/obj_loss': val_losses[1],
                    'val_metrics/cls_loss': val_losses[2],
                }

                if wandb_logger.wandb:
                    print("Logging validation results")
                    # W&B log training & validation results
                    wandb_logger.log({
                        'train_lrs/0_weights_no_decay': optimizer.param_groups[0]["lr"],
                        'train_lrs/1_weights_with_decay': optimizer.param_groups[1]["lr"],
                        'train_lrs/2_biases': optimizer.param_groups[2]["lr"]
                    })
                    wandb_logger.log({
                        'train_losses/box_loss': mloss[0].cpu().item(),
                        'train_losses/obj_loss': mloss[1].cpu().item(),
                        'train_losses/cls_loss': mloss[2].cpu().item(),
                    })
                    wandb_logger.log(val_met_dict)

                    wandb_logger.log({f'val_class_precision/{val_cls_names[i]}': val_mets.at[i, "pre"]
                                      for i in range(nc)})

                    wandb_logger.log({f'val_class_recall/{val_cls_names[i]}': val_mets.at[i, "rec"] for i in range(nc)})
                    wandb_logger.end_epoch(train_cfg.save_period > 0 and epoch % train_cfg.save_period == 0)

                # Local metric log for ML DB
                with open(results_file, 'a') as f:
                    vals = ["mean_precision", "mean_recall", "mAP_0.5", "mAP_0.95", "box_loss", "obj_loss", "cls_loss"]
                    f.write(progress_str + '%10.4g' * 7 % tuple([val_met_dict["val_metrics/{}".format(el)] for el in vals]) + '\n')

                # Save model
                if (not train_cfg.nosave) or (final_epoch and not train_cfg.evolve):  # if save
                    ckpt = {
                        'epoch': epoch,
                        'best_fitness': early_stopping.best_fitness,
                        'model': deepcopy(de_parallel(model)).half(),
                        'ema': deepcopy(ema.ema).half(),
                        'updates': ema.updates,
                        'optimizer': optimizer.state_dict(),
                        'opt': asdict(train_cfg),
                        'git': None,
                        'date': datetime.now().isoformat(),
                        'wandb_id': wandb_logger.wandb_run.id if wandb_logger.wandb else None
                    }

                    # Save last, best and delete
                    ep_ckpt_save_pth = os.path.join(weights_dir, f'epoch{epoch}.pt')
                    torch.save(ckpt, last_weight_pth)
                    if (early_stopping.best_epoch == epoch):
                        torch.save(ckpt, best_weight_pth)
                    if train_cfg.save_period > 0 and epoch % train_cfg.save_period == 0:
                        torch.save(ckpt, ep_ckpt_save_pth)
                        if train_cfg.optimize_ckpt_epochs:
                            strip_optimizer(ep_ckpt_save_pth)

                        if (not final_epoch):
                            s3_uri = mldb_logger.log_checkpoint(ckpt, ep_ckpt_save_pth, return_s3_uri = True)
                            if wandb_logger.wandb:
                                wandb_logger.save_s3_artifact(s3_uri, cfg.model_name, aliases = [f"epoch-{epoch}"])
                    del ckpt

        # EarlyStopping
        if train_cfg.global_rank != -1:  # if DDP training
            broadcast_list = [stop if train_cfg.global_rank == 0 else None]
            dist.broadcast_object_list(broadcast_list, 0)  # broadcast 'stop' to all ranks
            if train_cfg.global_rank != 0:
                stop = broadcast_list[0]
        if stop:
            break  # must break all DDP ranks

        # end epoch ----------------------------------------------------------------------------------------------------
    # end training -----------------------------------------------------------------------------------------------------

    with torch_distributed_zero_first(train_cfg.local_rank):
        if train_cfg.global_rank in {-1, 0}:
            print(f'\n{train_cfg.epochs - start_epoch + 1} epochs completed in {(time.time() - t0) / 3600:.3f} hours.')
            # Strip optimizers from saved weights
            for f in Path(last_weight_pth), Path(best_weight_pth):
                if f.exists():
                    strip_optimizer(f, f)

    torch.cuda.empty_cache()
    return wandb_logger, mldb_logger
