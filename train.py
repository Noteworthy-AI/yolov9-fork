import os
import math
import time
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
from torch.optim import lr_scheduler

from test import evaluate
from models.yolo import Model
from utils.metrics import fitness
from utils.loss_tal import ComputeLoss
from models.experimental import attempt_load
from utils.dataloaders import create_dataloader
from utils.autobatch import check_train_batch_size
from utils.torch_utils import EarlyStopping, ModelEMA, de_parallel, smart_DDP, smart_optimizer
from utils.general import TQDM_BAR_FORMAT, check_amp, check_img_size, colorstr, init_seeds, intersect_dicts, \
    labels_to_class_weights, labels_to_image_weights, one_cycle, one_flat_cycle, strip_optimizer


LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))
GLOBAL_RANK = int(os.getenv('RANK', -1))
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))


def train(cfg, device, wandb_logger, mldb_logger):
    # TODO: switch from print to logger?
    print("YOLOv7 Training Run")
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

    local_w_pth = cfg.get_local_weights_path()

    train_path = cfg.dset.get_local_path("train")
    val_path = cfg.dset.get_local_path("val")

    # Configure
    plots = not train_cfg.evolve  # create plots
    cuda = device.type != 'cpu'
    init_seeds(2 + GLOBAL_RANK)

    # Initialize YOLOv9 Model
    print("Initializing model & optimizer")
    ckpt = torch.load(local_w_pth, map_location='cpu')  # load checkpoint to CPU to avoid CUDA memory leak
    model = Model(cfg or ckpt['model'].yaml, ch=3, nc=nc, anchors=train_cfg.anchors).to(device)  # create
    exclude = ['anchor'] if not cfg.resume else []  # exclude keys
    state_dict = ckpt['model'].float().state_dict()  # to FP32
    state_dict = intersect_dicts(state_dict, model.state_dict(), exclude=exclude)  # intersect
    model.load_state_dict(state_dict, strict=False)  # load
    amp = check_amp(model)
    print(f"- transferred {len(state_dict)}/{len(model.state_dict())} items from {local_w_pth}")

    # Freeze
    # parameter names to freeze (full or partial)
    freeze = cfg.model.freeze
    freeze = [f'model.{x}.' for x in (freeze if len(freeze) > 1 else range(freeze[0]))]
    for k, v in model.named_parameters():
        v.requires_grad = True  # train all layers
        if any(x in k for x in freeze):
            print(f"- freezing {k}")
            v.requires_grad = False

    # Image size
    gs = max(int(model.stride.max()), 32)  # grid size (max stride)
    train_cfg.img_size = check_img_size(train_cfg.img_size, gs, floor=gs * 2)  # verify imgsz is gs-multiple

    # Batch size
    if GLOBAL_RANK == -1 and train_cfg.auto_batchsize:  # single-GPU only, estimate best batch size
        batch_size = check_train_batch_size(model, train_cfg.img_size, amp)
    else:
        batch_size = train_cfg.batch_size // WORLD_SIZE

    print(f"Training on {WORLD_SIZE} devices with batch size {batch_size} per device")

    # Optimizer
    nbs = 64  # nominal batch size
    accumulate = max(round(nbs / batch_size), 1)  # accumulate loss before optimizing
    train_cfg.weight_decay *= batch_size * accumulate / nbs  # scale weight_decay
    optimizer = smart_optimizer(model, train_cfg.optimizer, train_cfg.lr0, train_cfg.momentum, train_cfg.weight_decay)

    # Learning Rate Scheduler
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

    # Model Exponential Moving Average (EMA)
    ema = ModelEMA(model) if GLOBAL_RANK in {-1, 0} else None

    # Resume
    best_fitness, start_epoch = 0.0, 0
    if cfg.resume:
        pass    # TODO: fix resume from checkpoint
    del ckpt, state_dict

    # DP mode
    if cuda and GLOBAL_RANK == -1 and torch.cuda.device_count() > 1:
        print("Using DP mode, not DDP")
        model = torch.nn.DataParallel(model)

    # SyncBatchNorm
    if train_cfg.sync_bn and cuda and GLOBAL_RANK != -1:
        print("Using SyncBatchNorm for DDP")
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)

    # Trainloader
    print("Initializing data loaders")
    # TODO: experimental "close mosaic" & "min_items"
    train_cfg.close_mosaic = 0
    train_cfg.min_items = 0
    train_loader, dataset = create_dataloader(train_cfg, train_path, gs, LOCAL_RANK, batch_size, (nc == 1),
                                              augment=True, prefix=colorstr('train: '), shuffle=True)

    labels = np.concatenate(dataset.labels, 0)
    mlc = int(labels[:, 0].max())  # max label class
    lb_msg = f'Label class {mlc} exceeds nc={nc} in {cfg.dset.train_dataset_name}. Possible class labels are 0-{nc - 1}'
    assert mlc < nc, lb_msg

    # Process 0
    if GLOBAL_RANK in {-1, 0}:
        val_loader, _ = create_dataloader(train_cfg, val_path, gs, -1, batch_size, (nc == 1), pad=0.5, prefix=colorstr('val: '))

        if not cfg.resume:
            # TODO: autoanchor???
            model.half().float()  # pre-reduce anchor precision

    # DDP mode
    if cuda and GLOBAL_RANK != -1:
        model = smart_DDP(model)

    # Model attributes (TODO: scale to # of layers?)
    model.nc = nc  # attach number of classes to model
    model.hyp = asdict(train_cfg)  # attach hyperparameters to model
    model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device) * nc  # attach class weights
    model.names = names

    # Start training
    t0 = time.time()
    nb = len(train_loader)  # number of batches
    nw = max(round(train_cfg.warmup_epochs * nb), 100)  # number of warmup iterations, max(3 epochs, 100 iterations)

    last_opt_step = -1
    maps = np.zeros(nc)  # mAP per class
    results = (0, 0, 0, 0, 0, 0, 0)  # P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)
    scheduler.last_epoch = start_epoch - 1  # do not move
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    stopper, stop = EarlyStopping(patience=train_cfg.patience), False
    compute_loss = ComputeLoss(model)  # init loss class

    print(f"Using {train_loader.num_workers * WORLD_SIZE} workers across {WORLD_SIZE} devices")
    sample_model_input = train_loader[0][0]
    print(f"Training on tensors with shape {sample_model_input.shape}")
    print(f'Starting training for {train_cfg.epochs} epochs...')

    # Main Training Loop
    for epoch in range(start_epoch, train_cfg.epochs):
        model.train()

        # Update image weights (optional, single-GPU only)
        if train_cfg.image_weights:
            cw = model.class_weights.cpu().numpy() * (1 - maps) ** 2 / nc  # class weights
            iw = labels_to_image_weights(dataset.labels, nc=nc, class_weights=cw)  # image weights
            dataset.indices = random.choices(range(dataset.n), weights=iw, k=dataset.n)  # rand weighted idx
        if epoch == (train_cfg.epochs - train_cfg.close_mosaic):
            print("Closing dataloader mosaic")
            dataset.mosaic = False

        # TODO: update mosaic borders?

        mloss = torch.zeros(3, device=device)  # mean losses
        if GLOBAL_RANK != -1:
            train_loader.sampler.set_epoch(epoch)
        pbar = enumerate(train_loader)
        print(('\n' + '%11s' * 7) % ('Epoch', 'GPU_mem', 'box_loss', 'cls_loss', 'dfl_loss', 'Instances', 'Size'))
        if GLOBAL_RANK in {-1, 0}:
            pbar = tqdm(pbar, total=nb, bar_format=TQDM_BAR_FORMAT)  # progress bar
        optimizer.zero_grad()

        # Main epoch loop
        for i, (imgs, targets, paths, _) in pbar:
            ni = i + nb * epoch  # number integrated batches (since train start)
            imgs = imgs.to(device, non_blocking=True).float() / 255  # uint8 to float32, 0-255 to 0.0-1.0

            # Warmup
            if ni <= nw:
                xi = [0, nw]  # x interp
                # compute_loss.gr = np.interp(ni, xi, [0.0, 1.0])  # iou loss ratio (obj_loss = 1.0 or iou)
                accumulate = max(1, np.interp(ni, xi, [1, nbs / batch_size]).round())
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
                if GLOBAL_RANK != -1:
                    loss *= WORLD_SIZE  # gradient averaged between devices in DDP mode
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

            # Log
            if GLOBAL_RANK in {-1, 0}:
                mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
                mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'  # (GB)
                pbar.set_description(('%11s' * 2 + '%11.4g' * 5) %
                                     (f'{epoch}/{train_cfg.epochs - 1}', mem, *mloss, targets.shape[0], imgs.shape[-1]))
            # end batch ------------------------------------------------------------------------------------------------

        # Scheduler
        lr = [x['lr'] for x in optimizer.param_groups]  # for loggers
        scheduler.step()

        if GLOBAL_RANK in {-1, 0}:
            # mAP
            ema.update_attr(model, include=['yaml', 'nc', 'hyp', 'names', 'stride', 'class_weights'])
            final_epoch = (epoch + 1 == train_cfg.epochs) or stopper.possible_stop

            results, maps, _ = evaluate(cfg, val_loader, batch_size, device, nc, mode="val", half_precision=amp,
                                        model=ema.ema, plots=False, compute_loss=compute_loss)

            # Update best mAP
            fi = fitness(np.array(results).reshape(1, -1))  # weighted combination of [P, R, mAP@.5, mAP@.5-.95]
            stop = stopper(epoch=epoch, fitness=fi)  # early stop check
            if fi > best_fitness:
                best_fitness = fi

            # Save model
            if (not train_cfg.nosave) or (final_epoch and not train_cfg.evolve):  # if save
                ckpt = {
                    'epoch': epoch,
                    'best_fitness': best_fitness,
                    'model': deepcopy(de_parallel(model)).half(),
                    'ema': deepcopy(ema.ema).half(),
                    'updates': ema.updates,
                    'optimizer': optimizer.state_dict(),
                    'opt': asdict(train_cfg),
                    'git': None,
                    'date': datetime.now().isoformat()
                }

                # Save last, best and delete
                torch.save(ckpt, last_weight_pth)
                if best_fitness == fi:
                    torch.save(ckpt, best_weight_pth)
                if train_cfg.save_period > 0 and epoch % train_cfg.save_period == 0:
                    torch.save(ckpt, Path(weights_dir) / f'epoch{epoch}.pt')
                del ckpt

        # EarlyStopping
        if GLOBAL_RANK != -1:  # if DDP training
            broadcast_list = [stop if GLOBAL_RANK == 0 else None]
            dist.broadcast_object_list(broadcast_list, 0)  # broadcast 'stop' to all ranks
            if GLOBAL_RANK != 0:
                stop = broadcast_list[0]
        if stop:
            break  # must break all DDP ranks

        # end epoch ----------------------------------------------------------------------------------------------------
    # end training -----------------------------------------------------------------------------------------------------

    if GLOBAL_RANK in {-1, 0}:
        print(f'\n{train_cfg.epochs - start_epoch + 1} epochs completed in {(time.time() - t0) / 3600:.3f} hours.')
        for f in Path(last_weight_pth), Path(best_weight_pth):
            # Strip optimizers from saved weights
            if f.exists():
                if f is Path(last_weight_pth):
                    strip_optimizer(f, f)
                else:
                    strip_optimizer(f, f)

                if f is Path(best_weight_pth):
                    print("Final evaluation on best weights")
                    test_model = attempt_load(f, device).half()
                    results, _, _ = evaluate(cfg, val_loader, batch_size, device, nc, mode="val", model=test_model,
                                             plots=True, verbose=True, compute_loss=compute_loss)

    torch.cuda.empty_cache()
    return results
