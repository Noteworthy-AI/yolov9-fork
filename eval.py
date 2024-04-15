import os
import cv2
import glob
import wandb
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
from PIL import Image

from utils.general import TQDM_BAR_FORMAT, Profile, scale_boxes, xywh2xyxy, xyxy2xywh
from utils.general import non_max_suppression as nms
from utils.metrics import ConfusionMatrix, ap_per_class, box_iou
from utils.plots import output_to_target, plot_images
from utils.torch_utils import smart_inference_mode


def process_batch(detections, labels, iouv):
    """
    Return correct prediction matrix
    Arguments:
        detections (array[N, 6]), x1, y1, x2, y2, conf, class
        labels (array[M, 5]), class, x1, y1, x2, y2
    Returns:
        correct (array[N, 10]), for 10 IoU levels
    """
    correct = np.zeros((detections.shape[0], iouv.shape[0])).astype(bool)
    iou = box_iou(labels[:, 1:], detections[:, :4])
    correct_class = labels[:, 0:1] == detections[:, 5]
    for i in range(len(iouv)):
        x = torch.where((iou >= iouv[i]) & correct_class)  # IoU > threshold and classes match
        if x[0].shape[0]:
            matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()  # [label, detect, iou]
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                # matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            correct[matches[:, 1].astype(int), i] = True
    return torch.tensor(correct, dtype=torch.bool, device=iouv.device)


@smart_inference_mode()
def evaluate(cfg, dataloader, batch_size, device, n_classes, mode="test", model=None, conf_thres=0.001, iou_thres=0.7,
             max_det=300, augment=False, verbose=False, half_precision=True, plots=True, compute_loss=None,
             wandb_logger=None, n_pred_plot=3):

    plot_save_dir = os.path.join(cfg.get_local_output_dir(exist_ok=True), f"{mode}_plot_ims")
    os.makedirs(plot_save_dir, exist_ok=True)
    curr_epoch = wandb_logger.current_epoch

    # Initialize/load model and set device
    cuda = device.type != 'cpu'
    if mode == "val":
        assert model is not None, "Must specify model object for validation mode"
        device, pt, jit, engine = next(model.parameters()).device, True, False, False  # get model device, PyTorch model
        half_precision &= cuda  # half precision only supported on CUDA
        model.half() if half_precision else model.float()
    elif mode == "test":
        # TODO: setup test mode for standalone eval
        pass
    else:
        raise Exception("Invalid eval mode")

    # Configure
    model.eval()
    iouv = torch.linspace(0.5, 0.95, 10, device=device)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()

    seen = 0
    confusion_matrix = ConfusionMatrix(nc=n_classes)
    names = model.names if hasattr(model, 'names') else model.module.names  # get class names
    if isinstance(names, (list, tuple)):  # old format
        names = dict(enumerate(names))

    s = ('%22s' + '%11s' * 6) % ('Class', 'Images', 'Instances', 'P', 'R', 'mAP50', 'mAP50-95')
    tp, fp, p50, r50, maxf1, mp, mr, map50, ap50, map95, ap95 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    dt = Profile(), Profile(), Profile()  # profiling times
    loss = torch.zeros(3, device=device)
    jdict, stats, ap, ap_class, wandb_images = [], [], [], [], []

    # Logging
    log_imgs = 0
    if wandb_logger and wandb_logger.wandb:
        log_imgs = min(wandb_logger.log_imgs, 100)

    pbar = tqdm(dataloader, desc=s, bar_format=TQDM_BAR_FORMAT)  # progress bar
    plot_threads = []
    for batch_i, (im, targets, paths, shapes) in enumerate(pbar):
        with dt[0]:
            if cuda:
                im = im.to(device, non_blocking=True)
                targets = targets.to(device)
            im = im.half() if half_precision else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            nb, _, height, width = im.shape  # batch size, channels, height, width

        # Inference
        with dt[1]:
            preds, train_out = model(im) if compute_loss else (model(im, augment=augment), None)

        # Loss
        if compute_loss:
            loss += compute_loss(train_out, targets)[1]  # box, obj, cls

        # NMS
        targets[:, 2:] *= torch.tensor((width, height, width, height), device=device)  # to pixels
        with dt[2]:
            preds = nms(preds, conf_thres, iou_thres, multi_label=True, agnostic=(n_classes == 1), max_det=max_det)

        # Metrics
        for si, pred in enumerate(preds):
            labels = targets[targets[:, 0] == si, 1:]
            nl, npr = labels.shape[0], pred.shape[0]  # number of labels, predictions
            path, shape = Path(paths[si]), shapes[si][0]
            correct = torch.zeros(npr, niou, dtype=torch.bool, device=device)  # init
            seen += 1

            if npr == 0:
                if nl:
                    stats.append((correct, *torch.zeros((2, 0), device=device), labels[:, 0]))
                    if plots:
                        confusion_matrix.process_batch(detections=None, labels=labels[:, 0])
                continue

            # Predictions
            if n_classes == 1:
                pred[:, 5] = 0
            predn = pred.clone()
            scale_boxes(im[si].shape[1:], predn[:, :4], shape, shapes[si][1])  # native-space pred

            # W&B log validation images + predictions
            if (len(wandb_images) < log_imgs):
                box_data = [{"position": {"minX": xyxy[0], "minY": xyxy[1], "maxX": xyxy[2], "maxY": xyxy[3]},
                             "class_id": int(cls),
                             "box_caption": "%s %.3f" % (names[cls], conf),
                             "scores": {"class_score": conf},
                             "domain": "pixel"} for *xyxy, conf, cls in pred.tolist()]
                boxes = {"predictions": {"box_data": box_data, "class_labels": names}}  # inference-space
                wandb_images.append(wandb_logger.wandb.Image(im[si], boxes=boxes, caption=path.name))

            # Evaluate
            if nl:
                tbox = xywh2xyxy(labels[:, 1:5])  # target boxes
                scale_boxes(im[si].shape[1:], tbox, shape, shapes[si][1])  # native-space labels
                labelsn = torch.cat((labels[:, 0:1], tbox), 1)  # native-space labels
                correct = process_batch(predn, labelsn, iouv)
                if plots:
                    confusion_matrix.process_batch(predn, labelsn)
            stats.append((correct, pred[:, 4], pred[:, 5], labels[:, 0]))  # (correct, conf, pcls, tcls)

        # Plot images
        if plots and batch_i < n_pred_plot:
            lb_plot_path = Path(plot_save_dir) / f'ex_b{batch_i}_labels.png'
            pred_plot_path = Path(plot_save_dir) / f'ex_b{batch_i}_preds.png'
            lb_plot_thread = plot_images(im, targets, paths, lb_plot_path, names=names)   # labels
            pred_plot_thread = plot_images(im, targets, paths, pred_plot_path, names=names)  # preds
            threadjoin_status = [thread.join() for thread in [lb_plot_thread, pred_plot_thread]]

    # Compute metrics
    stats = [torch.cat(x, 0).cpu().numpy() for x in zip(*stats)]  # to numpy
    if len(stats) and stats[0].any():
        tp, fp, p50, r50, maxf1, ap, ap_class = ap_per_class(*stats, plot=plots, save_dir=plot_save_dir, names=names)
        ap50, ap95 = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
        mp, mr, map50, map95 = p50.mean(), r50.mean(), ap50.mean(), ap95.mean()
    nt = np.bincount(stats[3].astype(int), minlength=n_classes)  # number of targets per class

    # Print results
    pf = '%22s' + '%11i' * 2 + '%11.3g' * 4  # print format
    print(pf % ('all', seen, nt.sum(), mp, mr, map50, map95))
    if nt.sum() == 0:
        print(f'WARNING ⚠️ no labels found in {mode} set, can not compute metrics without labels')

    # Print results per class
    if (verbose or (n_classes < 50 and mode != "val")) and n_classes > 1 and len(stats):
        for i, c in enumerate(ap_class):
            print(pf % (names[c], seen, nt[c], p50[i], r50[i], ap50[i], ap95[i]))

    # Print speeds
    t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
    if mode != "val":
        shape = (batch_size, 3, cfg.training.img_size, cfg.training.img_size)
        print(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {shape}' % t)

    if plots:
        # Plot confusion matrix
        confusion_matrix.plot(save_dir=plot_save_dir, names=list(names.values()))

        # W&B log class-wise precision
        if wandb_logger and wandb_logger.wandb:
            if isinstance(maxf1, np.ndarray):
                maxf1 = maxf1.mean()
            rounded_max_f1 = round(maxf1, 3)
            precision_recall_table = wandb.Table(
                columns=['Category', f'Precision_@{rounded_max_f1}', f'Recall_@{rounded_max_f1}'],
                data=[[names[c], p50[i], r50[i]] for i, c in enumerate(ap_class)])
            wandb_logger.log({f'pr_tables/{mode}-split': precision_recall_table})

        m_fns = ["confusion_matrix.png", "PR_curve.png", "F1_curve.png", "P_curve.png", "R_curve.png"]
        m_ims = [os.path.join(plot_save_dir, im_name) for im_name in m_fns]
        wb_m_ims = [wandb_logger.wandb.Image(Image.open(f), caption='ep{}_{}'.format(curr_epoch, os.path.basename(f))) for f in m_ims]
        wandb_logger.log({'{}-metric-plots'.format(mode): wb_m_ims}, log_type="images")

        if wandb_images:
            wandb_logger.log({"Predictions per epoch (val set)/Images": wandb_images}, log_type="images")


    # Per-class mAP@0.95
    map95s = np.zeros(n_classes)
    for i, c in enumerate(ap_class):
        map95s[c] = ap95[i]

    # Return validation precision and recall per class
    ap_class_list = list(ap_class)
    class_names, classes_precision, classes_recall = [], [], []
    for i in range(len(names)):
        if i in ap_class_list:
            classes_precision.append(p50[ap_class_list.index(i)])
            classes_recall.append(r50[ap_class_list.index(i)])
        else:
            classes_precision.append(0.0)
            classes_recall.append(0.0)

    model.float()  # for training

    formatted_results = (mp, mr, map50, map95, *(loss.cpu() / len(dataloader)).tolist())
    return formatted_results, map95s, t, names, classes_precision, classes_recall, wandb_logger
