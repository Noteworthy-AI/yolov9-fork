import os
import wandb
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image, ImageFile

from utils.plots import plot_images
from utils.torch_utils import smart_inference_mode
from utils.general import non_max_suppression as nms
from utils.general import TQDM_BAR_FORMAT, scale_boxes, xywh2xyxy
from utils.object_det_eval import evaluate_detections, compute_obj_det_eval_metrics, compute_conf_mat
from utils.object_det_plot import plot_confusion_matrix, plot_mc_curve, plot_pr_curve, format_conf_mat_df

ImageFile.LOAD_TRUNCATED_IMAGES = True


def wandb_im_pred_plot(in_im, in_preds, im_fp, cls_names):
    box_data = [{"position": {"minX": xyxy[0], "minY": xyxy[1], "maxX": xyxy[2], "maxY": xyxy[3]},
                 "class_id": int(cls),
                 "box_caption": "%s %.3f" % (cls_names[cls], conf),
                 "scores": {"class_score": conf},
                 "domain": "pixel"} for *xyxy, conf, cls in in_preds.tolist()]

    boxes = {"predictions": {"box_data": box_data, "class_labels": cls_names}}  # inference-space
    return wandb.Image(in_im, boxes=boxes, caption=os.path.basename(im_fp))


@smart_inference_mode()
def evaluate(model, dataloader, eval_mode="test", loss_fn=None, nms_conf_thres=0.001, nms_iou_thres=0.7,
             max_nms_det=300, half_precision=True, plot_save_dir=None, wandb_logger=None, n_pred_plot=3,
             conf_mat_conf_thresh: float = 0.7, curr_epoch: int = 0):
    # Initialize/load model and set device
    if eval_mode == "val":
        # Configure model for validation mode
        device, pt, jit, engine = next(model.parameters()).device, True, False, False  # get model device, PyTorch model
        cuda = device.type != 'cpu'
        half_precision &= cuda  # half precision only supported on CUDA
        model.half() if half_precision else model.float()
        model.eval()
    elif eval_mode == "test":
        # TODO: setup test mode for standalone eval
        raise Exception("Test mode not implemented")
    else:
        raise Exception("Invalid eval mode")

    # Format class names
    names = model.names if hasattr(model, 'names') else model.module.names  # get class names
    if isinstance(names, (list, tuple)):  # old format
        names = dict(enumerate(names))
    n_classes = len(names.keys())

    # Initialize counters & start eval loop
    wb_pred_vis = []  # W&B image logging
    if wandb_logger and wandb_logger.wandb:
        log_imgs = wandb_logger.log_imgs
    else:
        log_imgs = 0

    loss = torch.zeros(3, device=device)
    eval_ious = [0.5 + 0.05 * i for i in range(10)]  # iou vector for mAP@0.5:0.95

    # Initialize example im/prediction saving
    if plot_save_dir is not None:
        eval_im_save_dir = os.path.join(plot_save_dir, "{}_example_ims".format(eval_mode))
        eval_met_save_dir = os.path.join(plot_save_dir, "{}_metrics".format(eval_mode))
        os.makedirs(eval_im_save_dir, exist_ok=True)
        os.makedirs(eval_met_save_dir, exist_ok=True)
    else:
        eval_im_save_dir = None
        eval_met_save_dir = None

    # Format results DF
    cols = ["pred_conf", "pred_class"] + ["iou_match_{}".format(round(eval_ious[i], 2)) for i in range(len(eval_ious))]
    results_df = pd.DataFrame(columns=cols)
    confusion_mat = np.zeros((n_classes + 1, n_classes + 1)).astype(int)
    for batch_i, (im, targets, im_paths, im_shapes) in enumerate(tqdm(dataloader, bar_format=TQDM_BAR_FORMAT)):
        # Format input image
        if cuda:
            im = im.to(device, non_blocking=True)
            targets = targets.to(device)
        im = im.half() if half_precision else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        nb, _, height, width = im.shape  # batch size, channels, height, width

        # Inference
        preds, train_out = model(im) if loss_fn is not None else (model(im, augment=False), None)

        # Loss
        if loss_fn is not None:
            loss += loss_fn(train_out, targets)[1]  # box, obj, cls

        # NMS
        targets[:, 2:] *= torch.tensor((width, height, width, height), device=device)  # to pixels
        preds = nms(preds, nms_conf_thres, nms_iou_thres, multi_label=True, max_det=max_nms_det)

        # Process predicted and label bounding boxes
        for si, pred in enumerate(preds):
            labels = targets[targets[:, 0] == si, 1:]
            nl, npr = labels.shape[0], pred.shape[0]

            if npr > 0:
                preds_i = pred.clone()
                scaled_pred_bboxes = scale_boxes(im[si].shape[1:], preds_i[:, :4], im_shapes[si][0], im_shapes[si][1])
                preds_i[:, :4] = scaled_pred_bboxes
            else:
                preds_i = torch.zeros((0, 6), device=device)

            if nl > 0:
                raw_gt_bboxes = xywh2xyxy(labels[:, 1:5])
                scaled_gt_bboxes = scale_boxes(im[si].shape[1:], raw_gt_bboxes, im_shapes[si][0], im_shapes[si][1])
                gt_i = torch.cat((labels[:, 0:1], scaled_gt_bboxes), 1)  # native-space labels
            else:
                gt_i = torch.zeros((0, 5), device=device)

            # Compute image-wise metrics
            im_result_df = evaluate_detections(preds_i, gt_i, eval_ious)
            if len(results_df) == 0:
                results_df = im_result_df.copy()
            else:
                results_df = pd.concat((results_df, im_result_df)).reset_index(drop=True).copy()

            # Compute image confusion matrix
            im_conf_mat = compute_conf_mat(preds_i, gt_i, conf_mat_conf_thresh, conf_mat_conf_thresh, n_classes)
            confusion_mat += im_conf_mat

            # Plot eval predictions on images for W&B
            if len(wb_pred_vis) < log_imgs:
                wb_im = wandb_im_pred_plot(im[si], pred, im_paths[si], names)
                wb_pred_vis.append(wb_im)

            # Plot images
            if plot_save_dir is not None and batch_i < n_pred_plot:
                lb_plot_path = os.path.join(eval_im_save_dir, 'ex_b{}_labels.png'.format(batch_i))
                pred_plot_path = os.path.join(eval_im_save_dir, 'ex_b{}_preds.png'.format(batch_i))
                lb_plot_thread = plot_images(im, targets, im_paths, lb_plot_path, names=names)  # labels
                pred_plot_thread = plot_images(im, targets, im_paths, pred_plot_path, names=names)  # preds
                threadjoin_status = [thread.join() for thread in [lb_plot_thread, pred_plot_thread]]

    # Compute aggregate metrics
    gt_class_dets = confusion_mat[:, :-1].sum(axis=0).astype(int)
    cw_metrics, mc_pts = compute_obj_det_eval_metrics(results_df, gt_class_dets, metric_iou_thresh=conf_mat_conf_thresh,
                                                      min_conf_thresh=nms_conf_thres, class_names=names)

    if plot_save_dir is not None:
        # Plot confusion matrix
        conf_mat_df = format_conf_mat_df(confusion_mat, names, normalize=True)
        conf_thread = plot_confusion_matrix(conf_mat_df,
                                            save_path=os.path.join(eval_met_save_dir, "confusion_matrix.png"))

        # Plot metric curves
        prcurve_thread = plot_pr_curve(mc_pts["px"], mc_pts["py"], cw_metrics["ap50"].to_numpy(),
                                       os.path.join(eval_met_save_dir, "pr_curve.png"), names)
        f1fp = os.path.join(eval_met_save_dir, "f1_curve.png")
        f1curve_thread = plot_mc_curve(mc_pts["px"], mc_pts["f1"], f1fp, names, ylabel='F1')
        prefp = os.path.join(eval_met_save_dir, "precision_curve.png")
        pcurve_thread = plot_mc_curve(mc_pts["px"], mc_pts["pre"], prefp, names, ylabel='Precision')
        recfp = os.path.join(eval_met_save_dir, "recall_curve.png")
        rcurve_thread = plot_mc_curve(mc_pts["px"], mc_pts["rec"], recfp, names, ylabel='Recall')
        im_threads = [conf_thread, prcurve_thread, f1curve_thread, pcurve_thread, rcurve_thread]
        threadjoin_status = [thread.join() for thread in im_threads]

        # Log to W&B
        if wandb_logger and wandb_logger.wandb:
            # Log eval batch examples
            wb_ex_ims = [wandb.Image(Image.open(os.path.join(eval_im_save_dir, fn)),
                                     caption='ep{}_{}'.format(curr_epoch, os.path.basename(fn)))
                         for fn in os.listdir(eval_im_save_dir)]
            wandb_logger.log({'{}_vis/example_batches'.format(eval_mode): wb_ex_ims})
            # Log curves & confusion matrices
            m_fns = ["confusion_matrix.png", "pr_curve.png", "f1_curve.png", "precision_curve.png", "recall_curve.png"]
            m_ims = [os.path.join(eval_met_save_dir, im_name) for im_name in m_fns]
            wb_m_ims = [wandb.Image(Image.open(f), caption='ep{}_{}'.format(curr_epoch, os.path.basename(f)))
                        for f in m_ims]

            wandb_logger.log({'{}_metric_plots/{}'.format(eval_mode, m_fns[i]): wb_m_ims[i] for i in range(len(m_ims))},
                             log_type=wandb_logger.tracked_logs.BestCandidateLogType)

            # Log prediction examples
            if wb_pred_vis:
                wandb_logger.log({"{}_images/prediction_examples".format(eval_mode): wb_pred_vis},
                                 log_type=wandb_logger.tracked_logs.EpochLogType)

            # Log eval summary: table with classwise metrics, bar plot with recall values
            wb_metrics_table = wandb.Table(dataframe=cw_metrics)
            bar_chart = wandb.plot.bar(
                wb_metrics_table,
                label="class_name",
                value="rec",
                title="Validation Recall per Class"
            )
            wandb_logger.log({"{}_summary/metric_table".format(eval_mode): wb_metrics_table})

            wandb_logger.log({"{}_summary/recall_chart".format(eval_mode): bar_chart})

    out_losses = (loss.cpu() / len(dataloader)).tolist()
    model.float()  # for training

    return cw_metrics, out_losses, names, wandb_logger
