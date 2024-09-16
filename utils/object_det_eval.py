import torch
import numpy as np
import pandas as pd


def compute_obj_det_eval_metrics(in_result_df: pd.DataFrame, in_gt_class_labels: np.ndarray, class_names: list[str]=None,
                                 metric_iou_thresh: float = 0.7, min_conf_thresh: float = 0.001):

    # Initialive IOU values
    iou_vals = [float(el.replace('iou_match_', "")) for el in in_result_df.columns if el.startswith('iou_match_')]
    iou_idx = np.abs(np.asarray(iou_vals) - metric_iou_thresh).argmin()
    err_msg = "Invalid input IOU threshold (must be in existing IOU buckets, was instead {}".format(metric_iou_thresh)
    assert np.abs(np.asarray(iou_vals) - metric_iou_thresh).min() < 1e-6, err_msg

    # Initialize metric arrays
    nc = in_gt_class_labels.shape[0]
    px, py = np.linspace(min_conf_thresh, 1, 1000), []  # for plotting
    aps = np.zeros((nc, len(iou_vals)))
    p, r, interp_confs = np.zeros((nc, 1000)), np.zeros((nc, 1000)), np.zeros((nc, 1000))
    if class_names is None:
        class_names = [i for i in range(nc)]

    # Compute object detection eval metrics for each class
    for class_idx in range(nc):
        n_cls_l = in_gt_class_labels[class_idx]
        class_preds = in_result_df.loc[in_result_df["pred_class"] == class_idx]
        sorted_class_df = class_preds.sort_values(by="pred_conf").copy().reset_index(drop=True)
        pred_arr = sorted_class_df[[el for el in sorted_class_df.columns if el.startswith("iou_match_")]].values
        # N_preds x N_metrics x N_iouvals
        confs = sorted_class_df["pred_conf"].values
        tps = pred_arr.cumsum(0)
        fps = (1 - pred_arr).cumsum(0)

        # Recall
        cls_rec = tps / (n_cls_l + 1e-7)
        cls_pre = tps/ (tps + fps + 1e-7)
        if len(confs) > 0:
            r[class_idx] = np.interp(-px, -confs, cls_rec[:, iou_idx], left=0)
            p[class_idx] = np.interp(-px, -confs, cls_pre[:, iou_idx], left=1)
        else:
            r[class_idx] = np.zeros((1000,))
            p[class_idx] = np.zeros((1000,))

        # AP from recall-precision curve
        for j in range(len(iou_vals)):
            aps[class_idx, j], mpre, mrec = compute_ap(cls_rec[:, j], cls_pre[:, j])
            if j == iou_idx:
                py.append(np.interp(px, mrec, mpre))  # values for PR curve

    # Compute F1, get optimal confidence threshold, return p/r/f1 at optimal vals.
    f1 = 2 * p * r / (p + r + 1e-6)
    metric_curves = {
        "pre": p.copy(),
        "rec": r.copy(),
        "f1": f1.copy(),
        "px": px.copy(),
        "py": py.copy(),
    }

    smoothed_f1 = np.apply_along_axis(lambda x: smooth(x, 0.1), 1, f1)
    max_f1_idxes = smoothed_f1.argmax(1)

    classwise_metrics = pd.DataFrame({
        "class_name": class_names,
        "support": in_gt_class_labels,
        "opt_conf": np.asarray([px[max_f1_idxes[i]] for i in range(len(max_f1_idxes))]),
        "f1": np.asarray([f1[i, max_f1_idxes[i]] for i in range(len(max_f1_idxes))]),
        "pre": np.asarray([p[i, max_f1_idxes[i]] for i in range(len(max_f1_idxes))]),
        "rec": np.asarray([r[i, max_f1_idxes[i]] for i in range(len(max_f1_idxes))]),
        "ap50": aps[:, 0],
        "ap95": aps.mean(1)
    })

    return classwise_metrics, metric_curves


def evaluate_detections(detections: torch.Tensor, labels: torch.Tensor, in_ious: list[float]) -> pd.DataFrame:
    """
    Return dataframe with per-detection evaluation results for an input images
    Arguments:
        detections (tensor[N, 6]), x1, y1, x2, y2, conf, class
        labels (tensor[M, 5]), class, x1, y1, x2, y2
        in_ious (list): list of iou levels to evaluate
    Returns:
        Pandas DF with per-detection results at different iou levels for this image
    """
    gt_bboxes = labels[:, 1:]
    pred_bboxes = detections[:, :4]
    all_ious = batch_torch_bbox_iou(gt_bboxes, pred_bboxes).cpu().numpy()
    all_class_matches = (labels[:, 0:1] == detections[:, 5]).cpu().numpy()

    # Initialize output df
    res_cols = ["pred_conf", "pred_class"] + ["iou_match_{}".format(round(in_ious[i], 2)) for i in range(len(in_ious))]
    results_df = pd.DataFrame(columns=res_cols, data=np.zeros((detections.shape[0], len(res_cols))).astype(int))
    results_df['pred_conf'] = detections[:, 4].cpu().numpy()
    results_df['pred_class'] = detections[:, 5].cpu().numpy()

    # Compute per-detection results at different IOU thresholds
    for i in range(len(in_ious)):
        cls_iou_matches = np.where((all_ious >= in_ious[i]) & all_class_matches)
        if cls_iou_matches[0].shape[0] > 0:
            unique_cls_iou_matches = get_unique_matches(cls_iou_matches, all_ious)
            results_df.loc[unique_cls_iou_matches[:, 1], 'iou_match_{}'.format(round(in_ious[i], 2))] = 1

            # Check for previous weird case
            for j in range(i):
                assert results_df.loc[unique_cls_iou_matches[:, 1], 'iou_match_{}'.format(round(in_ious[j], 2))].all()

    return results_df


def compute_conf_mat(detections, labels, iou_thresh, conf_thresh, n_classes) -> np.ndarray:
    """
    Return dataframe with per-detection evaluation results for an input images
    Arguments:
        detections (tensor[N, 6]), x1, y1, x2, y2, conf, class
        labels (tensor[M, 5]), class, x1, y1, x2, y2
        iou_thresh (float): IOU threshold for associating detections and labels
        conf_thresh (float): confidence threshold for considering detection
    Returns:
        (pd.DataFrame) Confusion matrix for detections and labels in this image

    """
    confident_preds = detections[(detections[:, 4] >= conf_thresh), :]
    conf_pred_bboxes = confident_preds[:, :4]
    gt_bboxes = labels[:, 1:]
    all_ious = batch_torch_bbox_iou(gt_bboxes, conf_pred_bboxes).cpu().numpy()
    all_class_matches = (labels[:, 0:1] == detections[:, 5]).cpu().numpy()
    # Compute confusion matrix for this set of detections
    iou_matches = np.where((all_ious >= iou_thresh))
    unique_iou_matches = get_unique_matches(iou_matches, all_ious)
    match_classes = torch.cat((confident_preds[unique_iou_matches[:, 1], 5:6], labels[unique_iou_matches[:, 0], 0:1]),
                              1)

    conf_mat = np.zeros((n_classes + 1, n_classes + 1))
    for i in range(n_classes):
        n_l = len(torch.where(labels[:, 0] == i)[0])
        n_p = len(torch.where(confident_preds[:, 5] == i)[0])
        for j in range(i, n_classes):
            # conf_mat[pred, actual]
            ij = ((match_classes[:, 0] == i) & (match_classes[:, 1] == j)).sum().item()
            ji = ((match_classes[:, 0] == j) & (match_classes[:, 1] == i)).sum().item()
            conf_mat[i, j] = ij
            conf_mat[j, i] = ji

        conf_mat[i, -1] = n_p - conf_mat[i, :].sum()
        conf_mat[-1, i] = n_l - conf_mat[:, i].sum()

    return conf_mat.astype(int)


def get_unique_matches(in_iou_matches: np.ndarray, in_iou_mat: np.ndarray) -> np.ndarray:
    """
    Converts matched indices in IOU matrix to list of unique matches, sorted by IOU
    Args:
        in_iou_matches (np.ndarray): indices representing matches between predictions & labels
        in_iou_mat (np.ndarray): matrix of all IOUs between predictions and labels

    Returns:
        (np.ndarray) Array representing

    """
    match_ious = in_iou_mat[in_iou_matches[0], in_iou_matches[1]]
    matches = np.concatenate((in_iou_matches[0][:, None], in_iou_matches[1][:, None], match_ious[:, None]), axis=1)
    if in_iou_matches[0].shape[0] > 1:
        matches = matches[matches[:, 2].argsort()[::-1]]
        matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
        matches = matches[matches[:, 2].argsort()[::-1]]
        matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
    return matches


def batch_torch_bbox_iou(boxes1: torch.Tensor, boxes2: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    As per:  https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    Arguments:
        boxes1 (Tensor[N, 4])
        boxes2 (Tensor[M, 4])
        eps (float)
    Returns:
        NxM matrix containing the pairwise IoU values for every element in boxes1 and boxes2
    """

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    (a1, a2), (b1, b2) = boxes1.unsqueeze(1).chunk(2, 2), boxes2.unsqueeze(0).chunk(2, 2)
    inter = (torch.min(a2, b2) - torch.max(a1, b1)).clamp(0).prod(2)

    # IoU = inter / (area1 + area2 - inter)
    return inter / ((a2 - a1).prod(2) + (b2 - b1).prod(2) - inter + eps)


def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves
    # Arguments
        recall:    The recall curve (list)
        precision: The precision curve (list)
    # Returns
        Average precision, precision curve, recall curve
    """

    # Append sentinel values to beginning and end
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))

    # Compute the precision envelope
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

    # Integrate area under curve
    method = 'interp'  # methods: 'continuous', 'interp'
    if method == 'interp':
        x = np.linspace(0, 1, 101)  # 101-point interp (COCO)
        ap = np.trapz(np.interp(x, mrec, mpre), x)  # integrate
    else:  # 'continuous'
        i = np.where(mrec[1:] != mrec[:-1])[0]  # points where x axis (recall) changes
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])  # area under curve

    return ap, mpre, mrec


def smooth(y, f=0.05):
    # Box filter of fraction f
    nf = round(len(y) * f * 2) // 2 + 1  # number of filter elements (must be odd)
    p = np.ones(nf // 2)  # ones padding
    yp = np.concatenate((p * y[0], y, p * y[-1]), 0)  # y padded
    return np.convolve(yp, np.ones(nf) / nf, mode='valid')  # y-smoothed


def compute_fitness(in_metrics_df: pd.DataFrame, in_metric_weights: list = (0., 0., 0.5, 0., 0.5)) -> float:
    mp = (in_metrics_df["pre"] * in_metrics_df["support"]).sum() / (in_metrics_df["support"]).sum()
    mr = (in_metrics_df["rec"] * in_metrics_df["support"]).sum() / (in_metrics_df["support"]).sum()
    mf1 = (in_metrics_df["f1"] * in_metrics_df["support"]).sum() / (in_metrics_df["support"]).sum()
    map50 = (in_metrics_df["ap50"] * in_metrics_df["support"]).sum() / (in_metrics_df["support"]).sum()
    map95 = (in_metrics_df["ap95"] * in_metrics_df["support"]).sum() / (in_metrics_df["support"]).sum()

    return (np.asarray([in_metric_weights]) @ np.asarray([mp, mr, mf1, map50, map95])).item()
