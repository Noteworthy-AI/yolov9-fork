import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from . import threaded
from .object_det_eval import smooth


def format_conf_mat_df(in_conf_mat: np.ndarray, cls_names: dict, normalize: bool = True):
    cm_data = in_conf_mat / ((in_conf_mat.sum(0).reshape(1, -1) + 1E-9) if normalize else 1)  # normalize columns
    cm_data[cm_data < 0.005] = np.nan  # don't annotate (would appear as 0.00)
    name_arr = [cls_names[i] for i in range(len(cls_names.keys()))] + ["background"]
    out_df = pd.DataFrame(cm_data, columns=name_arr, index=name_arr).round(3)
    return out_df


@threaded
def plot_confusion_matrix(in_conf_mat_df: pd.DataFrame, save_path: str):
    fig, ax = plt.subplots(1, 1, figsize=(12, 9), tight_layout=True)
    sns.set(font_scale=1.4)  # for label size
    sns.heatmap(
        in_conf_mat_df,
        ax=ax,
        annot=(len(in_conf_mat_df) < 30),
        cmap="Blues",
        annot_kws={"size": 16}
    )  # font size
    ax.set_ylabel('True')
    ax.set_ylabel('Predicted')
    ax.set_title('Confusion Matrix')
    fig.savefig(save_path, dpi=250)
    plt.close(fig)


@threaded
def plot_pr_curve(px: np.ndarray, py: np.ndarray, ap50: np.ndarray, save_dir: str ='pr_curve.png', names: list = ()):
    # Precision-recall curve
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)
    py = np.stack(py, axis=1)

    if 0 < len(names) < 21:  # display per-class legend if < 21 classes
        for i, y in enumerate(py.T):
            ax.plot(px, y, linewidth=1, label=f'{names[i]} {ap50[i]:.3f}')  # plot(recall, precision)
    else:
        ax.plot(px, py, linewidth=1, color='grey')  # plot(recall, precision)

    ax.plot(px, py.mean(1), linewidth=3, color='blue', label='all classes %.3f mAP@0.5' % ap50.mean())
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    ax.set_title('Precision-Recall Curve')
    fig.savefig(save_dir, dpi=250)
    plt.close(fig)


@threaded
def plot_mc_curve(px, py, save_dir='mc_curve.png', names=(), xlabel='Confidence', ylabel='Metric'):
    # Metric-confidence curve
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)

    if 0 < len(names) < 21:  # display per-class legend if < 21 classes
        for i, y in enumerate(py):
            ax.plot(px, y, linewidth=1, label=f'{names[i]}')  # plot(confidence, metric)
    else:
        ax.plot(px, py.T, linewidth=1, color='grey')  # plot(confidence, metric)

    y = smooth(py.mean(0), 0.05)
    ax.plot(px, y, linewidth=3, color='blue', label=f'all classes {y.max():.2f} at {px[y.argmax()]:.3f}')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    ax.set_title(f'{ylabel}-Confidence Curve')
    fig.savefig(save_dir, dpi=250)
    plt.close(fig)
