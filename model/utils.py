#!/usr/bin/env python
# coding=utf-8
from pathlib import Path
# import shutil
import matplotlib.pyplot as plt
import numpy as np

LABEL_TO_COLOR = {0: [0, 0, 0], 1: [255, 0, 0], 2: [0, 255, 0], 3: [0, 0, 255]}


def mask2rgb(mask):
    rgb = np.zeros(mask.shape + (3,), dtype=np.uint8)

    for i in np.unique(mask):
        rgb[mask == i] = LABEL_TO_COLOR[i]

    return rgb


def rgb2mask(rgb):
    mask = np.zeros((rgb.shape[0], rgb.shape[1]))

    for k, v in LABEL_TO_COLOR.items():
        mask[np.all(rgb == v, axis=2)] = k

    return mask


def plot_net_predictions(imgs, true_masks, masks_pred, batch_size):
    fig, ax = plt.subplots(3, batch_size, figsize=(20, 15))

    for i in range(batch_size):
        img = np.transpose(imgs[i].squeeze().cpu().detach().numpy(), (1, 2, 0))
        mask_pred = masks_pred[i].cpu().detach().numpy()
        mask_true = true_masks[i].cpu().detach().numpy()

        ax[0, i].imshow(img)
        ax[1, i].imshow(mask2rgb(mask_pred))
        ax[1, i].set_title('Predicted')
        ax[2, i].imshow(mask2rgb(mask_true))
        ax[2, i].set_title('Ground truth')

    return fig


def make_checkpoint_dir(dir_checkpoint):
        
    path = Path(dir_checkpoint)
    # not remove folder if it exists
    if path.exists():
        return
        # shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=False)