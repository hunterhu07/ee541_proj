#!/usr/bin/env python
# coding=utf-8
import os
from glob import glob
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import torch
import pandas as pd
import cv2
from model.rand_trans import RandomResizedCrop

from model.utils import mask2rgb, rgb2mask


def cell_id(cell_type):
    if cell_type == "astro":
        return 0
    elif cell_type == "cort":
        return 1
    elif cell_type == "shsy5y":
        return 2
    else:
        print("WRONG CELL TYPE")
        return 0


def run_length_encode(anno, H=520, W=704, cell_type=1):
    mask = np.zeros(W * H, dtype=np.uint8)  # 15
    anno_num = len(anno)
    for cell_i in range(anno_num):
        c_list = anno[cell_i].split(" ")
        run = 0
        for i, v in enumerate(c_list):
            if v == '':
                continue
            if i % 2 == 0:
                run = int(v) - 1
            else:
                length = int(v)
                for li in range(length):
                    mask[run + li] = 255
    mask = mask.reshape(H, W)
    return mask


class TestDataset(Dataset):
    def __init__(self, path, batch_size=8, transforms=None):
        self.img_paths = glob(path + "*")
        self.transforms = transforms
        self.last_len = batch_size
        pad_len = len(self.img_paths) % batch_size
        if pad_len > 0:
            self.last_len = pad_len
            for p in range(batch_size - pad_len):
                self.img_paths.append(self.img_paths[-1])
        len(self.img_paths)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        # img = Image.open(img_path)

        # transform
        if self.transforms:
            rands = np.random.rand(5)
            scale = (rands[0] * 0.4 + 0.3, rands[0] * 0.4 + 0.3)
            seed = (rands[1], rands[2])
            img = RandomResizedCrop(224, scale=scale, ratio=(1.0, 1.0), seed=seed)(img)
            if rands[3] > 0.5:
                img = transforms.RandomHorizontalFlip(1)(img)
            if rands[4] > 0.5:
                img = transforms.RandomVerticalFlip(1)(img)

        img = np.array(img) / 255
        img = torch.from_numpy(img).permute(2, 0, 1).float()
        img = transforms.Normalize((0.456, 0.456, 0.456), (0.225, 0.225, 0.225))(img)

        img_id = img_path.split("/")[-1].split(".")[-2]
        sample = {'image': img, 'id': img_id, "last_len": self.last_len}
        return sample


class CellDataset(Dataset):

    def __init__(self, df, transforms=None):
        self.df = df
        self.img_paths = df['image_path'].values
        self.annotation = df['annotation'].values
        self.cell_type = df['cell_type'].values
        self.transforms = transforms

    def __len__(self):
        return len(self.df) * 5

    def __getitem__(self, index):
        # transform_id = int(index / len(self.df))
        index = int(index % len(self.df))
        img_path = self.img_paths[index]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        shape_1 = img.shape[0]
        shape_2 = img.shape[1]
        img = Image.fromarray(img)
        # img = Image.open(img_path)
        annotation = self.annotation[index]
        # annotation = ' '.join(self.annotation[index])
        cell_type = cell_id(self.cell_type[index])
        mask = run_length_encode(annotation, H=shape_1, W=shape_2, cell_type=cell_type)
        mask_rgb = [np.zeros(shape_1 * shape_2, dtype=np.uint8).reshape(shape_1, shape_2),
                    np.zeros(shape_1 * shape_2, dtype=np.uint8).reshape(shape_1, shape_2),
                    np.zeros(shape_1 * shape_2, dtype=np.uint8).reshape(shape_1, shape_2)]
        mask_rgb[cell_type] = mask
        mask = cv2.merge(mask_rgb)
        mask = Image.fromarray(mask)

        # transform
        if self.transforms:
            rands = np.random.rand(5)
            scale = (rands[0] * 0.4 + 0.3, rands[0] * 0.4 + 0.3)
            seed = (rands[1], rands[2])
            img = RandomResizedCrop(224, scale=scale, ratio=(1.0, 1.0), seed=seed)(img)
            mask = RandomResizedCrop(224, scale=scale, ratio=(1.0, 1.0), seed=seed)(mask)
            if rands[3] > 0.5:
                img = transforms.RandomHorizontalFlip(1)(img)
                mask = transforms.RandomHorizontalFlip(1)(mask)
            if rands[4] > 0.5:
                img = transforms.RandomVerticalFlip(1)(img)
                mask = transforms.RandomVerticalFlip(1)(mask)

        mask = rgb2mask(np.array(mask))
        mask = torch.from_numpy(mask).long()
        img = np.array(img) / 255
        img = torch.from_numpy(img).permute(2, 0, 1).float()
        img = transforms.Normalize((0.456, 0.456, 0.456), (0.225, 0.225, 0.225))(img)

        sample = {'image': img, 'mask': mask}
        return sample


class ResDataset(Dataset):

    def __init__(self, df, transforms=None):
        self.df = df
        self.img_paths = df['image_path'].values
        self.cell_type = df['cell_type'].values
        self.transforms = transforms

    def __len__(self):
        return len(self.df) * 5

    def __getitem__(self, index):
        # transform_id = int(index / len(self.df))
        index = int(index % len(self.df))
        img_path = self.img_paths[index]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        # img = Image.open(img_path)
        # annotation = ' '.join(self.annotation[index])
        cell_type = cell_id(self.cell_type[index])
        if cell_type > 1:
            cell_type = 1

        # transform
        if self.transforms:
            rands = np.random.rand(5)
            scale = (rands[0] * 0.4 + 0.3, rands[0] * 0.4 + 0.3)
            seed = (rands[1], rands[2])
            img = RandomResizedCrop(224, scale=scale, ratio=(1.0, 1.0), seed=seed)(img)
            if rands[3] > 0.5:
                img = transforms.RandomHorizontalFlip(1)(img)
            if rands[4] > 0.5:
                img = transforms.RandomVerticalFlip(1)(img)
        img = np.array(img) / 255
        img = torch.from_numpy(img).permute(2, 0, 1).float()
        img = transforms.Normalize((0.456, 0.456, 0.456), (0.225, 0.225, 0.225))(img)

        sample = {'image': img, 'type': cell_type}
        return sample


def make_datasets_sart(path, k, k_no=0, astro=False):
    df = pd.read_csv(f'{path}/train.csv')
    df['image_path'] = path + '/images/' + df['id'] + '.png'
    tmp_df = df.drop_duplicates(subset=["id", "image_path", "cell_type"]).reset_index(drop=True)
    tmp_df["annotation"] = df.groupby("id")["annotation"].agg(list).reset_index(drop=True)
    df = tmp_df.copy()
    unit_length = int(len(df) / k)

    df_t = None
    for i in range(k):
        if i != k-1:
            if i == k_no:
                df_v = df[i*unit_length:(i+1)*unit_length]
            else:
                if df_t is None:
                    df_t = df[i*unit_length:(i+1)*unit_length]
                else:
                    df_t = pd.concat([df_t, df[i*unit_length:(i+1)*unit_length]], axis=0)
        else:
            if i == k_no:
                df_v = df[i*unit_length:]
            else:
                if df_t is None:
                    df_t = df[i*unit_length:]
                else:
                    df_t = pd.concat([df_t, df[i*unit_length:]], axis=0)

    train_length = int(len(df) * 0.9)
    df_t_o = df[:train_length]
    df_v_o = df[train_length:]
    train_dataset = CellDataset(df_t, transforms=True)
    val_dataset = CellDataset(df_v, transforms=True)
    return train_dataset, val_dataset


def make_dataloaders_sart(path, k, params, k_no=0):
    train_dataset, val_dataset = make_datasets_sart(path, k, k_no)
    train_loader = DataLoader(train_dataset, drop_last=True, **params)
    val_loader = DataLoader(val_dataset, drop_last=True, **params)

    return train_loader, val_loader


def make_test_dataloaders_sart(path, params):
    params['shuffle'] = False
    test_dataset = TestDataset(path)
    test_loader = DataLoader(test_dataset, drop_last=True, **params)
    return test_loader
