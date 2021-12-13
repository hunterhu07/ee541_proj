#!/usr/bin/env python
# coding=utf-8
import sys
import os
from optparse import OptionParser
import numpy as np
import time
import torch
import torch.nn as nn
from torch import optim
from tensorboardX import SummaryWriter

from model import ResUNet, UNet, eval_net_loader, make_checkpoint_dir, make_dataloaders_sart, plot_net_predictions
# from model import make_test_dataloaders_sart
from copy import deepcopy
import warnings
warnings.filterwarnings('ignore')


def train_epoch(epoch, train_loader, criterion, optimizer, batch_size, scheduler, summarize_step=50):
    net.train()
    epoch_loss = 0
    record_loss = 0
    for i, sample_batch in enumerate(train_loader):
        imgs = sample_batch['image']
        true_masks = sample_batch['mask']

        imgs = imgs.to(device)
        true_masks = true_masks.to(device)
        # true_masks_np = true_masks.detach().numpy()
        outputs, output_res = net(imgs)
        probs = torch.softmax(outputs, dim=1)
        masks_pred = torch.argmax(probs, dim=1)
        # probs_np = probs.detach().numpy()
        loss = criterion(outputs, true_masks)
        epoch_loss += loss.item()
        record_loss += loss.item()
        print(f'epoch = {epoch+1:d}, iteration = {i:d}/{len(train_loader):d}, loss = {loss.item():.5f}')
        # figg = plot_net_predictions(imgs, masks_pred, masks_pred, batch_size)
        # figg.savefig(f'./pics/reu_lg_{i}.png')
        # save to summary
        if i % summarize_step == 0:
            if i != 0:
                record_loss = record_loss / summarize_step
            print(
                f'epoch = {epoch + 1:d}, iteration = {i:d}/{len(train_loader):d}, loss = {record_loss:.5f}, time = {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}')
            writer.add_scalar('train_loss_iter', record_loss,
                              i + len(train_loader) * epoch)
            writer.add_figure('predictions vs. actuals',
                              plot_net_predictions(imgs, true_masks, masks_pred, batch_size),
                              global_step=i + len(train_loader) * epoch)
            record_loss = 0
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch finished ! Loss: {epoch_loss / i:.5f}, lr:{scheduler.get_lr()}')


def validate_epoch(epoch, train_loader, val_loader, device):
    class_iou, mean_iou = eval_net_loader(net, val_loader, 4,
                                          device)  # mean_iou = (class_iou[1] + class_iou[2] + class_iou[3])/3
    print('Class IoU:', ' '.join(f'{x:.3f}' for x in class_iou), f'  |  Mean IoU: {mean_iou:.3f}')
    # save to summary
    writer.add_scalar('mean_iou', mean_iou, len(train_loader) * (epoch + 1))

    return (class_iou[1] + class_iou[2] + class_iou[3])/3


def train_net(train_loader, val_loader, net, device, epochs=5, batch_size=1, lr=0.1, save_cp=True, start_epoch=0):
    print(f'''
    Starting training:
        Epochs: {epochs}
        Batch size: {batch_size}
        Learning rate: {lr}
        Training size: {len(train_loader.dataset)}
        Validation size: {len(val_loader.dataset)}
        Checkpoints: {str(save_cp)}
        Device: {str(device)}
    ''')

    down_params = []
    down_params += net.inc.parameters()
    down_params += net.down1.parameters()
    down_params += net.down2.parameters()
    down_params += net.down3.parameters()
    down_params += net.down4.parameters()
    up_params = []
    up_params += net.up1_0.parameters()
    up_params += net.up2_0.parameters()
    up_params += net.up3_0.parameters()
    up_params += net.up4_0.parameters()
    up_params += net.up1_1.parameters()
    up_params += net.up2_1.parameters()
    up_params += net.up3_1.parameters()
    up_params += net.up4_1.parameters()
    up_params += net.outc_0.parameters()
    up_params += net.outc_1.parameters()

    # optimizer = optim.SGD(net. parameters(),lr=lr, momentum=0.9, weight_decay=0.0005)
    optimizer = optim.AdamW([{'params': down_params, 'lr': lr * 0.5}, {'params': up_params, 'lr': lr}], lr=lr,
                            weight_decay=0.0005)
    # optimizer = optim.AdamW(net.parameters(), lr=lr, weight_decay=0.0005)
    # multiply learning rate by 0.1 after 30% of epochs
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=int(0.3 * (epochs-start_epoch)), gamma=0.1)

    criterion = nn.CrossEntropyLoss()
    best_epoch = -1
    best_precision = 0
    for epoch in range(start_epoch, epochs):
        improve = ""
        print('Starting epoch {}/{}.'.format(epoch + 1, epochs))
        # if epoch == start_epoch:
        #     precision = validate_epoch(epoch, train_loader, val_loader,device)
        train_epoch(epoch, train_loader, criterion, optimizer, batch_size, scheduler)
        precision = validate_epoch(epoch, train_loader, val_loader, device)
        scheduler.step()
        print(f'precision:{precision}, best_precision:{best_precision}({best_epoch})')
        if precision < best_precision:
            improve = "_low"
        else:
            best_precision = precision
            best_epoch = epoch
        if save_cp:
            state_dict = net.state_dict()
            torch.save(state_dict, dir_checkpoint + f'CP{epoch + 1}' + improve + '.pth')
            print('Checkpoint {} saved !'.format(epoch + 1))

    writer.close()


def get_args():
    parser = OptionParser()
    parser.add_option('-e', '--epochs', dest='epochs', default=100, type='int',
                      help='number of epochs')
    parser.add_option('-s', '--start-epochs', dest='start_epoch', default=0, type='int',
                      help='start number of epochs')
    parser.add_option('-b', '--batch-size', dest='batchsize', default=8,
                      type='int', help='batch size')
    parser.add_option('-l', '--learning-rate', dest='lr', default=0.00001,
                      type='float', help='learning rate')
    parser.add_option('-c', '--load', dest='load',
                      default='./model/CP99.pth',
                      help='the path to load model')
    parser.add_option('-i', '--init', dest='init',
                      default=False, help='init the model')
    parser.add_option('-f', '--folder', dest='folder',
                      default='sartorius', help='folder of train&validation images')
    parser.add_option('-k', '--k-fold', dest='k',
                      default=10, help='cross validation')
    parser.add_option('-n', '--k-fold_number', dest='k_n',
                      default=9, help='integer between [0,k)')

    (options, args) = parser.parse_args()
    return options


def load_trained_weight(net, resnet_path='./checkpoints/resnet.pth', unet0_path='./checkpoints/unet_a_decoder.pth',
                        unet1_path='./checkpoints/unet.pth'):
    net.resnet.load_state_dict(torch.load(resnet_path))
    unet0 = UNet(n_channels=3, n_classes=4)
    unet0.load_state_dict(torch.load(unet0_path))
    unet1 = UNet(n_channels=3, n_classes=4)
    unet1.load_state_dict(torch.load(unet1_path))  ### unet1
    net.inc = deepcopy(unet1.inc)
    net.down1 = deepcopy(unet1.down1)
    net.down2 = deepcopy(unet1.down2)
    net.down3 = deepcopy(unet1.down3)
    net.down4 = deepcopy(unet1.down4)
    net.up1_0 = deepcopy(unet0.up1)
    net.up2_0 = deepcopy(unet0.up2)
    net.up3_0 = deepcopy(unet0.up3)
    net.up4_0 = deepcopy(unet0.up4)
    net.outc_0 = deepcopy(unet0.outc)
    net.up1_1 = deepcopy(unet1.up1)
    net.up2_1 = deepcopy(unet1.up2)
    net.up3_1 = deepcopy(unet1.up3)
    net.up4_1 = deepcopy(unet1.up4)
    net.outc_1 = deepcopy(unet1.outc)
    return net


if __name__ == '__main__':

    device = ("cuda" if torch.cuda.is_available() else "cpu")
    args = get_args()

    dir_data = f'./data/{args.folder}'
    dir_checkpoint = f'./checkpoints/{args.folder}_res_unet_b{args.batchsize}/'
    dir_summary = f'./runs/{args.folder}_res_unet_b{args.batchsize}'
    params = {'batch_size': args.batchsize, 'shuffle': True, 'num_workers': 4}

    if not args.load:
        make_checkpoint_dir(dir_checkpoint)
    writer = SummaryWriter(dir_summary)

    k_no = int(args.k_n)
    if k_no < 0 or k_no >= args.k:
        k_no = args.k - 1
    train_loader, val_loader = make_dataloaders_sart(dir_data, args.k, params, k_no)

    net = ResUNet(n_channels=3, n_classes=4)
    if args.init:
        net = load_trained_weight(net)

    net.to(device)

    if args.load:
        net.load_state_dict(torch.load(args.load, map_location=device))
        print('Model loaded from {}'.format(args.load))

    for k, v in net.named_parameters():
        if 'res' in k:
            v.requires_grad = False

    #     for k, v in net.named_parameters():
    #         if 'down' in k or 'inc' in k:
    #             v.requires_grad = False

    for k, v in net.named_parameters():
        print(f'{k}: {v.requires_grad}')
    if args.init:
        torch.save(net.state_dict(), './checkpoints/res-unet_init.pth')

    # images model in parallel on multiple-GPUs
    if torch.cuda.device_count() > 1:
        print("Model training on", torch.cuda.device_count(), "GPUs")
        net = nn.DataParallel(net)

    try:
        # predict(test_loader)
        train_net(train_loader, val_loader, net, device, epochs=args.epochs, start_epoch=args.start_epoch,
                  batch_size=args.batchsize, lr=args.lr)

    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        print('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
