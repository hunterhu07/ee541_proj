#!/usr/bin/env python
# coding=utf-8
import sys
import os
from optparse import OptionParser
import numpy as np
import torch
import torch.nn as nn

from model import ResUNet, UNet, make_checkpoint_dir, make_test_dataloaders_sart, plot_net_predictions
from copy import deepcopy
import warnings
warnings.filterwarnings('ignore')


def get_args():
    parser = OptionParser()
    parser.add_option('-b', '--batch-size', dest='batchsize', default=8,
                      type='int', help='batch size')
    parser.add_option('-c', '--load', dest='load',
                      default='./model/CP99.pth',
                      help='load model')
    parser.add_option('-i', '--init', dest='init',
                      default=False, help='init the model')
    parser.add_option('-t', '--test-path', dest='testpath',
                      default='./data/sartorius/test/', help='folder of test images')
    parser.add_option('-o', '--output-path', dest='outputpath',
                      default='./data/sartorius/submission.csv', help='the path of the submission file')

    (options, args) = parser.parse_args()
    return options


def predict(net, device, data_loader, batch_size, output_path, save_fig=False):
    net.eval()
    output_lines = []
    max_len = len(data_loader)
    last_len = batch_size
    with torch.no_grad():
        for i, sample_batch in enumerate(data_loader):
            imgs = sample_batch['image']
            ids = sample_batch['id']
            imgs = imgs.to(device)
            outputs, output_res = net(imgs)
            probs = torch.softmax(outputs, dim=1)
            masks_pred = torch.argmax(probs, dim=1)
            masks_pred_lst = masks_pred.detach().numpy()#.tolist()
            if i == max_len - 1:
                last_len = sample_batch['last_len'].detach().numpy().tolist()[0]
                masks_pred_lst = masks_pred_lst[:last_len]
            else:
                masks_pred_lst = masks_pred_lst
            masks = []
            for n, mask in enumerate(masks_pred_lst):
                mask, c_t = single_cell(mask)
                masks.append(mask)
            masks = np.array(masks)
            if save_fig:
                masks_tensor = torch.from_numpy(np.array(masks))
                figg = plot_net_predictions(imgs, masks_tensor, masks_tensor, last_len)
                figg.savefig('./pics/reu_test_7.png')
            for n, mask in enumerate(masks):
                run_lengths = encoding(mask)
                for run_length in run_lengths:
                    output_lines.append(f"{ids[n]},{run_length}")
    submission_csv(output_lines, output_path)


def single_cell(mask):
    count = []
    types = [1, 2, 3]  # 1:Astro, 2:Cort, 3:Shy2sh
    for v in types:
        count.append(np.sum(mask == v))
    cell_type = types[np.argmax(count)]
    mask[mask != cell_type] = 0
    return mask, cell_type


def neighbour(node, mask):
    x, y = node
    neighbours = []
    xs = [-1, 0, 1]
    ys = [-1, 0, 1]
    shape_1 = mask.shape[0]
    shape_2 = mask.shape[1]
    for xi in xs:
        for yi in ys:
            xn = x + xi
            yn = y + yi
            if xn < 0 or yn < 0 or xn >= shape_1 or yn >=shape_2:
                continue
            if mask[xn][yn] != 0:
                neighbours.append((xn, yn))
    return neighbours


def pos(i, j, W=704, H=520):
    return i*W+j+1


def encoding(mask):
    shape_1 = mask.shape[0]
    shape_2 = mask.shape[1]
    m_stack = []
    m_dict = dict()
    cells_group = []

    for i in range(shape_1):
        for j in range(shape_2):
            if mask[i][j] != 0:
                m_stack.append((i, j))
                m_dict[(i, j)] = 1
                tmp_group = []
                while True:
                    if m_stack == []:
                        break
                    node = m_stack[-1]
                    m_stack = m_stack[:-1]
                    del m_dict[node]
                    if mask[node[0]][node[1]] == 0:
                        continue
                    mask[node[0]][node[1]] = 0
                    tmp_group.append(node)
                    next_nodes = neighbour(node, mask)
                    for next_node in next_nodes:
                        if next_node not in m_dict:
                            m_stack.append(next_node)
                            m_dict[next_node] = 1
                cells_group.append(tmp_group)
    # print(cells_group)
    run_length = []
    for cells in cells_group:
        if len(cells) == 1:
            run_length.append(f'{pos(cells[0][0],cells[0][1],shape_2)} {1}')
            continue
        rl_str = ""
        cells = sorted(cells)
        last_i = cells[0][0]
        last_j = cells[0][1]
        last_run = pos(cells[0][0], cells[0][1], shape_2)
        last_len = 1
        for cell in cells[1:]:
            if cell[0] == last_i:
                if cell[1] == last_j + 1:
                    last_len += 1
                    last_j = cell[1]
                else:
                    rl_str += f'{last_run} {last_len} '
                    last_i = cell[0]
                    last_j = cell[1]
                    last_run = pos(cell[0], cell[1], shape_2)
                    last_len = 1
            else:
                rl_str += f'{last_run} {last_len} '
                last_i = cell[0]
                last_j = cell[1]
                last_run = pos(cell[0], cell[1], shape_2)
                last_len = 1
        rl_str += f'{last_run} {last_len}'
        run_length.append(rl_str)

    return run_length


def encoding_test(X1=520, X2=704):
    xray = np.random.rand(X1*X2).reshape(X1, X2)
    xray[xray > 0.5] = 1
    xray[xray <= 0.5] = 0
    # xray = np.array([0,1,1,1,0,0,0,0,1,1,1,1]).reshape(3,4)
    xray = xray.astype(int)
    print(xray)
    import time
    t1 = time.time()
    outp = encoding(xray)
    t2 = time.time()
    print(f"time={t2-t1}")
    # for group in outp:
    #     gg = np.zeros((X1, X2), dtype=np.uint8)
    #     for cell in group:
    #         gg[cell[0]][cell[1]] = 1
    #     print("-----")
    #     print(gg)
    print("-----")

def submission_csv(lines, path):
    with open(path, 'w') as f:
        f.write('id,predicted\n')
        for line in lines:
            f.write(line+"\n")
    return

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

    dir_test = args.testpath
    dir_output = args.outputpath
    params = {'batch_size': args.batchsize, 'shuffle': False, 'num_workers': 4}

    test_loader = make_test_dataloaders_sart(dir_test, params)

    net = ResUNet(n_channels=3, n_classes=4)
    if args.init:
        net = load_trained_weight(net)

    net.to(device)

    if args.load:
        net.load_state_dict(torch.load(args.load, map_location=device))
        print('Model loaded from {}'.format(args.load))

    if args.init:
        torch.save(net.state_dict(), './checkpoints/res-unet_init.pth')

    # images model in parallel on multiple-GPUs
    if torch.cuda.device_count() > 1:
        print("Model training on", torch.cuda.device_count(), "GPUs")
        net = nn.DataParallel(net)

    predict(net, device, test_loader, batch_size=args.batchsize, output_path=dir_output, save_fig=False)
