import os
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

import utils.file_utils as fu
import numpy as np

import datasets.dataset3D60 as dataset3D60
import datasets.structured3d as sturctured3d

from utils.losses import *
import tqdm

import vgg_multiscale_network

from Metrics.depth_metrics import Depth_Evaluator
from Metrics.norm_metrics import Norm_Evaluator_Valid

def main():
    torch.cuda.empty_cache()

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--test-batch-size', type=int, default=1, metavar='N',
                        help='input batch size for testing')
    parser.add_argument('--seed', type=int, default=42, metavar='S',
                        help='random seed')

    args = parser.parse_args()
    use_cuda = torch.cuda.is_available()
    print("use_cuda:", use_cuda)
    torch.manual_seed(args.seed)
    device = torch.device('cuda' if use_cuda else 'cpu')
    kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}
    np.random.seed(args.seed)


    # test_set = dataset3D60.Dataset3D60('./datasets/3d60_test.txt', 256, 512)
    # test_set = dataset3D60.Dataset3D60('./datasets/stanford2d3d_test.txt', 256, 512)
    # test_set = dataset3D60.Dataset3D60('./datasets/Matterport3D_test.txt', 256, 512)
    # test_set = dataset3D60.Dataset3D60('./datasets/SunCG_test.txt', 256, 512)
    test_set = sturctured3d.Structured3d('./datasets/structured3d_test.txt', 256, 512)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.test_batch_size, shuffle=False, drop_last=False, **kwargs)

    model = vgg_multiscale_network.DeepNet()

    if torch.cuda.is_available():
        model.cuda()
                                                
    checkpoint = torch.load(model_path,map_location='cuda:0')
    model.load_state_dict(checkpoint['model'])

    start_epoch = checkpoint['epoch']
    print('loading epoch {} successfully！'.format(start_epoch))

    norm_evaluator_valid = Norm_Evaluator_Valid()
    norm_evaluator_valid.reset_eval_metrics()

    depth_evaluator_valid = Depth_Evaluator()
    depth_evaluator_valid.reset_eval_metrics()

    pbar = tqdm.tqdm(test_loader)
    pbar.set_description("Evaluating")

    model.eval()
    with torch.no_grad():
        for batch_idx, test_sample in enumerate(pbar):
            rgb = test_sample["ori_rgb"].to(device)
            norm = test_sample["gt_surface"]
            depth = test_sample["gt_depth"]
            mask = test_sample["mask"]
            mask = mask>0

            outputs = model(rgb)

            output_depth = outputs["pred_depth"].detach().cpu()*mask
            output_depth = torch.median(depth) / torch.median(output_depth) * output_depth

            output_norm = outputs["pred_normal"]
            output_norm = F.normalize(output_norm, p = 2, dim = 1).detach().cpu()*mask

            norm_evaluator_valid.compute_eval_metrics(norm, output_norm, mask, full_region=False)
            depth_evaluator_valid.compute_eval_metrics(depth[mask], output_depth[mask], full_region=False)

            fu.save_norm_tensor_as_float(output_path, test_sample['surface_filename'][0]+'_rgb', test_sample["aug_rgb"][0])
            fu.save_norm_tensor_as_float(output_path, test_sample['surface_filename'][0]+'_pred', output_norm[0])
            fu.save_norm_tensor_as_float(output_path, test_sample['surface_filename'][0]+'_gt', norm[0])
            fu.save_depth_tensor_as_float(output_path, test_sample['depth_filename'][0]+'_pred', output_depth[0])
            fu.save_depth_tensor_as_float(output_path, test_sample['depth_filename'][0]+'_gt', depth[0])

    depth_evaluator_valid.print(eva_path)
    norm_evaluator_valid.print(eva_path)


if __name__ == "__main__":
    model_path = './saved_models/models/pretrained_model.pkl'
    print(model_path)

    eva_path = './results/'

    output_path = './results/3d60_results/'
    # output_path = './results/structured3d_results/'

    if not os.path.exists(output_path):
        os.mkdir(output_path)
    if not os.path.exists(eva_path):
        os.mkdir(eva_path)
    main()