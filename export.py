import argparse
import logging
import os

import numpy as np
import torch
import torch.nn as nn

from unet import UNet
from utils.dataset import BasicDataset

def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', '-m', default='MODEL.pth',
                        metavar='FILE',
                        help="Specify the file in which the model is stored")
    return parser.parse_args()

def onnx_export(model: nn.Module, fn: str):
    # switch to evaluation mode
    labels = []
    # SJH: inputs hardcode based on known sizes
    dummy_input = torch.randn(1, 3, 640, 959)
    dummy_input = dummy_input.cuda(0)
    print(dummy_input.shape)
    torch.onnx.export(model, dummy_input, fn,
                      verbose=True,
                      input_names=['input'], output_names=['output'],
                      export_params=True, opset_version=10)

# SJH: This is related to this bug: https://github.com/pytorch/pytorch/issues/35516
                      # opset_version=12,
                      # do_constant_folding=True) # Try with 11

if __name__ == "__main__":
    args = get_args()

    net = UNet(n_channels=3, n_classes=1)

    logging.info("Loading model {}".format(args.model))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    net.to(device=device)
    net.load_state_dict(torch.load(args.model, map_location=device))

    logging.info("Model loaded !")

    fn = os.path.splitext(os.path.basename(args.model))[0] + ".onnx"
    print("Exporting as ONNX", fn)
    onnx_export(net, fn)
