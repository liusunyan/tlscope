# Copyright (c) OpenMMLab. All rights reserved.
from argparse import ArgumentParser

import mmcv

import mmcv_custom   # noqa: F401,F403
import mmseg_custom   # noqa: F401,F403
from mmseg.apis import inference_segmentor, init_segmentor, show_result_pyplot
from mmseg.core.evaluation import get_palette
from mmcv.runner import load_checkpoint
from mmseg.core import get_classes
import cv2
import os.path as osp
import os
import pickle

def test_single_image(model, img_name, out_dir, color_palette, opacity):
    result = inference_segmentor(model, img_name)
    
    # show the results
    if hasattr(model, 'module'):
        model = model.module
    img = model.show_result(img_name, result,
                            palette=color_palette,
                            show=False, opacity=opacity)
    
    # save the results
    mmcv.mkdir_or_exist(out_dir)
    out_path = osp.join(out_dir, osp.basename(img_name))
    cv2.imwrite(out_path, img)
    print(f"Result is save at {out_path}")
    return img_name, result[0]


def main():
    parser = ArgumentParser()
    parser.add_argument('img', help='Image file or a directory contains images')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('--out', type=str, default="demo", help='out dir')
    parser.add_argument('--pkl_out', type=str, default="demo", help='out dir')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--palette',
        default='ade20k',
        choices=['ade20k', 'cityscapes', 'cocostuff', 'TLS','Tumor_necrosis'],
        help='Color palette used for segmentation map')
    parser.add_argument(
        '--opacity',
        type=float,
        default=0.5,
        help='Opacity of painted segmentation map. In (0, 1] range.')
    args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    model = init_segmentor(args.config, checkpoint=None, device=args.device)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
        print(f"1CLASSES is set to {model.CLASSES}")
    else:
        model.CLASSES = get_classes(args.palette)
        print(f"2CLASSES is set to {model.CLASSES}")

    # check arg.img is directory of a single image.
    if not os.path.exists(args.out):
        os.makedirs(args.out)
    if not os.path.exists(args.pkl_out):
        os.makedirs(args.pkl_out)
    if osp.isdir(args.img):
        for img in os.listdir(args.img):
            image_name,pixel_data = test_single_image(model, osp.join(args.img, img), args.out, get_palette(args.palette), args.opacity)
            # Save the pixel-level inference results as a pickle file
            basename = osp.basename(image_name)
            print(osp.join(args.pkl_out, basename + '.pkl'))
            with open(osp.join(args.pkl_out, basename + '.pkl'), 'wb') as f:
                pickle.dump(pixel_data, f)
    else:
        image_name,pixel_data = test_single_image(model, args.img, args.out, get_palette(args.palette), args.opacity)
        # Save the pixel-level inference results as a pickle file
        basename = osp.basename(image_name)
        print(osp.join(args.out, basename + '.pkl'))
        with open(osp.join(args.out, basename + '.pkl'), 'wb') as f:
            pickle.dump(pixel_data, f)
    print(f"Inference results saved in {args.out}")
    print(f"Inference results saved in {basename}")

if __name__ == '__main__':
    main()