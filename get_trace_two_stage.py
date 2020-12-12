import argparse

from mmcv import Config

from mmdet.models import build_detector

import torch


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument(
        '--checkpoint', help='the checkpoint file to trace')
    parser.add_argument(
        '--tracedbone', help='the name of tracedpoint')
    parser.add_argument(
        '--tracedbbox', help='the name of tracedpoint')
    parser.add_argument(
        '--shape',
        type=int,
        nargs='+',
        default=[1280, 800],
        help='input image size')
    args = parser.parse_args()
    return args

def main():

    args = parse_args()

    if len(args.shape) == 1:
        input_shape = (1, 3, args.shape[0], args.shape[0])
    elif len(args.shape) == 2:
        input_shape = (1, 3, ) + tuple(args.shape)
    else:
        raise ValueError('invalid input shape')

    cfg = Config.fromfile(args.config)
    model = build_detector(
        cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg).cuda()

    if model.__class__.__name__.lower() == "fasterrcnn":
        model.forward = model.forward_trace_fasterrcnn
    else:
        print ("two stage trace only support fasterrcnn")
        return

    checkpoint = torch.load(args.checkpoint)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    print (input_shape)
    img = torch.rand(input_shape).cuda()
    traced_bone = torch.jit.trace(model, img)
    traced_bone.save(args.tracedbone)

    bbox_feats = torch.rand(1000, 256, 7, 7).cuda()
    traced_bbox = torch.jit.trace(model.roi_head.bbox_head, bbox_feats)
    traced_bbox.save(args.tracedbbox)





if __name__ == '__main__':
    main()