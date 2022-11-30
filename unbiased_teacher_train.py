#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import default_argument_parser, default_setup, launch

# hacky way to register
from ubteacher.modeling import *
from ubteacher.engine import *
from ubteacher import add_ubteacher_config


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_ubteacher_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)
    if cfg.SEMISUPNET.Trainer == "ubteacher":
        Trainer = UBTeacherTrainer
    elif cfg.SEMISUPNET.Trainer == "ubteacher_rcnn":
        Trainer = UBRCNNTeacherTrainer
    else:
        raise ValueError("Trainer Name is not found.")

    if args.eval_only:
        if cfg.SEMISUPNET.Trainer == "ubteacher":
            model = Trainer.build_model(cfg)
            model_teacher = Trainer.build_model(cfg)
            ensem_ts_model = EnsembleTSModel(model_teacher, model)

            DetectionCheckpointer(
                ensem_ts_model, save_dir=cfg.OUTPUT_DIR
            ).resume_or_load(cfg.MODEL.WEIGHTS, resume=args.resume)
            res = Trainer.test(cfg, ensem_ts_model.modelTeacher)

        else:
            model = Trainer.build_model(cfg)
            DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
                cfg.MODEL.WEIGHTS, resume=args.resume
            )
            res = Trainer.test(cfg, model)
        return res

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)

    # import pickle
    # import torch
    # with open("output/model_supervised_25.2.pkl", 'rb') as f:
    #     myModel = pickle.load(f)
    # # -----------------------------------------------------------------------------------------
    # for w in myModel['model']:
    #     if w not in ['modelTeacher.roi_heads.box_predictor.bbox_pred.weight', 'modelTeacher.roi_heads.box_predictor.bbox_pred.bias'] and w in trainer.checkpointer.model.state_dict():
    #         trainer.checkpointer.model.state_dict()[w] = torch.from_numpy(myModel['model'][w])
    #         print('exchanged', w)

    return trainer.train()


from detectron2.data.datasets import register_coco_instances

register_coco_instances("nyu_train", {}, "/data/sbcaesar/nyu/labeled_data/annotation/labeled_train.json",
                        "/data/sbcaesar/nyu/labeled_data/train2017")
register_coco_instances("nyu_val", {}, "/data/sbcaesar/nyu/labeled_data/annotation/labeled_val.json",
                        "/data/sbcaesar/nyu/labeled_data/val2017")
register_coco_instances("coco_train", {}, "/data/sbcaesar/unbiased-teacher-v2/datasets/coco/annotations/instances_train2017.json",
                        "/data/sbcaesar/unbiased-teacher-v2/datasets/coco/train2017")
register_coco_instances("coco_val", {}, "/data/sbcaesar/unbiased-teacher-v2/datasets/coco/annotations/instances_val2017.json",
                        "/data/sbcaesar/unbiased-teacher-v2/datasets/coco/val2017")

# Must needed
import ubteacher.data.datasets.builtin

if __name__ == "__main__":
    args = default_argument_parser().parse_args()

    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
