import logging
import os
from collections import OrderedDict

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultTrainer, default_setup, launch, default_argument_parser
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
    verify_results,
)
from detectron2.modeling import GeneralizedRCNNWithTTA
import torch
from detectron2.data.detection_utils import _apply_exif_orientation, convert_PIL_to_numpy
from detectron2.data import transforms as T
import torchvision
import numpy as np


def build_evaluator(cfg, dataset_name, output_folder=None):
    """
    Create evaluator(s) for a given dataset.
    This uses the special metadata "evaluator_type" associated with each builtin dataset.
    For your own dataset, you can simply create an evaluator manually in your
    script and do not have to worry about the hacky if-else logic here.
    """
    if output_folder is None:
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
    evaluator_list = []
    evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
    if evaluator_type in ["sem_seg", "coco_panoptic_seg"]:
        evaluator_list.append(
            SemSegEvaluator(
                dataset_name,
                distributed=True,
                output_dir=output_folder,
            )
        )
    if evaluator_type in ["coco", "coco_panoptic_seg"]:
        evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder))
    if evaluator_type == "coco_panoptic_seg":
        evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
    if evaluator_type == "cityscapes_instance":
        return CityscapesInstanceEvaluator(dataset_name)
    if evaluator_type == "cityscapes_sem_seg":
        return CityscapesSemSegEvaluator(dataset_name)
    elif evaluator_type == "pascal_voc":
        return PascalVOCDetectionEvaluator(dataset_name)
    elif evaluator_type == "lvis":
        return LVISEvaluator(dataset_name, output_dir=output_folder)
    if len(evaluator_list) == 0:
        raise NotImplementedError(
            "no Evaluator for the dataset {} with the type {}".format(dataset_name, evaluator_type)
        )
    elif len(evaluator_list) == 1:
        return evaluator_list[0]
    return DatasetEvaluators(evaluator_list)


class Trainer(DefaultTrainer):
    """
    We use the "DefaultTrainer" which contains pre-defined default logic for
    standard training workflow. They may not work for you, especially if you
    are working on a new research project. In that case you can write your
    own training loop. You can use "tools/plain_train_net.py" as an example.
    """

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        return build_evaluator(cfg, dataset_name, output_folder)

    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("detectron2.trainer")
        # In the end of training, run an evaluation with TTA
        # Only support some R-CNN models.
        logger.info("Running inference with test-time augmentation ...")
        model = GeneralizedRCNNWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
            )
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res


def setup():
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file("supervised_final_25.2/supervised_evaluation.yaml")
    cfg.freeze()
    return cfg


class my_model(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, batched_inputs):
        # preprocess
        new_batch = []
        for x in batched_inputs:
            trans = torchvision.transforms.ToPILImage()
            image = trans(x)
            image = _apply_exif_orientation(image)
            image = convert_PIL_to_numpy(image, "BGR")

            aug_input = T.AugInput(image, sem_seg=None)
            augmentation = [T.ResizeShortestEdge(800, 1333, 'choice')]
            augs = T.AugmentationList(augmentation)
            transforms = augs(aug_input)
            image, sem_seg_gt = aug_input.image, aug_input.sem_seg

            image = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
            d = {"height": x.shape[1], "width": x.shape[2], "image": image}
            new_batch.append(d)
        batched_inputs = new_batch

        predictions = self.model(batched_inputs)
        
        # postprocess
        pred = {}
        if len(predictions[0]['instances'].get('pred_boxes')) != 0:
            pred["boxes"] = predictions[0]['instances'].get('pred_boxes').tensor
            pred["scores"] = predictions[0]['instances'].get('scores')
            pred["labels"] = predictions[0]['instances'].get('pred_classes')

        return [pred]
        
def get_model():
    print("Get Model!")
    cfg = setup()
    model = Trainer.build_model(cfg)
    DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=False
        )

    return my_model(model)
