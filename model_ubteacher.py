
from detectron2.config import get_cfg
from ubteacher import add_ubteacher_config
from detectron2.checkpoint import DetectionCheckpointer
from ubteacher.modeling import *
from ubteacher.engine import *
import ubteacher.data.datasets.builtin
import torch
from detectron2.data.detection_utils import _apply_exif_orientation, convert_PIL_to_numpy
from detectron2.data import transforms as T
import torchvision
import numpy as np


config_file = "semi_supervise_final_24.7/semi_supervised_evaluation.yaml"
# Selena: "semi_supervise_final_24.7/semi_supervised_evaluation.yaml"
# Caesar: "configs/Faster-RCNN/nyu/semi_supervised_evaluation.yaml"
def setup():
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_ubteacher_config(cfg)
    cfg.merge_from_file(config_file)
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
    cfg = setup()
    if cfg.SEMISUPNET.Trainer == "ubteacher":
        Trainer = UBTeacherTrainer
    elif cfg.SEMISUPNET.Trainer == "ubteacher_rcnn":
        Trainer = UBRCNNTeacherTrainer

    if cfg.SEMISUPNET.Trainer == "ubteacher":
        model = Trainer.build_model(cfg)
        model_teacher = Trainer.build_model(cfg)
        ensem_ts_model = EnsembleTSModel(model_teacher, model)

        DetectionCheckpointer(
            ensem_ts_model, save_dir=cfg.OUTPUT_DIR
        ).resume_or_load(cfg.MODEL.WEIGHTS, resume=False)

    else:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=False
        )

    return my_model(model)