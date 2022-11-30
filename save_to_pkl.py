import copy
import os
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.engine import default_argument_parser
import pickle
from detectron2.config import get_cfg
import logging
import os
from collections import OrderedDict

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, hooks, launch
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
from collections import OrderedDict
from ubteacher.engine import *
from ubteacher import add_ubteacher_config
from ubteacher.modeling import *





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


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    print(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def setup_semi(args):
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


def save_supervise_to_pkl(args):
    args.config_file = "configs/supervised-RCNN/supervised_evaluation.yaml"
    cfg = setup(args)

    model = Trainer.build_model(cfg)
    DetectionCheckpointer(model).resume_or_load(cfg.MODEL.WEIGHTS, resume=False)  # args.resume
    checkpointer = DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR)

    weights = OrderedDict()
    for p in checkpointer.model.state_dict():
        weights['modelTeacher.' + p] = checkpointer.model.state_dict()[p].cpu().numpy()
        weights['modelStudent.' + p] = checkpointer.model.state_dict()[p].cpu().numpy()

    output_path = "output/model_supervised_pretrained_teacher_and_student.pkl"
    with open(os.path.join(output_path), 'wb') as f:
        myModel = {'model': weights, '__author__': "Haowen Guan [haowen@nyu.edu]"}
        pickle.dump(myModel, f)
        print('!!! pkl model saved to', output_path, "!!!")


def save_supervise_to_semi_eval(args):
    args.config_file = "configs/supervised-RCNN/supervised_evaluation.yaml"
    cfg = setup(args)

    model = Trainer.build_model(cfg)
    DetectionCheckpointer(model).resume_or_load(cfg.MODEL.WEIGHTS, resume=False)  # args.resume
    checkpointer = DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR)

    # -----------------------------------------------------------------------------------------
    input_path = "output/model_unbias_eval.pkl"
    with open(os.path.join(input_path), 'rb') as f:
        myModel = pickle.load(f)
    # -----------------------------------------------------------------------------------------

    for p in checkpointer.model.state_dict():
        if p in myModel['model'] and p not in \
                ['roi_heads.box_predictor.bbox_pred.weight', 'roi_heads.box_predictor.bbox_pred.bias']:
            myModel['model'][p] = checkpointer.model.state_dict()[p].cpu().numpy()

    output_path = "output/model_super_unbias_eval.pkl"
    with open(os.path.join(output_path), 'wb') as f:
        pickle.dump(myModel, f)
        print('!!! pkl model saved to', output_path, "!!!")


def save_semi_to_supervise(args):
    semi_args = copy.deepcopy(args)
    args.config_file = "configs/supervised-RCNN/supervised_evaluation.yaml"
    cfg = setup(args)
    model = Trainer.build_model(cfg)
    DetectionCheckpointer(model).resume_or_load(cfg.MODEL.WEIGHTS, resume=False)  # args.resume
    checkpointer = DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR)

    weights = OrderedDict()
    for p in checkpointer.model.state_dict():
        if p.startswith('roi_heads.box_predictor.bbox_pred.weight') or p.startswith('roi_heads.box_predictor.bbox_pred.bias'):
            weights[p] = checkpointer.model.state_dict()[p].cpu().numpy()

    semi_args.config_file = "configs/Faster-RCNN/nyu/semi_supervised_evaluation.yaml"
    cfg_semi = setup_semi(semi_args)
    Trainer_semi = UBRCNNTeacherTrainer
    trainer_semi = Trainer_semi(cfg_semi)
    trainer_semi.resume_or_load(resume=args.resume)

    for p in trainer_semi.checkpointer.model.state_dict():
        if p.startswith('modelTeacher') and not p.startswith('modelTeacher.roi_heads.box_predictor.bbox_pred.bias') \
                and not p.startswith('modelTeacher.roi_heads.box_predictor.bbox_pred.weight'):
            weights[p[13:]] = trainer_semi.checkpointer.model.state_dict()[p].cpu().numpy()

    output_path = "output/model_semi_to_supervise.pkl"
    with open(os.path.join(output_path), 'wb') as f:
        myModel = {'model': weights, '__author__': "Haowen Guan [haowen@nyu.edu]"}
        pickle.dump(myModel, f)
        print('!!! pkl model saved to', output_path, "!!!")


def save_semi_eval(args):
    semi_args = args

    cfg_semi = setup_semi(semi_args)
    Trainer_semi = UBRCNNTeacherTrainer
    trainer_semi = Trainer_semi(cfg_semi)
    trainer_semi.resume_or_load(resume=args.resume)

    weights = OrderedDict()
    for p in trainer_semi.checkpointer.model.state_dict():
        if p.startswith('modelTeacher.'):
            weights[p[13:]] = trainer_semi.checkpointer.model.state_dict()[p].cpu().numpy()

    output_path = "output/model_unbias_eval.pkl"
    with open(os.path.join(output_path), 'wb') as f:
        myModel = {'model': weights, '__author__': "Haowen Guan [haowen@nyu.edu]"}
        pickle.dump(myModel, f)
        print('!!! pkl model saved to', output_path, "!!!")


from detectron2.data.datasets import register_coco_instances

register_coco_instances("nyu_train", {}, "/data/sbcaesar/nyu/labeled_data/annotation/labeled_train.json",
                        "/data/sbcaesar/nyu/labeled_data/train2017")
register_coco_instances("nyu_val", {}, "/data/sbcaesar/nyu/labeled_data/annotation/labeled_val.json",
                        "/data/sbcaesar/nyu/labeled_data/val2017")

# Must needed
import ubteacher.data.datasets.builtin

if __name__ == "__main__":
    # Remember to change the weight path in the below config files
    args = default_argument_parser()
    args.add_argument('--save_semi_pkl', action='store_true', help="Convert unbiased teacher to pkl for evaluation")
    args.add_argument('--save_supervise_pkl', action='store_true', help="Convert supervise model to pkl")
    args.add_argument('--convert_semi_to_supervise', action='store_true', help="Convert semi model back to supervise")
    args = args.parse_args()

    if args.save_supervise_pkl:
        args.config_file = "configs/supervised-RCNN/supervised_evaluation.yaml"
        save_supervise_to_pkl(args)
    elif args.save_semi_pkl:
        args.config_file = "configs/Faster-RCNN/nyu/semi_supervised_evaluation.yaml"
        save_semi_eval(args)
    elif args.convert_semi_to_supervise:
        save_semi_to_supervise(args)

