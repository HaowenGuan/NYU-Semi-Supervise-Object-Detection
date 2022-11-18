
from detectron2.modeling import build_model
from detectron2.config import get_cfg
from ubteacher import add_ubteacher_config
from detectron2.engine import default_argument_parser, default_setup, launch
from detectron2.checkpoint import DetectionCheckpointer
from ubteacher.modeling import *
from ubteacher.engine import *
from detectron2.data.datasets import register_coco_instances

config_file = "configs/Faster-RCNN/coco-standard/faster_rcnn_R_50_FPN_ut2_sup100_run0.yaml"
def setup():
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_ubteacher_config(cfg)
    print("CON FIG", config_file)
    cfg.merge_from_file(config_file)
    cfg.freeze()
    return cfg

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
        # res = Trainer.test(cfg, ensem_ts_model.modelTeacher)

    else:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=False
        )
    return model

