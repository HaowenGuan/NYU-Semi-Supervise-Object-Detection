
from detectron2.config import get_cfg
from ubteacher import add_ubteacher_config
from detectron2.checkpoint import DetectionCheckpointer
from ubteacher.modeling import *
from ubteacher.engine import *
import ubteacher.data.datasets.builtin

config_file = "semi_supervise_final_24.7/semi_supervised_evaluation.yaml"
def setup():
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_ubteacher_config(cfg)
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

    else:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=False
        )
    return model