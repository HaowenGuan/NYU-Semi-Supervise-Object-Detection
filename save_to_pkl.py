import os
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.engine import default_argument_parser
import pickle
import detectron2.tools.train_net as train_net


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)

    cfg = train_net.setup(args)

    model = train_net.Trainer.build_model(cfg)
    DetectionCheckpointer(model).resume_or_load(cfg.MODEL.WEIGHTS, resume=False)  # args.resume
    checkpointer = DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR)

    from collections import OrderedDict
    weights = OrderedDict()
    for p in checkpointer.model.state_dict():
        weights[p] = checkpointer.model.state_dict()[p].cpu().numpy()

    with open(os.path.join("output/model_supervised.pkl"), 'wb') as f:
        myModel = {'model': weights, '__author__': "Haowen Guan [haowen@nyu.edu]"}
        pickle.dump(myModel, f)
