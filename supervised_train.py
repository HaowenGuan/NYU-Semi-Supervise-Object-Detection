from detectron2.engine import default_argument_parser, launch
import detectron2.tools.train_net as train_net

# register NYU dataset
from detectron2.data.datasets import register_coco_instances
register_coco_instances("nyu_train", {}, "/data/sbcaesar/nyu/labeled_data/annotation/labeled_train.json", "/data/sbcaesar/nyu/labeled_data/train2017")
register_coco_instances("nyu_val", {}, "/data/sbcaesar/nyu/labeled_data/annotation/labeled_val.json", "/data/sbcaesar/nyu/labeled_data/val2017")


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        train_net.main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )