# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from detectron2.modeling.meta_arch.rcnn import GeneralizedRCNN
import torch
from detectron2.data.detection_utils import _apply_exif_orientation, convert_PIL_to_numpy
from detectron2.data import transforms as T
import torchvision
import numpy as np


@META_ARCH_REGISTRY.register()
class TwoStagePseudoLabGeneralizedRCNN(GeneralizedRCNN):
    def forward(
        self, batched_inputs, branch="supervised", given_proposals=None, val_mode=False
    ):
        if not self.training and (not val_mode):
            if type(batched_inputs[0]) != dict:
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
                # print(batched_inputs)
                predictions = self.inference(batched_inputs)
                
                pred = {}
                if len(predictions[0]['instances'].get('pred_boxes')) != 0:
                    pred["boxes"] = predictions[0]['instances'].get('pred_boxes').tensor
                    pred["scores"] = predictions[0]['instances'].get('scores')
                    pred["labels"] = predictions[0]['instances'].get('pred_classes')

                return [pred]
            return self.inference(batched_inputs)

        images = self.preprocess_image(batched_inputs)

        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None

        features = self.backbone(images.tensor)

        if branch == "supervised":
            # Region proposal network
            proposals_rpn, proposal_losses = self.proposal_generator(
                images, features, gt_instances
            )

            # # roi_head lower branch
            _, detector_losses = self.roi_heads(
                images, features, proposals_rpn, gt_instances, branch=branch
            )

            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            return losses, [], [], None

        elif branch == "unsup_data_weak":
            # Region proposal network
            proposals_rpn, _ = self.proposal_generator(
                images, features, None, compute_loss=False
            )

            # roi_head lower branch (keep this for further production)  # notice that we do not use any target in ROI head to do inference !
            proposals_roih, ROI_predictions = self.roi_heads(
                images,
                features,
                proposals_rpn,
                targets=None,
                compute_loss=False,
                branch=branch,
            )

            return {}, proposals_rpn, proposals_roih, ROI_predictions

        elif branch == "unsup_data_train":  #

            # Region proposal network
            proposals_rpn, proposal_losses = self.proposal_generator(
                images, features, gt_instances
            )

            # # roi_head lower branch
            _, detector_losses = self.roi_heads(
                images, features, proposals_rpn, gt_instances, branch=branch
            )

            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            return losses, [], [], None
