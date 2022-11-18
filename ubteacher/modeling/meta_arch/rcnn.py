# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from detectron2.modeling.meta_arch.rcnn import GeneralizedRCNN
import torch


@META_ARCH_REGISTRY.register()
class TwoStagePseudoLabGeneralizedRCNN(GeneralizedRCNN):
    def forward(
        self, batched_inputs, branch="supervised", given_proposals=None, val_mode=False
    ):
        if (not self.training) and (not val_mode):
            if type(batched_inputs[0]) != dict:
                new_batch = []
                for x in batched_inputs:
                    d = {"image": x}
                    new_batch.append(d)
                batched_inputs = new_batch
            # return self.inference(batched_inputs)
            predictions = self.inference(batched_inputs)
            if len(predictions) == 1:
                pred = {}
                # if len(predictions[0]['instances'].get('pred_boxes')) == 0:
                #     pred["boxes"] = torch.tensor([[1,2,3,4]])
                #     pred["scores"] = torch.tensor([1.])
                #     pred["labels"] = torch.tensor([2])
                # else:
                xmin, ymin, len_x, len_y = predictions[0]['instances'].get('pred_boxes').tensor.unbind(1)
                pred["boxes"] = torch.stack((xmin, ymin, xmin + len_x, ymin + len_y), dim=1)
                pred["scores"] = predictions[0]['instances'].get('scores')
                pred["labels"] = predictions[0]['instances'].get('pred_classes')
                return [pred]
            return predictions

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
