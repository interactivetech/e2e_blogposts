import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torch
import os

def load_model_ddp(loaded_model,model_state_dict):
    '''
    '''
    try:
        loaded_model.load_state_dict(model_state_dict)
    except Exception:
        # If the checkpointed model is non-DDP and the current model is DDP, append
        # module prefix to the checkpointed data
        if isinstance(loaded_model, torch.nn.parallel.DistributedDataParallel):
            print("Loading non-DDP checkpoint into a DDP model.")
            self._add_prefix_in_state_dict_if_not_present(model_state_dict, "module.")
        else:
            # If the checkpointed model is DDP and if we are currently running in
            # single-slot mode, remove the module prefix from checkpointed data
            print("Loading DDP checkpoint into a non-DDP model.")
            torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(
                model_state_dict, "module."
            )
        loaded_model.load_state_dict(model_state_dict)
    return loaded_model
def build_frcnn_model(num_classes):
    print("Loading pretrained model...")
    # load an detection model pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    try:
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        # replace the pre-trained head with a new one
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 61)
        path = os.path.join('/run/determined/workdir/shared_fs/01 - Users/andrew.mendez/frcnn_xview.pth')
        model=load_model_ddp(model,torch.load(path,map_location=torch.device('cpu')))
    except Exception as e:
        print(e)
        pass
    # get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.min_size=800
    model.max_size=1333
    # RPN parameters
    model.rpn_pre_nms_top_n_train=2000
    model.rpn_pre_nms_top_n_test=1000
    model.rpn_post_nms_top_n_train=2000
    model.rpn_post_nms_top_n_test=1000
    model.rpn_nms_thresh=0.7
    model.rpn_fg_iou_thresh=0.7
    model.rpn_bg_iou_thresh=0.3
    model.rpn_batch_size_per_image=256
    model.rpn_positive_fraction=0.5
    model.rpn_score_thresh=0.05
    # Box parameters
    model.box_score_thresh=0.0
    model.box_nms_thresh=0.5
    model.box_detections_per_img=300
    model.box_fg_iou_thresh=0.5
    model.box_bg_iou_thresh=0.5
    model.box_batch_size_per_image=512
    model.box_positive_fraction=0.25
    return model