# torchrun --nproc_per_node=1 detection/train.py\
#     --dataset coco --data-path=/run/determined/workdir/shared_fs/data/xview_dataset/ --model fasterrcnn_resnet50_fpn --epochs 26\
#     --lr-steps 16 22 --aspect-ratio-group-factor 3

torchrun --nproc_per_node=8 detection/train.py    \
    --dataset coco --data-path=/run/determined/workdir/shared_fs/data/xview_dataset/ \
    --model fasterrcnn_resnet50_fpn \
    --epochs 26    \
    --lr-steps 16 22 \
    --aspect-ratio-group-factor 3 \
    --lr 0.005 \
    --weight-decay 0.0005 \
    --momentum 0.9

# python detection/train.py\
#     --dataset coco --data-path=/run/determined/workdir/shared_fs/data/xview_dataset/ --model fasterrcnn_resnet50_fpn --epochs 26\
#     --lr-steps 16 22 --aspect-ratio-group-factor 3