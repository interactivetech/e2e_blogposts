# torchrun --nproc_per_node=1 detection/train.py\
#     --dataset coco --data-path=/run/determined/workdir/shared_fs/data/xview_dataset/ --model fasterrcnn_resnet50_fpn --epochs 26\
#     --lr-steps 16 22 --aspect-ratio-group-factor 3

python detection/train.py\
    --dataset coco --data-path=/run/determined/workdir/shared_fs/data/xview_dataset/ --model fasterrcnn_resnet50_fpn --epochs 26\
    --lr-steps 16 22 --aspect-ratio-group-factor 3