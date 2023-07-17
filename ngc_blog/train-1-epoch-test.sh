# Test only

# python detection/train.py    --dataset coco --data-path=xview_dataset/ --model fasterrcnn_resnet50_fpn --epochs 480 --resume checkpoints/model_479.pth  --lr-steps 16 22 --aspect-ratio-group-factor 3 --batch-size 16 --lr 0.02 --print-freq 1 --test-only 2>&1 | tee test.log 

# # No freezing layers
python detection/train.py    --dataset coco --data-path=xview_dataset/ --model fasterrcnn_resnet50_fpn --epochs 481 --resume checkpoints/model_479.pth  --lr-steps 16 22 --aspect-ratio-group-factor 3 --batch-size 16 --lr 0.02  2>&1 | tee out_no_freezing.log
