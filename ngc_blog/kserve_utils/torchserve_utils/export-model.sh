torch-model-archiver --model-name xview-fasterrcnn \
  --version 1.0 \
  --model-file /home/ubuntu/kserve-pytorch/serve/examples/object_detector/fast-rcnn/model-xview.py \
  --serialized-file model_8_stripped.pth --handler /home/ubuntu/kserve-pytorch/serve/examples/object_detector/fasterrcnn_handler.py \
  --extra-files /home/ubuntu/kserve-pytorch/serve/examples/object_detector/fast-rcnn/xview-labels/index_to_name.json