python batch_face_yolo.py /mnt/sunlab-nas-1/CVAT/merged_labels_1update.csv \
      --base /mnt/sunlab-nas-1/CVAT \
      --weights /mnt/sunlab-nas-1/CVAT/yolov8s-face-lindevs.pt \
      --sample-rate 10 \
      --conf 0.7 \
      --iou 0.5 \
      --max-save 20 \
      --out-dir /mnt/sunlab-nas-1/CVAT/700/out_frames \
      --poor-threshold 0.50
