python batch_face_yolo.py /mnt/sunlab-nas-1/CVAT/Result_videomae/val_predictions.csv \
      --weights /mnt/sunlab-nas-1/CVAT/yolov8s-face-lindevs.pt \
      --sample-rate 10 \
      --conf 0.6 \
      --iou 0.5 \
      --max-save 20 \
      --out-dir /mnt/sunlab-nas-1/CVAT/Result_poorQ/out_frames \
      --t1 0.35 \
      --t2 0.55 \
      --output-csv /mnt/sunlab-nas-1/CVAT/Result_poorQ/face_yolo_result.csv