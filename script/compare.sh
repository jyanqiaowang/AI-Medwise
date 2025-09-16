python compare_two_excel.py \
  --file-a /mnt/sunlab-nas-1/CVAT/underlit_annotated_datset.xlsx \
  --file-b /mnt/sunlab-nas-1/CVAT/merged_labels_2update_with_face_yolo.csv \
  --key-col "Video Name" \
  --decision-col "Decision" \
  --ratio-col-a "FaceDetectRatio" \
  --ratio-col-b "UnderlitRatio" \
  --label-col "label" \
  --out-dir /mnt/sunlab-nas-1/CVAT/compare_out
