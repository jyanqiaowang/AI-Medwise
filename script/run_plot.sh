#!/bin/bash
# Change confs as needed
# pattern is the naming pattern of the csv files that correspond to each confidence level
# ratio-col is the column name in the csv that contains the detection ratio
# label-col is the column name in the csv that contains the ground truth labels
# positive-str is the string in the label-col that indicates a positive case
# thresholds are the range of detection ratios to evaluate precision and recall
# out-png is the output file for the precision-recall curve plot
# out-csv is the output file for the precision-recall values
python plot_pr_from_confs.py \
  --dir /mnt/sunlab-nas-1/CVAT \
  --confs 0.4,0.5,0.6,0.7,0.8 \
  --pattern "{conf}conf_1update_yolo.csv" \
  --ratio-col FaceDetectRatio \
  --label-col Label \
  --positive-str "poor quality" \
  --thresholds 0.20:0.70:0.05 \
  --out-png /mnt/sunlab-nas-1/CVAT/pr_curves.png \
  --out-csv /mnt/sunlab-nas-1/CVAT/pr_table.csv
