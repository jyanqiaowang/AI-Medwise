## 📌 Overview
This project implements a **data verification pipeline** that builds upon different detection methods.  
The main idea:  
- Videos are first pre-labeled as **poor quality**.  
- This algorithm further verifies poor quality by checking **face detectability** using YOLO and **illumination** of the background and face/body.  
- If no reliable face/body is detected or the illumination is too bad for further analysis, the video is confirmed as poor quality.  

This method provides an **automated and scalable way** to label videos, reducing the need for manual review.

---

## 🚀 Features
- Face detection and verification using YOLO.  
- Adjustable confidence thresholds (`conf=0.4–0.8`) for detection experiments.  
- Precision–Recall curve evaluation to select the best threshold.  
- Supports further rule-based decision making:
  - **High confidence poor quality**  
  - **Borderline cases (human review required)**  
  - **Not poor quality**

---

## 📂 Project Structure
```text
repo/
│── data/                # data input that contains label
│── results/             # Detection results, CSV outputs
│── scripts/             # Core Python scripts
│── notebooks/           # Analysis notebooks (PR curve, evaluation)
│── README.md            # Project documentation
