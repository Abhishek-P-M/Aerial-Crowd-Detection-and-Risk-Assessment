# Deep Learning–Based Aerial Crowd Detection and Risk Assessment
Deep Learning · Computer Vision · Object Detection Systems

This project presents a complete deep-learning–based pipeline for detecting people in aerial (drone) imagery and quantifying crowd congestion for automated monitoring. 
Using a customized version of the VisDrone dataset and a YOLO-based detector, the system identifies individuals, computes crowd-density metrics, and highlights frames with potentially risky gatherings.

Model Used

A YOLO family model (YOLOv8/YOLOv11/YOLOv12) was used and trained on the customized dataset.
YOLO was chosen due to:
Excellent performance on small aerial objects
Real-time inference capability
Strong generalization on crowded, cluttered scenes

Run on Google Colab

You can run the full project on Google Colab using the provided notebook:
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1X4QI4HJxnC_1Ify7hc7hYt1IonA4bxWV#scrollTo=D82AB85c1KDG)
