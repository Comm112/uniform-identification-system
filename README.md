# uniform-identification-system
A computer vision system for identifying school uniforms and logging attendance using deep learning and a web-based interface.
Uniform Identification System

Project Overview

The Uniform Identification System is a computer vision–based application designed to automatically identify school uniforms from images and real-time camera input. The system uses a trained deep learning model to classify uniforms and log attendance based on the detected school.

This project demonstrates how artificial intelligence can be applied to automate identification tasks and reduce manual attendance processes.

Problem Statement

Manual identification of school uniforms and attendance tracking is time-consuming and prone to human error. Schools and institutions require a faster and more reliable method to identify uniforms, especially in environments with many students.

This project addresses the problem by using image classification to recognize uniforms automatically and log attendance in real time.

Objectives

Build a deep learning model capable of identifying different school uniforms

Improve model performance using fine-tuning and better preprocessing

Deploy the trained model into a web-based application

Enable real-time uniform detection using a webcam

Store detected attendance records automatically

System Features

Upload an image and receive:

Predicted school name

Confidence score

Real-time webcam detection with bounding boxes

Automatic attendance logging using SQLite

Clean dark-themed web interface

Simple and user-friendly design

Model & Techniques Used

Model Format: ONNX

Architecture: CNN (converted from PyTorch)

Image Size: 224 × 224

Preprocessing:

Resizing

Normalization

Inference Engine: ONNX Runtime

Application Interface

The system is deployed as a web application that allows users to:

Upload images for classification

View prediction results and confidence levels

Monitor an attendance log generated in real time

Screenshots of the interface and attendance log are included in the project report.

Evaluation Metrics

The model performance was evaluated using:

Accuracy

Precision

Recall

F1-Score

These metrics help measure both correctness and reliability of the uniform classification.

Project Structure
uniform-identification-system/
│
├── app.py
├── real_time_attendance.py
├── export_onnx.py
├── classes.json
├── model.onnx
├── templates/
├── static/
├── sample_images/
├── requirements.txt
└── README.md

Dataset Notice

The full dataset used for training and validation is not included in this repository due to size and privacy considerations.
Only sample images are provided for demonstration purposes.

How to Run the Project

Clone the repository

Create and activate a virtual environment

Install dependencies:

pip install -r requirements.txt


Run the web application:

python app.py


Open your browser and visit:

http://127.0.0.1:5000

Testing

The system was tested using:

New images not included in the training dataset

Real-time webcam input

Results show that the model can generalize reasonably well but may still produce incorrect predictions in challenging conditions.
