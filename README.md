# Smart Attendance System

## Project Overview
This project presents a **Smart Attendance System** developed as part of the
CSAI 498 / CSAI 499 Graduation Project.

The system replaces traditional manual attendance with an **AI-based solution**
that uses computer vision and deep learning to automatically detect and recognize
students from images captured periodically during a class session.

Instead of processing continuous video streams, the system captures **one image
every minute**, identifies students present in each image, and calculates their
actual attendance duration based on how many frames they appear in.

---

## System Workflow
1. Capture classroom images periodically (1 frame per minute)
2. Detect faces in each image
3. Crop detected faces with padding
4. Recognize students using a **ResNet-based face recognition model**
5. Calculate attendance duration based on frame count
6. Store attendance records in a centralized database
7. Visualize results using a web-based dashboard

---

## Technologies Used
- Python
- OpenCV
- NumPy
- PyTorch
- ResNet18
- DNN SSD (OpenCV)
- Haar Cascade (fallback)
- HTML / CSS
- Flask
- Git & GitHub

---


## Current Progress
- Student image data collected
- Face detection implemented 
- Face cropping and preprocessing pipeline completed
- Labeled dataset prepared (Name + ID)
- **ResNet-based face recognition model training started**
- UI prototypes created (login page, dashboard)



---

## Team Members
- Hania Khaled – DSAI
- Farah Samir – DSAI
- Yassmin Raafat – SW
- Hager Ali – DSAI

---

## Supervisor
Dr. Mohamed Ghalwash

---

## Course Information
CSAI 498 / CSAI 499 – Graduation Project  
Zewail City of Science and Technology


