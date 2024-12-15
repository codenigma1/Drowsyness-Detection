# 🚗 Drowsiness Detection System

A robust **Drowsiness Detection System** using two approaches:
1. **Deep Learning with Transfer Learning (InceptionV3)** 🚀
2. **MediaPipe for Eye Aspect Ratio (EAR) and Head Pose Estimation** 📈

This project aims to enhance road safety by detecting driver drowsiness in real-time using computer vision and deep learning techniques. The system alerts drivers when drowsiness is detected to prevent accidents.

---

## 🛠 Features

- **Real-Time Detection** 📹  
  - Uses webcam feed for real-time monitoring to identify drowsiness.

- **Deep Learning Approach** 🧠  
  - **Robust and Flexible**: The transfer learning model with **InceptionV3** is trained to detect drowsiness under various conditions. It can accurately classify eye states (open/closed) even with head movement, moderate lighting changes, and different face orientations.  
  - Suitable for applications where precision and adaptability are crucial.  

  ![Deep Learning Approach Demo](path/to/deep_learning_demo.gif)  
  _*GIF: Demonstrates detection under varying conditions.*_

- **MediaPipe Approach** 🧩  
  - **Lightweight and Efficient**: This method calculates the **Eye Aspect Ratio (EAR)** and monitors head pose without the need for neural network training.  
  - **Limitations**: It works well when the neck is not excessively tilted down, making it ideal for scenarios with minimal head movement.  
  - Highly efficient, suitable for resource-constrained devices.  

  ![MediaPipe Approach Demo](path/to/mediapipe_demo.gif)  
  _*GIF: Demonstrates detection using EAR and head pose.*_

- **Custom Alerts** 🔊  
  - Plays an alarm sound if drowsiness is detected for an extended duration.  

- **Interactive Visualization** 🖼  
  - Displays live status, EAR values, and predictions with overlays in real-time.

---
