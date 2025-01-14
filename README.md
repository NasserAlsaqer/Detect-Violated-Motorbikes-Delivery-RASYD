---

## License Plate Detection and Traffic Violation Reporting System

<div align="center">
  <img src="https://github.com/user-attachments/assets/a1e397ad-2b6c-4d79-ba72-cc902868c788" alt="RASYD Logo" width="300px">
</div>

This project implements an end-to-end system for detecting traffic violations and automatically sending email notifications with evidence. Using a combination of **YOLOv8** for object detection, an **OCR model** for license plate recognition, and an **email notification system**, the project aims to streamline traffic violation reporting.

---

## Table of Contents
- [Project Overview](#project-overview)
- [Process Overview](#process-overview)
- [Dataset](#dataset)
- [Demo](#demo)
- [Evaluation](#Evaluation)
- [Deployment](#deployment)
- [Key Features](#key-features)
- [Techniques and Libraries Used](#techniques-and-libraries-used)
- [Setup and Usage](#setup-and-usage)
- [Tools](#Tools)
- [Contributing](#Contributing)
- [Team](#team)

---

## Project Overview

The system detects traffic violations such as **No Helmet** and **Entering Red Lane**, extracts the license plate number (in both Arabic and English), checks the license plate in a database for contact information, and sends an email notification to the violator. An image showing the violation is attached to the email.

---

## Process Overview

![process_flow](https://github.com/user-attachments/assets/a539e6fb-ef62-4f4f-a75e-3e898daeec1c)

1. **Raw Image**: The process starts with an image of the motorbike captured from a traffic camera.
2. **YOLOv8 Detection**: YOLOv8 is used to detect the motorbike, the license plate, and whether the rider is wearing a helmet.
3. **Lane Boundary Detection**: The system checks whether the motorbike is in a restricted lane, like the red lane.
4. **OCR Detection**: The **GOT-OCR2_0** model extracts the license plate characters in both Arabic and English.
5. **Database Integration**: The detected license plate is cross-referenced with a database that stores the violator's contact details, such as email and phone number.
6. **Email Integration**: A violation report is generated and sent via email to the violator with an image of the violation and license plate details.

---

## Dataset

You can access the dataset we used for motorbike delivery detection at the following link:

[Motorbike Delivery Dataset 2.0 on Roboflow](https://universe.roboflow.com/motorbikedelivery/motorbikedelivery_2.0)

The project began with a dataset of **2,814 source images**, which were manually annotated with **6 classes**:

- **Motorbike Delivery**
- **Motorbike Sport**
- **Helmet**
- **No Helmet**
- **Person**
- **License Plate**

then we generate the dataset to be 6772

### Dataset Split
We split the dataset into **Train**, **Validation**, and **Test** sets:

| Dataset Split  | Number of Images | Percentage |
| -------------- | ---------------- | ---------- |
| **Train Set**  | 5937             | 88%        |
| **Validation** | 564              | 8%         |
| **Test Set**   | 271              | 4%         |

### Preprocessing and Augmentation
The following preprocessing and augmentation techniques were applied to generate a final dataset of **6,772 images**:
- **Resize**: Stretch to 640x640 pixels.
- **Grayscale**: Applied to 10% of images.
- **Exposure Adjustments**: Between -20% and +20%.
- **Blur**: Up to 2.5px.
- **Noise**: Up to 1.99% of pixels.
- **Bounding Box Brightness**: Between -20% and +20%.

---

## Demo

### Video Demo:

[Watch the full system demo here](https://youtu.be/LgCqOEWqY0A?feature=shared)

- **YOLOv8 Detection**: Detects motorbikes, helmets, and lanes in real time.
- **OCR in Action**: Recognizes English characters from license plates.
- **Violation Detection**: Identifies and logs violations like "No Helmet" and "Entering Red Lane."
- **Email Notification**: Automatically sends a violation report with an attached image of the offense.

![Lane_violation](https://github.com/user-attachments/assets/5da05a03-e676-4ffd-8b8c-6424977d1c74)

---

### Steps to Run the Demo:

1. **Upload Images**: Upload images or video frames to the system.
2. **YOLOv8 Model**: Detects motorbikes, helmets, and restricted lanes.
3. **OCR**: The system extracts the license plate number using **GOT-OCR2_0** and cross-references the database.
4. **Violation Report**: A report is generated and sent via email, containing details of the violation, the license plate, and the attached image.

---

## Evaluation

| Class             | mAP50  |
|-------------------|--------|
| **All**           | 0.711  |
| **Helmet**        | 0.633  |
| **License_plate** | 0.726  |
| **MotorbikeDelivery** | 0.833  |
| **MotorbikeSport**    | 0.895  |
| **No_Helmet**     | 0.286  |
| **Person**        | 0.817  |

---

## Deployment

[Try our model here](https://huggingface.co/spaces/TheKnight115/T5_final_project)

---

## Key Features

### 1. License Plate Detection and OCR:
- **YOLOv8 Model**: The system uses a fine-tuned YOLOv8 model to detect motorbikes, helmets, and lanes in input images. This model is pre-trained to identify motorbikes violating traffic rules, such as driving without a helmet or entering a restricted lane.
- **GOT-OCR Transformer**: The license plates detected by YOLOv8 are passed to the **GOT-OCR2_0 transformer** model for character recognition. This model supports both Arabic and English characters commonly found on Saudi Arabian license plates.

### 2. Data Preprocessing and Image Manipulation:
- **Image Preprocessing**: Input images are processed using OpenCV and converted to **PIL format** for text overlaying and image manipulation.
- **Text Overlay**: The `draw_text_pil` function overlays the detected license plate number (in Arabic or English) on the original image for better visualization.

### 3. License Plate Text Processing:
- **Text Filtering**: The OCR-extracted text is cleansed using the `filter_license_plate_text` function to remove unwanted characters and ensure proper formatting.
- **Conversion to Arabic**: The system can convert English characters on license plates to Arabic for better readability and compliance with local standards using the `convert_to_arabic` function.

### 4. Traffic Violation Detection:
- The system supports multiple traffic violation types:
  - **No Helmet**
  - **Entering Red Lane**
  - **Combination of both (No Helmet + Entering Red Lane)**
- Based on the detected violation, the system automatically generates a notification with all the necessary details.

### 5. Email Notification System:
- **Automated Email Sending**: The `send_email` function sends an email to the violator using the violator's contact information retrieved from a **license plate database**. The email includes:
  - The violation type (in Arabic)
  - The detected license plate number (in Arabic or English)
  - An attached image of the violation

---

## Techniques and Libraries Used

### 1. Object Detection:
- **YOLOv8**: Pre-trained and fine-tuned to detect motorbikes, helmets, and lane violations with bounding boxes around the objects of interest.

### 2. Optical Character Recognition (OCR):
- **GOT-OCR2_0 Transformer**: A transformer-based OCR model capable of recognizing Arabic and English characters from license plates.

### 3. Image Manipulation:
- **OpenCV** and **Pillow (PIL)**: Used for image processing, conversion between formats, and adding visual elements such as text overlays.

### 4. Regular Expressions:
- **Text Filtering**: Regular expressions are used to filter and format OCR-extracted license plate text.

### 5. Email Automation:
- **SMTP**: Python's `smtplib` is used to send email notifications automatically, including dynamically generated email content and image attachments.

### 6. Database Lookup:
- **License Plate Database**: The system checks the detected license plate in a pre-existing database to retrieve the contact information of the vehicle owner. This information is used to send the violation email.

---

## Setup and Usage

### Prerequisites:
- **Python 3.x**
- Required libraries:
  - `transformers`
  - `opencv-python`
  - `Pillow`
  - `re`
  - `smtplib`
  - `email`
  - `torch`
  - `datasets`
  - `ultralytics`
  - `numpy`
  - `matplotlib`

You can install the necessary packages using pip:

```bash
pip install transformers opencv-python Pillow smtplib email torch datasets ultralytics numpy matplotlib
```
## Tools

- **Manage Dataset with [Roboflow](https://roboflow.com/)**:  
  ![Roboflow](https://img.shields.io/badge/-Roboflow-purple?logo=roboflow&logoColor=white)
  
- **Build the User Interface using [Streamlit](https://streamlit.io/)**:  
  ![Streamlit](https://img.shields.io/badge/-Streamlit-red?logo=streamlit&logoColor=white)
  
- **Deploy on [Hugging Face Spaces](https://huggingface.co/spaces)**:  
  ![HuggingFace](https://img.shields.io/badge/-Hugging%20Face-yellow?logo=huggingface&logoColor=white)
  
- **Collaboration on [GitHub](https://github.com/)**:  
  ![GitHub](https://img.shields.io/badge/-GitHub-black?logo=github&logoColor=white)

## Contributing

We welcome contributions from the community! If you'd like to contribute to the project, please follow these guidelines:

- Fork the repository.
- Create a new branch for your feature or bug fix.
- Commit your changes with descriptive messages.
- Open a pull request to the `main` branch.

### Areas to Contribute:
- **Enhancing OCR Detection**: Help improve the accuracy of detecting license plates, especially for challenging lighting conditions and different plate formats.
- **System of Database**: Contribute to enhancing the database integration, especially with improving the search efficiency for license plate violations.
- **Violation Value Integration**: Add functionality to assign monetary or point-based penalties for each type of violation detected by the system.
- **Automating Region of Interest (ROI) Detection**: Work on optimizing the YOLO model to automatically set and adjust regions of interest based on different traffic conditions.


## TEAM
- [![Github](https://img.shields.io/badge/Github-Nasser%20Alsaqer-181717?style=flat-square&logo=github)](https://github.com/NasserAlsaqer) 
  [![LinkedIn](https://img.shields.io/badge/LinkedIn-Nasser%20Alsaqer-blue?style=flat-square&logo=linkedin)](https://www.linkedin.com/in/nasser-alsaqer/)

- [![Github](https://img.shields.io/badge/Github-Khaled%20Alduwaysan-181717?style=flat-square&logo=github)](https://github.com/Duwaysan) 
  [![LinkedIn](https://img.shields.io/badge/LinkedIn-Khaled%20Alduwaysan-blue?style=flat-square&logo=linkedin)](https://www.linkedin.com/in/kduwaysan/)

- [![Github](https://img.shields.io/badge/Github-Abdulrahman%20Alghamdi-181717?style=flat-square&logo=github)](https://github.com/AbdulrhmanBakrgh) 
  [![LinkedIn](https://img.shields.io/badge/LinkedIn-Abdulrahman%20Alghamdi-blue?style=flat-square&logo=linkedin)](https://www.linkedin.com/in/abdulrahman-bakr-3a2895236/)



- [![Github](https://img.shields.io/badge/Github-Fares%20Altoukhi-181717?style=flat-square&logo=github)](https://github.com/TheKnight909) 
  [![LinkedIn](https://img.shields.io/badge/LinkedIn-Fares%20Altoukhi-blue?style=flat-square&logo=linkedin)](https://www.linkedin.com/in/fares-altoukhi/)

- [![Github](https://img.shields.io/badge/Github-Alhanouf%20Alhumid-181717?style=flat-square&logo=github)](https://github.com/alhanoufalh) 
  [![LinkedIn](https://img.shields.io/badge/LinkedIn-Alhanouf%20Alhumid-blue?style=flat-square&logo=linkedin)](https://www.linkedin.com/in/alhanouf-alhumid-40a7391b0/?originalSubdomain=sa)
