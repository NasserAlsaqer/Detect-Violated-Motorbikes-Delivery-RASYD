# processor.py

import cv2
import numpy as np
import os
from ultralytics import YOLO
from transformers import AutoProcessor, AutoModelForCausalLM, pipeline
from PIL import Image, ImageDraw, ImageFont
import re
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import torch
from dotenv import load_dotenv
from transformers import AutoProcessor, AutoModel

# Load environment variables
load_dotenv()

# Email credentials (Use environment variables for security)
FROM_EMAIL = os.getenv("FROM_EMAIL")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")
TO_EMAIL = os.getenv("TO_EMAIL")
SMTP_SERVER = 'smtp.gmail.com'
SMTP_PORT = 465

# Arabic dictionary for converting license plate text
arabic_dict = {
    "0": "٠", "1": "١", "2": "٢", "3": "٣", "4": "٤", "5": "٥",
    "6": "٦", "7": "٧", "8": "٨", "9": "٩", "A": "ا", "B": "ب",
    "J": "ح", "D": "د", "R": "ر", "S": "س", "X": "ص", "T": "ط",
    "E": "ع", "G": "ق", "K": "ك", "L": "ل", "Z": "م", "N": "ن",
    "H": "ه", "U": "و", "V": "ي", " ": " "
}

# Define class colors
class_colors = {
    0: (0, 255, 0),    # Green (Helmet)
    1: (255, 0, 0),    # Blue (License Plate)
    2: (0, 0, 255),    # Red (MotorbikeDelivery)
    3: (255, 255, 0),  # Cyan (MotorbikeSport)
    4: (255, 0, 255),  # Magenta (No Helmet)
    5: (0, 255, 255),  # Yellow (Person)
}

# Load the OCR model
processor = AutoProcessor.from_pretrained("stepfun-ai/GOT-OCR2_0", trust_remote_code=True)
model_ocr = AutoModel.from_pretrained("stepfun-ai/GOT-OCR2_0", trust_remote_code=True).to('cuda')

# Load YOLO model
# Ensure the path to the model is correct
model = YOLO('yolov8_Medium.pt')  # Update the path as needed

# Define lane area coordinates (example coordinates)
red_lane = np.array([[2,1583],[1,1131],[1828,1141],[1912,1580]], np.int32)

# Path for Arabic font
font_path = "alfont_com_arial-1.ttf"

# Dictionary to track violations per license plate
violations_dict = {}

def filter_license_plate_text(license_plate_text):
    """Filter and format the license plate text."""
    license_plate_text = re.sub(r'[^A-Z0-9]+', "", license_plate_text)
    match = re.search(r'(\d{4})\s*([A-Z]{2})', license_plate_text)
    return f"{match.group(1)} {match.group(2)}" if match else None

def convert_to_arabic(license_plate_text):
    """Convert license plate text from Latin to Arabic script."""
    return "".join(arabic_dict.get(char, char) for char in license_plate_text)

def send_email(license_text, violation_image_path, violation_type): 
    """Send an email notification with violation details and image attachment."""
    # Define the subject and body based on violation type
    subjects = {
        'No Helmet, In Red Lane': 'تنبيه مخالفة: عدم ارتداء خوذة ودخول المسار الأيسر',
        'In Red Lane': 'تنبيه مخالفة: دخول المسار الأيسر',
        'No Helmet': 'تنبيه مخالفة: عدم ارتداء خوذة'
    }
    bodies = {
        'No Helmet, In Red Lane': f"لعدم ارتداء الخوذة ولدخولها المسار الأيسر ({license_text}) تم تغريم دراجة نارية التي تحمل لوحة",
        'In Red Lane': f"لدخولها المسار الأيسر ({license_text}) تم تغريم دراجة نارية التي تحمل لوحة",
        'No Helmet': f"لعدم ارتداء الخوذة ({license_text}) تم تغريم دراجة نارية التي تحمل لوحة"
    }
    
    subject = subjects.get(violation_type, 'تنبيه مخالفة')
    body = bodies.get(violation_type, f"تم تغريم دراجة نارية التي تحمل لوحة ({license_text}) بسبب مخالفة.")
    
    # Create the email message
    msg = MIMEMultipart()
    msg['From'] = FROM_EMAIL
    msg['To'] = TO_EMAIL
    msg['Subject'] = subject 
    msg.attach(MIMEText(body, 'plain'))

    # Attach the violation image
    if os.path.exists(violation_image_path):
        with open(violation_image_path, 'rb') as attachment_file:
            part = MIMEBase('application', 'octet-stream')
            part.set_payload(attachment_file.read())
            encoders.encode_base64(part)
            part.add_header('Content-Disposition', f'attachment; filename={os.path.basename(violation_image_path)}')
            msg.attach(part)

    # Send the email using SMTP
    try:
        with smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT) as server:
            server.login(FROM_EMAIL, EMAIL_PASSWORD)
            server.sendmail(FROM_EMAIL, TO_EMAIL, msg.as_string())
            print("Email with attachment sent successfully!")
    except Exception as e:
        print(f"Failed to send email: {e}")

def draw_text_pil(img, text, position, font_path, font_size, color):
    """Draw text on an image using PIL for better font support."""
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    try:
        font = ImageFont.truetype(font_path, size=font_size)
    except IOError:
        print(f"Font file not found at {font_path}. Using default font.")
        font = ImageFont.load_default()
    draw.text(position, text, font=font, fill=color)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

def process_frame(frame, font_path, violation_image_path='violation.jpg'):
    """Process a single video frame for violations."""
    results = model.track(frame)
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
        label = model.names[int(box.cls)]
        color = class_colors.get(int(box.cls), (255, 255, 255))
        confidence = box.conf[0].item()

        # Draw bounding box and label
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
        cv2.putText(frame, f'{label}: {confidence:.2f}', (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        if label == 'MotorbikeDelivery' and confidence >= 0.4:
            motorbike_crop = frame[max(0, y1 - 50):y2, x1:x2]
            delivery_center = ((x1 + x2) // 2, y2)
            in_red_lane = cv2.pointPolygonTest(red_lane, delivery_center, False)
            violation_types = []
            if in_red_lane >= 0:
                violation_types.append("In Red Lane")

            # Detect sub-objects within the motorbike crop
            sub_results = model(motorbike_crop)

            for sub_box in sub_results[0].boxes:
                sub_x1, sub_y1, sub_x2, sub_y2 = map(int, sub_box.xyxy[0].cpu().numpy())
                sub_label = model.names[int(sub_box.cls)]
                if sub_label == 'No_Helmet':
                    violation_types.append("No Helmet")
                elif sub_label == 'License_plate':
                    license_crop = motorbike_crop[sub_y1:sub_y2, sub_x1:sub_x2]
                    if violation_types:
                        # Save violation image
                        cv2.imwrite(violation_image_path, frame)
                        license_plate_pil = Image.fromarray(cv2.cvtColor(license_crop, cv2.COLOR_BGR2RGB))
                        license_plate_pil.save('license_plate.png')

                        # Perform OCR
                        try:
                            license_plate_text = model_ocr.chat(processor, temp_image_path, ocr_type='ocr')                           
                        except Exception as e:
                            print(f"OCR failed: {e}")
                            license_plate_text = ""

                        filtered_text = filter_license_plate_text(license_plate_text)
                        if filtered_text:
                            if filtered_text not in violations_dict:
                                violations_dict[filtered_text] = violation_types
                                send_email(filtered_text, violation_image_path, ', '.join(violation_types))
                            else:
                                current = set(violations_dict[filtered_text])
                                new = set(violation_types)
                                updated = current | new
                                if updated != current:
                                    violations_dict[filtered_text] = list(updated)
                                    send_email(filtered_text, violation_image_path, ', '.join(updated))
                            
                            arabic_text = convert_to_arabic(filtered_text)
                            frame = draw_text_pil(frame, filtered_text, (x1, y2 + 30), font_path, 30, (255, 255, 255))
                            frame = draw_text_pil(frame, arabic_text, (x1, y2 + 60), font_path, 30, (0, 255, 0))
    return frame

def process_image(image_path, font_path, violation_image_path='violation.jpg'):
    """Process an uploaded image and return the processed image."""
    frame = cv2.imread(image_path)
    if frame is None:
        print("Error loading image")
        return None

    processed = process_frame(frame, font_path, violation_image_path)
    return processed

def process_video(video_path):
    # Paths for saving violation images
    violation_image_path = 'violation.jpg'

    # Track emails already sent to avoid duplicate emails
    sent_emails = {}

    # Dictionary to track violations per license plate
    violations_dict = {}

    # Open video file
    cap = cv2.VideoCapture(video_path)

    # Check if the video file opened successfully
    if not cap.isOpened():
        print("Error opening video file")
        return None

    # Define codec and output video settings
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_video_path = 'output_violation.mp4'
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # Frame width
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # Frame height
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    margin_y = 50

    # Process the video frame by frame
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # End of video

        # Draw the red lane polygon on each frame
        cv2.polylines(frame, [red_lane], isClosed=True, color=(0, 0, 255), thickness=3)  # Red lane

        # Perform detection using YOLO on the current frame
        results = model.track(frame)

        # Process each detection in the results
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())  # Bounding box coordinates
            label = model.names[int(box.cls)]  # Class name (MotorbikeDelivery, Helmet, etc.)
            color = class_colors[int(box.cls)]
            confidence = box.conf[0].item()

            # Initialize flags and variables for the violations
            helmet_violation = False
            lane_violation = False
            violation_type = []

            # Draw bounding box around detected object
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

            # Add label to the box (e.g., 'MotorbikeDelivery')
            cv2.putText(frame, f'{label}: {confidence:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Detect MotorbikeDelivery
            if label == 'MotorbikeDelivery' and confidence >= 0.4:
                motorbike_crop = frame[max(0, y1 - margin_y):y2, x1:x2]
                delivery_center = ((x1 + x2) // 2, (y2))
                in_red_lane = cv2.pointPolygonTest(red_lane, delivery_center, False)
                if in_red_lane >= 0:
                    lane_violation = True
                    violation_type.append("In Red Lane")

                # Perform detection within the cropped motorbike region
                sub_results = model(motorbike_crop)

                for result in sub_results[0].boxes:
                    sub_x1, sub_y1, sub_x2, sub_y2 = map(int, result.xyxy[0].cpu().numpy())  # Bounding box coordinates
                    sub_label = model.names[int(result.cls)]
                    sub_color = (255, 0, 0)  # Red color for the bounding box of sub-objects

                    # Draw bounding box around sub-detected objects (No_Helmet, License_plate, etc.)
                    cv2.rectangle(motorbike_crop, (sub_x1, sub_y1), (sub_x2, sub_y2), sub_color, 2)
                    cv2.putText(motorbike_crop, sub_label, (sub_x1, sub_y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, sub_color, 2)

                    if sub_label == 'No_Helmet':
                        helmet_violation = True
                        violation_type.append("No Helmet")
                        continue
                    if sub_label == 'License_plate':
                        license_crop = motorbike_crop[sub_y1:sub_y2, sub_x1:sub_x2]

                        # Apply OCR if a violation is detected
                        if helmet_violation or lane_violation:
                            # Perform OCR on the license plate
                            cv2.imwrite(violation_image_path, frame)
                            license_plate_pil = Image.fromarray(cv2.cvtColor(license_crop, cv2.COLOR_BGR2RGB))
                            temp_image_path = 'license_plate.png'
                            license_plate_pil.save(temp_image_path)
                            license_plate_text = model_ocr.chat(processor, temp_image_path, ocr_type='ocr')
                            filtered_text = filter_license_plate_text(license_plate_text)
                            
                            if filtered_text:
                                # Track violations for the license plate
                                if filtered_text not in violations_dict:
                                    violations_dict[filtered_text] = violation_type
                                    send_email(filtered_text, violation_image_path, ', '.join(violation_type))
                                else:
                                    # Update violations if new ones are found
                                    current_violations = set(violations_dict[filtered_text])
                                    new_violations = set(violation_type)
                                    updated_violations = list(current_violations | new_violations)

                                    if updated_violations != violations_dict[filtered_text]:
                                        violations_dict[filtered_text] = updated_violations
                                        send_email(filtered_text, violation_image_path, ', '.join(updated_violations))

                                # Draw OCR text (English and Arabic) on the original frame
                                arabic_text = convert_to_arabic(filtered_text)
                                frame = draw_text_pil(frame, filtered_text, (x1, y2 + 30), font_path, font_size=30, color=(255, 255, 255))
                                frame = draw_text_pil(frame, arabic_text, (x1, y2 + 60), font_path, font_size=30, color=(0, 255, 0))

        # Write the processed frame to the output video
        out.write(frame)

    # Release resources when done
    cap.release()
    out.release()

    return output_video_path