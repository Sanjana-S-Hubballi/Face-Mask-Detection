import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# === PARAMETERS ===
IMG_SIZE = 224

IMAGE_FOLDER = 'D:/D-DRIVE/FaceMaskDetection/pythonProject/dataset/images'
OUTPUT_MASK = 'D:/D-DRIVE/FaceMaskDetection/pythonProject/classifiedimages/mask'
OUTPUT_NOMASK = 'D:/D-DRIVE/FaceMaskDetection/pythonProject/classifiedimages/nomask'

os.makedirs(OUTPUT_MASK, exist_ok=True)
os.makedirs(OUTPUT_NOMASK, exist_ok=True)

# === Face Detection Model (OpenCV SSD + ResNet10) ===
face_net = cv2.dnn.readNetFromCaffe(
    'D:/D-DRIVE/FaceMaskDetection/pythonProject/deploy.prototxt',
    'D:/D-DRIVE/FaceMaskDetection/pythonProject/res10_300x300_ssd_iter_140000.caffemodel'
)

# === Mask Classification Model (MobileNetV2) ===
mask_model = load_model('D:/D-DRIVE/FaceMaskDetection/pythonProject/src/mask_detector.h5')

# === Process each image ===
for filename in os.listdir(IMAGE_FOLDER):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        image_path = os.path.join(IMAGE_FOLDER, filename)
        image = cv2.imread(image_path)
        if image is None:
            print(f"❌ Could not read {image_path}")
            continue

        (h, w) = image.shape[:2]

        # === Detect faces ===
        blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300),
                                     (104.0, 177.0, 123.0))
        face_net.setInput(blob)
        detections = face_net.forward()

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x1, y1, x2, y2) = box.astype("int")

                # Ensure coordinates are within image bounds
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w - 1, x2), min(h - 1, y2)

                face = image[y1:y2, x1:x2]
                if face.size == 0:
                    continue

                # === Preprocess face for MobileNetV2 ===
                face_resized = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
                face_array = img_to_array(face_resized) / 255.0
                face_array = np.expand_dims(face_array, axis=0)

                pred = mask_model.predict(face_array)[0][0]
                label = 'mask' if pred < 0.5 else 'nomask'
                save_dir = OUTPUT_MASK if label == 'mask' else OUTPUT_NOMASK

                save_name = f"{filename[:-4]}_{i}.jpg"
                save_path = os.path.join(save_dir, save_name)
                cv2.imwrite(save_path, face)
                print(f"✅ Saved {label} face → {save_path}")

print("🎉 All detected faces classified and saved!")
