import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# === PARAMETERS ===
DATA_DIR = 'D:/D-DRIVE/FaceMaskDetection/pythonProject/classifiedimages'
IMG_SIZE = 224
BATCH_SIZE = 32

model = load_model('best_face_mask_model.h5')

# === Validation generator ===
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

val_data = datagen.flow_from_directory(
    DATA_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation',
    shuffle=False
)

# === Predict & collect results ===
y_true = []
y_pred = []
filenames = val_data.filenames

results = []

index = 0

for i in range(len(val_data)):
    X, y_batch = val_data[i]
    preds = model.predict(X)
    y_batch = y_batch.tolist()
    pred_batch = [1 if p >= 0.5 else 0 for p in preds]

    y_true.extend(y_batch)
    y_pred.extend(pred_batch)

    for j in range(len(y_batch)):
        results.append((
            filenames[index],
            'Mask' if y_batch[j] == 0 else 'No Mask',
            'Mask' if pred_batch[j] == 0 else 'No Mask'
        ))
        index += 1

    if index >= len(filenames):
        break

# === Compute metrics ===
cm = confusion_matrix(y_true, y_pred)
report = classification_report(y_true, y_pred, target_names=['Mask', 'No Mask'])

TP = cm[0][0]
FN = cm[0][1]
FP = cm[1][0]
TN = cm[1][1]

FRR = FN / (TP + FN) if (TP + FN) != 0 else 0
FAR = FP / (FP + TN) if (FP + TN) != 0 else 0

# === Save EVERYTHING to output.txt ===
with open('output.txt', 'w', encoding='utf-8') as f:
    # Detailed predictions
    f.write("Filename,True Label,Predicted Label\n")
    for fname, true_label, pred_label in results:
        f.write(f"{fname},{true_label},{pred_label}\n")

    f.write("\n=== Confusion Matrix ===\n")
    f.write(str(cm) + "\n\n")

    f.write("=== Classification Report ===\n")
    f.write(report + "\n")

    f.write(f"False Rejection Rate (FRR): {FRR:.2f}\n")
    f.write(f"False Acceptance Rate (FAR): {FAR:.2f}\n")

print("✅ ALL results saved to output.txt")