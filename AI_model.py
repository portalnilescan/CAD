import torch
from PIL import Image
from torchvision import transforms
import pydicom
import numpy as np
from PIL import Image
from ultralytics import YOLO
import numpy as np
import torch
import torch
import torch.nn as nn
import numpy as np
import pydicom
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet101, densenet121
import segmentation_models_pytorch as smp
import os


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
label_map={0:"Normal",1:"Abnormal"}
# Finding_map={
#  0: 'Aortic Enlargement',
#  1: 'Atelectasis',
#  2: 'Calcification',
#  3: 'Cardiomegaly',
#  4: 'Consolidation',
#  5: 'Ild',
#  6: 'Infiltration',
#  7: 'Lung Opacity',
#  8: 'Nodule/Mass',
#  9: 'Other Lesion',
#  10: 'Pleural Effusion',
#  11: 'Pleural Thickening',
#  12: 'Pneumothorax',
#  13: 'Pulmonary Fibrosis'
#  }
# # =========  Step 1: Load All Models Once

#Loading Yolob

# Load the YOLO model from the specified path
Yolo_path = 'Yolo_Multi_label_classifier_14class_VinBig.pt'

Yolo_model = YOLO(Yolo_path)




# =========  Load Models Once ==========

def load_ensemble_models():
    transform_chexnet = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5482]*3, std=[0.2667]*3)
    ])
    transform_seg = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5482]*3, std=[0.2667]*3)
    ])

    seg_model = smp.Unet(encoder_name='resnet34', in_channels=3, classes=1)
    seg_model.load_state_dict(torch.load("xray_Segmention_model.pth", map_location=DEVICE))
    seg_model.eval().to(DEVICE)

    def get_resnet_model():
        model = resnet101(weights=None)
        model.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(2048, 2))
        return model.to(DEVICE)

    def get_chexnet_model():
        model = densenet121(weights=None)
        model.classifier = nn.Linear(1024, 2)
        return model.to(DEVICE)

    model_seg = get_resnet_model()
    model_seg.load_state_dict(torch.load("ResNet_Segmentation.pth", map_location=DEVICE))
    model_seg.eval()

    model_chex_all = get_chexnet_model()
    model_chex_all.load_state_dict(torch.load("ChexNet_Full.pth", map_location=DEVICE))
    model_chex_all.eval()

    model_chex_2 = get_chexnet_model()
    model_chex_2.load_state_dict(torch.load("ChexNet_2classes.pth", map_location=DEVICE))
    model_chex_2.eval()

    return {
        "transform_chexnet": transform_chexnet,
        "transform_seg": transform_seg,
        "seg_model": seg_model,
        "model_seg": model_seg,
        "model_chex_all": model_chex_all,
        "model_chex_2": model_chex_2
    }
# =========  Step 2: Predict Single DICOM Image
def predict_image_class(dicom_path, models_dict, threshold=0.71):
    # Load image
    dicom = pydicom.dcmread(dicom_path,force=True)
    image_array = dicom.pixel_array.astype(np.float32)
    image_array -= image_array.min()
    image_array /= image_array.max()
    image_array *= 255
    image_array = image_array.astype(np.uint8)

    if len(image_array.shape) == 2:
        image_pil = Image.fromarray(image_array).convert("RGB")
    else:
        image_pil = Image.fromarray(image_array)

    # Transforms
    transform_chexnet = models_dict["transform_chexnet"]
    transform_seg = models_dict["transform_seg"]

    img_chexnet = transform_chexnet(image_pil).unsqueeze(0).to(DEVICE)
    img_seg_input = transform_seg(image_pil).unsqueeze(0).to(DEVICE)

    # Apply segmentation mask
    with torch.no_grad():
        mask = torch.sigmoid(models_dict["seg_model"](img_seg_input)).squeeze().cpu().numpy()
        mask = (mask > 0.5).astype(np.float32)

    image_resized = image_pil.resize((512, 512))
    image_np = np.array(image_resized).astype(np.float32) / 255.0
    masked_image = image_np * np.expand_dims(mask, axis=-1)
    masked_pil = Image.fromarray((masked_image * 255).astype(np.uint8))
    img_segmented = transform_chexnet(masked_pil).unsqueeze(0).to(DEVICE)

    # Get probabilities from each model
    def get_prob(model, img_tensor):
        with torch.no_grad():
            out = model(img_tensor)
            return torch.softmax(out, dim=1)[0, 1].item()

    p1 = get_prob(models_dict["model_seg"], img_segmented)
    p2 = get_prob(models_dict["model_chex_all"], img_chexnet)
    p3 = get_prob(models_dict["model_chex_2"], img_chexnet)

    probs = [p1, p2, p3]
    final_prob = np.mean(probs)
    pred_class = int(final_prob >= threshold)
    # label = "Abnormal" if pred_class else "Normal"

    # print(f"\n DICOM File: {dicom_path}")
    # print(f" Ensemble Probability: {final_prob:.4f}")
    # print(f" Prediction: {label}")

    return pred_class , image_pil
#once

def predict_image_class_from_png(image_path, models_dict, threshold=0.71):
    # Load PNG or JPG image and convert to RGB
    image_pil = Image.open(image_path).convert("RGB")

    # Transforms
    transform_chexnet = models_dict["transform_chexnet"]
    transform_seg = models_dict["transform_seg"]

    img_chexnet = transform_chexnet(image_pil).unsqueeze(0).to(DEVICE)
    img_seg_input = transform_seg(image_pil).unsqueeze(0).to(DEVICE)

    # Apply segmentation mask
    with torch.no_grad():
        mask = torch.sigmoid(models_dict["seg_model"](img_seg_input)).squeeze().cpu().numpy()
        mask = (mask > 0.5).astype(np.float32)

    # Apply mask to the image
    image_resized = image_pil.resize((512, 512))
    image_np = np.array(image_resized).astype(np.float32) / 255.0
    masked_image = image_np * np.expand_dims(mask, axis=-1)
    masked_pil = Image.fromarray((masked_image * 255).astype(np.uint8))
    img_segmented = transform_chexnet(masked_pil).unsqueeze(0).to(DEVICE)

    # Get probabilities from each model
    def get_prob(model, img_tensor):
        with torch.no_grad():
            out = model(img_tensor)
            return torch.softmax(out, dim=1)[0, 1].item()

    p1 = get_prob(models_dict["model_seg"], img_segmented)
    p2 = get_prob(models_dict["model_chex_all"], img_chexnet)
    p3 = get_prob(models_dict["model_chex_2"], img_chexnet)

    probs = [p1, p2, p3]
    final_prob = np.mean(probs)
    pred_class = int(final_prob >= threshold)
    label = "Abnormal" if pred_class else "Normal"

    # Optional print
    # print(f" Image: {image_path}")
    # print(f" Ensemble Probability: {final_prob:.4f}")
    # print(f" Prediction: {label}")

    return label


models_dict = load_ensemble_models()
def yolo_predict(image):
    """
    Uses the loaded YOLO model to predict findings in the given image (PIL.Image or numpy array).
    Returns:
        A list of dictionaries for each detection:
        {
            "Name": <class label>,
            "confidence": <confidence score>,
            "coordinates": [x1, y1, x2, y2]
        }
    """
    results = Yolo_model(image, conf=0.25)
    detections = []

    for result in results:
        boxes = result.boxes
        names = result.names
        for box in boxes:
            cls_id = int(box.cls[0].item())
            label = names[cls_id]
            confidence = float(box.conf[0].item())
            coords = box.xyxy[0].tolist()  # [x1, y1, x2, y2]
            detections.append({
                "Name": label,
                "confidence": confidence,
                "coordinates": coords
            })

    return detections



# Predict from DICOM
def classify (dicom_path):
    # Extract StudyInstanceUID if available
    ds = pydicom.dcmread(dicom_path, force =True)
    study_id = getattr(ds, "StudyInstanceUID", "Unknown")
    ds = pydicom.dcmread(dicom_path, force =True)
    patient_id = getattr(ds, "PatientID", "Unknown")
    predicted_class,full_d_image = predict_image_class(dicom_path, models_dict)
    if predicted_class == 1:
        # If abnormal, use YOLO for further classification
        yolo_labels = yolo_predict(full_d_image)
        if yolo_labels:
            finding = yolo_labels  # Take the first label from YOLO
    
    return { "result":predicted_class
            , "patient_id": patient_id
            , "StudyInstanceUID": study_id
            , "finding": finding if predicted_class == 1 else [{"Name": "Normal", "confidence": "unknown", "coordinates": [0, 0, 0, 0]}]}
           
     
  

# folder_path = "cases"

# for file_name in os.listdir(folder_path):
#     file_path = os.path.join(folder_path, file_name)
#     try:
#         result = classify(file_path)
#         print(f"{file_name}: {result}")
#     except Exception as e:
#         print(f"Error processing {file_name}: {e}")