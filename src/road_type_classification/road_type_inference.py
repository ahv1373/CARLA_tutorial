import os

import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image

from road_type_cnn_classifier import RoadTypeCNNClassifier  # Import your trained model

class_names = ['offroad', 'onroad']

# Load the trained model
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'saved_model', 'road_type_classifier_final.pth')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = RoadTypeCNNClassifier().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# Define image preprocessing (same as used during training)
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])


def predict(image_path: str, ground_truth=None):
    # Read image using OpenCV
    image = cv2.imread(image_path)
    original_image = image.copy()  # Keep a copy for visualization
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

    # Preprocess image
    image = transform(Image.fromarray(image))
    image = image.unsqueeze(0).to(device)  # Add batch dimension & move to GPU/CPU

    # Inference
    with torch.no_grad():
        output = model(image)
        prediction = (output.item() > 0.5)  # Threshold at 0.5 for binary classification

    # Get predicted class name
    predicted_class = class_names[int(prediction)]
    gt_text = f"GT: {ground_truth}" if ground_truth is not None else "GT: Unknown"
    pred_text = f"Prediction: {predicted_class}"

    # Visualization using OpenCV
    cv2.putText(original_image, gt_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(original_image, pred_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Show image
    cv2.imshow("Road Type Classification", original_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Example Usage
if __name__ == "__main__":
    data_dir: str = os.path.join(os.path.dirname(__file__), '..' , 'data', 'test')
    # walk through the test directory and predict the road type for each image
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.png'):
                image_path_ = os.path.join(root, file)
                ground_truth_ = os.path.basename(root)
                predict(image_path_, ground_truth_)
