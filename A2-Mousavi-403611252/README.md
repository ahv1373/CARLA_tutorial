# Overview
In this assignment, a **Convolutional Neural Network (CNN)** is trained to classify weather conditions (Day, Night, Rain, Fog) using images captured in different environments from the **CARLA simulator**.

The project is divided into two main parts:
1. **Data Collection** from CARLA in different weather settings.
2. **CNN Model Training** for weather classification and visualization using a confusion matrix.

---

# Part 1: Data Collection with CARLA

# Features
- A camera is mounted on a moving autopilot vehicle in CARLA.
- Four weather conditions simulated:
  - Day
  - Night
  - Rain
  - Fog
- For each condition, **50 images** are captured and saved.

# Output
Captured images are stored in:  
`carla_dataset_a2/`  
Each image filename follows this pattern:  
`<weather_name>_<image_id>.png`

---

# Part 2: CNN Model for Weather Classification

# Features
- A custom CNN architecture is trained on the collected dataset.
- Images are resized to **128x128** and split into **80% training** and **20% testing**.
- The model outputs the **weather label** for a given image.
- After training, a **confusion matrix** visualizes classification performance.

# Results
- Model trained for **10 epochs** using Adam optimizer.
- Final accuracy and loss are printed.
- Confusion matrix is saved as:  
  `confusion_matrix.png`

# Model Architecture
```plaintext
Conv2D → ReLU → MaxPool
Conv2D → ReLU → MaxPool
Conv2D → ReLU → MaxPool
Flatten → FC (128) → ReLU → FC (4)
```

# Output Files
- `weather_cnn_model.pth`: Trained model weights
- `confusion_matrix.png`: Model performance visualization

---