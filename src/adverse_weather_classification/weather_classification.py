import os
import time
from typing import Tuple

import cv2
import numpy as np
from tensorflow import keras


class AdverseWeatherClassifier:
    def __init__(self, model_path: str, model_input_size: Tuple[int, int] = (256, 256)):
        self.model = None
        self.model_path = model_path
        self.model_input_size = model_input_size
        self.class_labels = ['day', 'night']

    def load_model(self):
        start_time = time.time()
        self.model = keras.models.load_model(self.model_path)
        print("Model loaded in {:.2f} seconds.".format(time.time() - start_time))

    def __preprocess_frame__(self, frame: np.ndarray) -> np.ndarray:
        frame = cv2.resize(frame, self.model_input_size).astype('float32') / 255.0
        frame = np.expand_dims(frame, axis=0)
        return frame

    def predict(self, preprocessed_frame: np.ndarray) -> str:
        if self.model is None:
            raise ValueError("Model is not loaded. Please call load_model() method first.")
        predictions = self.model.predict(preprocessed_frame)
        predicted_class = self.class_labels[np.argmax(predictions)]
        return predicted_class

    def execute(self, frame: np.ndarray) -> str:
        preprocessed_frame = self.preprocess_frame(frame)
        predicted_class = self.predict(preprocessed_frame)
        return predicted_class


if __name__ == "__main__":
    img_dir = "/home/ahv/PycharmProjects/Visual-Inertial-Odometry/simulation/CARLA/output/testing_imgs"
    model_path = "/home/ahv/PycharmProjects/Visual-Inertial-Odometry/simulation/CARLA/output/checkpoint_directory"
    adverse_weather_classifier = AdverseWeatherClassifier(model_path)
    adverse_weather_classifier.load_model()
    for root, dirs, files in os.walk(img_dir):
        # Shuffle the files
        np.random.shuffle(files)
        for file in files:
            img = cv2.imread(os.path.join(root, file))
            visualization_img = img.copy()
            predicted_class = adverse_weather_classifier.execute(img)
            cv2.putText(visualization_img, predicted_class, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("Image", visualization_img)
            cv2.waitKey(0)
    cv2.destroyAllWindows()
