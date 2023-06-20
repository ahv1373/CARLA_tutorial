import os
import time
from typing import Tuple

import cv2
import numpy as np
from tensorflow import keras


class AdverseWeatherClassifier:
    def __init__(self, model_path, model_input_size: Tuple[int, int] = (256, 256)) -> None:
        self.model = None
        self.model_path = model_path
        self.model_input_size = model_input_size
        self.class_labels = ['ClearNoon', 'ClearSunset','fog','HardRainSunset']

    def load(self):
        start_time = time.time()
        self.model = keras.models.load_model(self.model_path)
        print("Model loaded in {} seconds.".format(time.time() - start_time))

    def __preprocess_frame__(self, frame: np.ndarray) -> np.ndarray:
        frame = cv2.resize(frame, self.model_input_size)
        frame = frame.astype('float32') / 255.0
        frame = np.expand_dims(frame, axis=0)
        return frame

    def predict(self, preprocessed_frame: np.ndarray) -> str:
        if self.model is not None:
            predictions = self.model.predict(preprocessed_frame)
        else:
            raise ValueError("Model is not loaded. Please call load() method first.")
        predicted_class = self.class_labels[np.argmax(predictions)]
        return predicted_class

    def exec(self, frame: np.ndarray) -> str:
        preprocessed_frame = self.__preprocess_frame__(frame)
        predicted_class = self.predict(preprocessed_frame)
        return predicted_class


if __name__ == "__main__":
    img_dir = r"D:\Set\TestFinal"
    model_path_ = r"D:\output\checkpoints\best_model.h5"
    adverse_weather_classifier = AdverseWeatherClassifier(model_path_)
    adverse_weather_classifier.load()
    for root, dirs, files in os.walk(img_dir):
        # shuffle the files
        np.random.shuffle(files)
        for file in files:
            img = cv2.imread(os.path.join(root, file))
            visualization_img = img.copy()
            predicted_class_ = adverse_weather_classifier.exec(img)
            cv2.putText(visualization_img, predicted_class_, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("Image", visualization_img)
            cv2.waitKey(0)
    cv2.destroyAllWindows()
