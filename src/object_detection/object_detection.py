from typing import List, Tuple
import glob
import os

from torchvision.models import detection
import numpy as np
import torch
import cv2
import time
from src.object_detection.coco_classes import COCOUtils


class TorchObjectDetection:
    def __init__(self, model_name: str = "frcnn-resnet", confidence_threshold: float = 0.5):
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.classes = COCOUtils().coco_classes_list
        self.colors = np.random.uniform(0, 255, size=(len(self.classes), 3))
        supported_model_names = ["frcnn-resnet", "frcnn-mobilenet", "retinanet"]
        if model_name not in supported_model_names:
            raise ValueError(f"Supported model names are: {', '.join(supported_model_names)}")
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold

    def load_model(self):
        start_time = time.time()
        if self.model_name == "frcnn-resnet":
            self.model = detection.fasterrcnn_resnet50_fpn(pretrained=True, progress=True,
                                                           num_classes=len(self.classes),
                                                           pretrained_backbone=True).to(self.device)
        elif self.model_name == "frcnn-mobilenet":
            self.model = detection.fasterrcnn_mobilenet_v3_large_320_fpn(pretrained=True, progress=True,
                                                                         num_classes=len(self.classes),
                                                                         pretrained_backbone=True).to(self.device)
        elif self.model_name == "retinanet":
            self.model = detection.retinanet_resnet50_fpn(pretrained=True, progress=True,
                                                          num_classes=len(self.classes),
                                                          pretrained_backbone=True).to(self.device)
        self.model.eval()
        print(f"Model {self.model_name} loaded successfully in {time.time() - start_time:.2f} seconds.")

    def __detect__(self, preprocessed_img: torch.FloatTensor) -> Tuple[List[list], List[int], List[float]]:
        bboxes_list, class_index_list, scores_list = [], [], []
        detections = self.model(preprocessed_img)[0]
        for i in range(len(detections["boxes"])):
            confidence = detections["scores"][i]
            if confidence > self.confidence_threshold:
                idx = int(detections["labels"][i])
                box = detections["boxes"][i].detach().cpu().numpy()
                start_x, start_y, end_x, end_y = box.astype("int")
                bboxes_list.append([start_x, start_y, end_x, end_y])
                class_index_list.append(idx)
                scores_list.append(confidence)
        return bboxes_list, class_index_list, scores_list

    def __preprocess_frame__(self, frame: np.array) -> torch.FloatTensor:
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = image.transpose((2, 0, 1))
        image = np.expand_dims(image, axis=0) / 255.0
        image = torch.FloatTensor(image).to(self.device)
        return image

    def visualize(self, frame, bboxes_list: List[list], class_index_list: List[int], scores_list: List[float]):
        for bbox, class_index, confidence in zip(bboxes_list, class_index_list, scores_list):
            start_x, start_y, end_x, end_y = bbox
            label = f"{self.classes[class_index]}: {confidence * 100:.2f}%"
            cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), self.colors[class_index], 2)
            y = start_y - 15 if start_y - 15 > 15 else start_y + 15
            cv2.putText(frame, label, (start_x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors[class_index], 2)

    def execute(self, frame: np.array) -> Tuple[List[list], List[int], List[float]]:
        preprocessed_img = self.preprocess_frame(frame)
        return self.detect(preprocessed_img)


if __name__ == "__main__":
    img_dir = "PATH TO THE DIRECTORY CONTAINING THE IMAGES"
    img_list = glob.glob(os.path.join(img_dir, "*.jpg"))  # Get all the jpg files in the directory
    img_list.sort()
    object_detector = TorchObjectDetection(model_name="frcnn-resnet")
    object_detector.load_model()
    for img_name in img_list:
        img_raw_name = os.path.basename(img_name)
        frame = cv2.imread(img_name)
        bboxes_list, class_index_list, scores_list = object_detector.execute(frame)
        object_detector.visualize(frame, bboxes_list, class_index_list, scores_list)
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1)  # wait for 1ms before moving on to the next frame
        if key == ord("q"):
            break
        elif key == ord("s"):
            cv2.imwrite(os.path.join(os.path.dirname(__file__), img_raw_name), frame)
    cv2.destroyAllWindows()
