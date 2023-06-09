# import the necessary packages
from typing import List, Tuple

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
        # choices fo model_name are: "frcnn-resnet", "frcnn-mobilenet", "retinanet"
        if model_name not in ["frcnn-resnet", "frcnn-mobilenet", "retinanet"]:
            raise ValueError("Supported model names are: 'frcnn-resnet', 'frcnn-mobilenet', 'retinanet'")
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold

    def load(self):
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
        print(f"Model {self.model_name} loaded successfully in {time.time() - start_time} seconds.")

    def __detect__(self, preprocessed_img: torch.FloatTensor) -> Tuple[List[list], List[int], List[float]]:
        bboxes_list, class_index_list, scores_list = [], [], []
        detections = self.model(preprocessed_img)[0]
        # loop over the detections
        for i in range(0, len(detections["boxes"])):
            # extract the confidence (i.e., probability) associated with the
            # prediction
            confidence = detections["scores"][i]
            # filter out weak detections by ensuring the confidence is
            # greater than the minimum confidence
            if confidence > self.confidence_threshold:
                # extract the index of the class label from the detections,
                # then compute the (x, y)-coordinates of the bounding box
                # for the object
                idx = int(detections["labels"][i])
                box = detections["boxes"][i].detach().cpu().numpy()
                start_x, start_y, end_x, end_y = box.astype("int")
                bboxes_list.append([start_x, start_y, end_x, end_y])
                class_index_list.append(idx)
                scores_list.append(confidence)
        return bboxes_list, class_index_list, scores_list

    def __preprocess_frame__(self, frame: np.array) -> torch.FloatTensor:
        # convert the image from BGR to RGB channel ordering and change the
        # image from channels last to channels first ordering
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = image.transpose((2, 0, 1))

        # add the batch dimension, scale the raw pixel intensities to the
        # range [0, 1], and convert the image to a floating point tensor
        image = np.expand_dims(image, axis=0)
        image = image / 255.0
        image = torch.FloatTensor(image)
        image = image.to(self.device)
        return image

    def visualize(self, frame, bboxes_list: List[list], class_index_list: List[int], scores_list: List[float]):
        for bbox, class_index, confidence in zip(bboxes_list, class_index_list, scores_list):
            start_x, start_y, end_x, end_y = bbox
            label = "{}: {:.2f}%".format(self.classes[class_index], confidence * 100)
            # draw the prediction on the frame
            cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), self.colors[class_index], 2)
            y = start_y - 15 if start_y - 15 > 15 else start_y + 15
            cv2.putText(frame, label, (start_x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors[class_index], 2)

    def exec(self, frame=np.array):
        preprocessed_img = self.__preprocess_frame__(frame)
        bboxes_list, class_index_list, scores_list = self.__detect__(preprocessed_img)
        return bboxes_list, class_index_list, scores_list


if __name__ == "__main__":
    import glob
    import os

    img_dir = "PATH TO THE DIRECTORY CONTAINING THE IMAGES"
    img_list = glob.glob(img_dir + "/*.jpg")  # Get all the jpg files in the directory
    img_list.sort()
    object_detector = TorchObjectDetection(model_name="frcnn-resnet")
    object_detector.load()
    for img_name in img_list:
        img_raw_name = os.path.basename(img_name)
        frame_ = cv2.imread(img_name)
        bboxes_list_, class_index_list_, scores_list_ = object_detector.exec(frame_)
        object_detector.visualize(frame_, bboxes_list_, class_index_list_, scores_list_)
        cv2.imshow("Frame", frame_)
        key = cv2.waitKey(1)  # wait for 1ms before moving on to the next frame
        if key == ord("q"):
            break
        elif key == ord("s"):
            cv2.imwrite(f"{os.path.dirname(__file__)}/{img_raw_name}", frame_)
    cv2.destroyAllWindows()
