import os
import cv2
import torch
import numpy as np
import torchvision
import argparse
from torchvision.models.detection import SSDLite320_MobileNet_V3_Large_Weights


def get_model(num_classes):
    model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(weights=SSDLite320_MobileNet_V3_Large_Weights.DEFAULT)
    num_anchors = model.head.classification_head.module_list[0][1].out_channels // 91  # original num_classes for COCO

    # Change the classification head
    for i in range(len(model.head.classification_head.module_list)):
        conv_layer = model.head.classification_head.module_list[i][1]
        conv_layer.out_channels = num_anchors * num_classes
        model.head.classification_head.module_list[i][1] = conv_layer

    return model


def normalize_image(image):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = image.astype(np.float32) / 255.0  # Convert uint8 to float
    image = (image - mean) / std
    return image


def predict(image_path, model, device, detection_threshold):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = normalize_image(image)
    image = torch.from_numpy(image).permute(2, 0, 1).float().to(device).unsqueeze(0)
    model.eval()
    with torch.no_grad():
        outputs = model(image)
    boxes = outputs[0]['boxes'].data.cpu().numpy()
    scores = outputs[0]['scores'].data.cpu().numpy()

    keep_boxes = scores >= detection_threshold
    max_score_box = scores.argmax()

    if max_score_box not in np.where(keep_boxes)[0]:
        keep_boxes = np.append(keep_boxes, max_score_box)

    boxes = boxes[keep_boxes].astype(np.int32)
    return boxesdef predict(image_path, model, device, detection_threshold):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = normalize_image(image)
    image = torch.from_numpy(image).permute(2, 0, 1).float().to(device).unsqueeze(0)
    model.eval()
    with torch.no_grad():
        outputs = model(image)
    boxes = outputs[0]['boxes'].data.cpu().numpy()
    scores = outputs[0]['scores'].data.cpu().numpy()

    keep_boxes = scores >= detection_threshold
    max_score_box = scores.argmax()

    if max_score_box not in np.where(keep_boxes)[0]:
        keep_boxes = np.append(keep_boxes, max_score_box)

    boxes = boxes[keep_boxes].astype(np.int32)
    return boxes


def draw_boxes(image_path, boxes, save_path):
    image = cv2.imread(image_path)
    for box in boxes:
        cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 2)
    cv2.imwrite(save_path, image)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str)
    args = parser.parse_args()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = get_model(num_classes=2)
    model.load_state_dict(torch.load("ssdlite_weights.pth"))
    model.to(device)
    boxes = predict(args.source, model, device, detection_threshold=0.5)
    draw_boxes(args.source, boxes, args.source.split('.')[0]+'_out.png')


if __name__ == "__main__":
    main()
