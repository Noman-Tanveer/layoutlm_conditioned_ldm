import os
import json
from typing import TypedDict
from PIL import Image

import torch
import numpy as np
from transformers import LayoutLMv3Processor, LayoutLMv3Tokenizer, LayoutLMv3ImageProcessor
from torchvision import transforms
from torch.utils.data import Dataset

embedding = TypedDict('embedding', {'input_ids': torch.Tensor, 'bbox': torch.Tensor, 'attention_mask': torch.Tensor, 'pixel_values': torch.Tensor})
target_transforms = transforms.Compose([transforms.PILToTensor()])

class FUNSD(Dataset):

    def __init__(self, cfg):
        base_dir = cfg.data_dir
        self.img_dir = os.path.join(base_dir, "images")
        self.annotations_dir = os.path.join(base_dir, "annotations")
        self.img_files = os.listdir(self.img_dir)
        self.tokenizer = LayoutLMv3Tokenizer.from_pretrained("microsoft/layoutlmv3-base")
        self.feature_extractor = LayoutLMv3ImageProcessor("microsoft/layoutlmv3-base", size={"height": cfg.resolution, "width": cfg.resolution}, apply_ocr=False)
        self.processor = LayoutLMv3Processor(feature_extractor=self.feature_extractor, tokenizer=self.tokenizer)
        self.transform = transforms.Compose([transforms.Normalize([0.5], [0.5])])

    def __len__(self):
        return len(self.img_files)

    def read_img(self, img_path:str) -> bytes:
        img = Image.open(img_path)
        rgb_img = img.convert('RGB')
        return rgb_img

    def load_labels(self, lbl_file:str) -> dict:
        with open(lbl_file) as f:
            labels =json.load(f)
        return labels

    def get_words_and_boxes(self, lbl_dict: dict) -> dict:
        words = []
        boxes = []
        doc_class = list(lbl_dict.keys())[0]
        for line in lbl_dict[doc_class]:
            for word in line["words"]:
                words.append(word["text"])
                boxes.append(word["box"])
        assert len(boxes) == len(words)
        return words, boxes

    def __call__(self, img_name):
        img_path = os.path.join(self.img_dir, img_name)
        labels_path = os.path.join(self.annotations_dir, os.path.basename(img_path).replace("png", "json"))
        image = self.read_img(img_path)
        labels = self.load_labels(labels_path)
        words, boxes = self.get_words_and_boxes(labels)
        encoding = self.processor(image, words, boxes=boxes, return_tensors="pt", padding="max_length", max_length=512, truncation=True)
        for key, val in encoding.items():
            if key == "pixel_values":
                encoding[key] = self.transform(val)
        return encoding

    def __getitem__(self, idx) -> embedding:
        encoding = dict()
        img_name = self.img_files[idx]
        img_path = os.path.join(self.img_dir, img_name)
        labels_path = os.path.join(self.annotations_dir, os.path.basename(img_path).replace("png", "json"))
        image = self.read_img(img_path)
        labels = self.load_labels(labels_path)
        words, boxes = self.get_words_and_boxes(labels)
        encoding = self.processor(image, words, boxes=boxes, return_tensors="pt", padding="max_length", max_length=512, truncation=True)
        for key, val in encoding.items():
            if key == "pixel_values":
                encoding[key] = self.transform(val)
            encoding[key] = torch.squeeze(val, 0)
        return encoding

if __name__ == "__main__":
    import types
    cfg = types.SimpleNamespace()
    cfg.resolution = 512
    cfg.data_dir = "../FUNSD_dataset/training_data"
    dataset = FUNSD(cfg=cfg)
    print(next(iter(dataset)).keys())
