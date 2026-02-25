# Xử lý COCO data
# src/dataset.py
import os
import torch
from torch.utils.data import Dataset
from pycocotools.coco import COCO
from PIL import Image
import torchvision.transforms as T

class COCODetection(Dataset):
    def __init__(self, img_folder, ann_file, img_size=640, training=True):
        self.coco = COCO(ann_file) #
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.img_folder = img_folder
        self.img_size = img_size
        self.cat_ids_sorted = sorted(self.coco.getCatIds()) #
        
        self.transform = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __getitem__(self, index):
        img_id = self.ids[index]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        target = self.coco.loadAnns(ann_ids)
        
        img_info = self.coco.loadImgs(img_id)[0]
        path = img_info['file_name']
        img = Image.open(os.path.join(self.img_folder, path)).convert('RGB')
        
        boxes = []
        labels = []
        for obj in target:
            x, y, w, h = obj['bbox']
            # Chuyển sang cxcywh normalized cho SetCriterion
            cx = (x + w/2) / img_info['width']
            cy = (y + h/2) / img_info['height']
            nw = w / img_info['width']
            nh = h / img_info['height']
            boxes.append([cx, cy, nw, nh])
            labels.append(self.cat_ids_sorted.index(obj['category_id']))

        return self.transform(img), {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32),
            "labels": torch.as_tensor(labels, dtype=torch.long),
            "image_id": torch.tensor([img_id])
        }

    def __len__(self):
        return len(self.ids)

def collate_fn(batch):
    return tuple(zip(*batch)) #