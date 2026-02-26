# src/dataset.py
import os
import torch
from torch.utils.data import Dataset
from pycocotools.coco import COCO
from PIL import Image
import torchvision.transforms as T


class COCODetection(Dataset):
    def __init__(self, img_folder, ann_file, img_size=640, training=True, ignore_cat_ids=(0,)):
        self.coco = COCO(ann_file)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.img_folder = img_folder
        self.img_size = img_size
        self.training = training

        # ---- category mapping (filter by ID) ----
        all_cat_ids = sorted(self.coco.getCatIds())
        self.cat_ids_sorted = [cid for cid in all_cat_ids if cid not in set(ignore_cat_ids)]

        self.cat2label = {cid: i for i, cid in enumerate(self.cat_ids_sorted)}
        self.label2cat = {i: cid for i, cid in enumerate(self.cat_ids_sorted)}

        self.transform = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __getitem__(self, index):
        img_id = self.ids[index]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)

        img_info = self.coco.loadImgs(img_id)[0]
        path = img_info["file_name"]
        w_img = float(img_info["width"])
        h_img = float(img_info["height"])

        img = Image.open(os.path.join(self.img_folder, path)).convert("RGB")

        boxes = []
        labels = []
        for obj in anns:
            if obj.get("iscrowd", 0) == 1:
                continue

            cat_id = obj.get("category_id", None)
            if cat_id not in self.cat2label:
                # skip ignored category (e.g., printed-circuit-board id=0)
                continue

            x, y, w, h = obj["bbox"]
            if w <= 1 or h <= 1:
                continue

            # cxcywh normalized in [0,1]
            cx = (x + w / 2.0) / w_img
            cy = (y + h / 2.0) / h_img
            nw = w / w_img
            nh = h / h_img

            # clamp to [0,1]
            cx = min(max(cx, 0.0), 1.0)
            cy = min(max(cy, 0.0), 1.0)
            nw = min(max(nw, 0.0), 1.0)
            nh = min(max(nh, 0.0), 1.0)

            if nw <= 0 or nh <= 0:
                continue

            boxes.append([cx, cy, nw, nh])
            labels.append(self.cat2label[cat_id])

        img_t = self.transform(img)

        target = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32) if len(boxes) else torch.zeros((0, 4), dtype=torch.float32),
            "labels": torch.as_tensor(labels, dtype=torch.long) if len(labels) else torch.zeros((0,), dtype=torch.long),
            "image_id": torch.tensor([img_id]),
        }
        return img_t, target

    def __len__(self):
        return len(self.ids)


def collate_fn(batch):
    imgs, targets = zip(*batch)
    imgs = torch.stack(list(imgs), dim=0)
    return imgs, list(targets)