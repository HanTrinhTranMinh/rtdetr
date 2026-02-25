# main.py
import os
import json
import argparse

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.rtdetr import RTDETR
from src.dataset import COCODetection, collate_fn
from src.criterion import SetCriterion, box_cxcywh_to_xyxy


def postprocess(outputs, img_size: int, score_thresh: float = 0.05):
    """
    outputs:
      - pred_logits: (N, Q, K+1)
      - pred_boxes : (N, Q, 4) in [0,1] cxcywh

    return:
      - scores: (N,Q)
      - labels: (N,Q) in [0..K-1]
      - boxes : (N,Q,4) xyxy in pixel coords
    """
    logits = outputs["pred_logits"]
    boxes = outputs["pred_boxes"]

    prob = logits.softmax(-1)
    scores, labels = prob[..., :-1].max(-1)  # ignore no-object

    xyxy = box_cxcywh_to_xyxy(boxes).clamp(0, 1)  # normalized
    xyxy = xyxy * img_size  # pixel
    return scores, labels, xyxy


@torch.no_grad()
def evaluate_coco(
    model,
    loader,
    coco_gt,
    img_size: int,
    out_json: str,
    device: str,
    cat_ids_sorted,
    score_thresh: float = 0.05,
):
    """
    cat_ids_sorted: list of coco category_ids in the SAME order used by dataset label mapping.
                   label i  -> category_id = cat_ids_sorted[i]
    """
    model.eval()
    results = []

    for imgs, targets in tqdm(loader, desc="Eval"):
        imgs = imgs.to(device, non_blocking=True)
        outputs = model(imgs)

        scores, labels, boxes = postprocess(outputs, img_size, score_thresh=score_thresh)

        for b in range(imgs.size(0)):
            image_id = int(targets[b]["image_id"].item())

            s = scores[b].detach().cpu()
            l = labels[b].detach().cpu()
            bb = boxes[b].detach().cpu()  # xyxy pixel

            keep = s > score_thresh
            s = s[keep]
            l = l[keep]
            bb = bb[keep]

            for i in range(len(s)):
                x1, y1, x2, y2 = bb[i].tolist()
                w = max(0.0, x2 - x1)
                h = max(0.0, y2 - y1)

                label_i = int(l[i].item())
                if label_i < 0 or label_i >= len(cat_ids_sorted):
                    continue

                results.append(
                    {
                        "image_id": image_id,
                        "category_id": int(cat_ids_sorted[label_i]),
                        "bbox": [float(x1), float(y1), float(w), float(h)],
                        "score": float(s[i].item()),
                    }
                )

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f)

    from pycocotools.cocoeval import COCOeval

    coco_dt = coco_gt.loadRes(out_json)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()


def train_one_epoch(model, criterion, loader, optimizer, device: str, epoch: int, epochs: int):
    model.train()
    pbar = tqdm(loader, desc=f"Epoch {epoch}/{epochs}")

    for imgs, targets in pbar:
        imgs = imgs.to(device, non_blocking=True)

        outputs = model(imgs)
        losses = criterion(outputs, targets)
        loss = losses["loss_total"]

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()

        pbar.set_postfix({k: float(v.detach().cpu()) for k, v in losses.items()})


def main():
    ap = argparse.ArgumentParser()

    # data
    ap.add_argument("--data_root", type=str, required=True, help="root folder contains train/val images + annotations")
    ap.add_argument("--train_img", type=str, default="train2017")
    ap.add_argument("--val_img", type=str, default="val2017")
    ap.add_argument("--ann_train", type=str, default="annotations/instances_train2017.json")
    ap.add_argument("--ann_val", type=str, default="annotations/instances_val2017.json")

    # model / train
    ap.add_argument("--num_classes", type=int, default=34)
    ap.add_argument("--img_size", type=int, default=640)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--num_queries", type=int, default=300)
    ap.add_argument("--eval_only", action="store_true")
    ap.add_argument("--ckpt", type=str, default="rtdetr_scratch.pth")
    ap.add_argument("--eval_every", type=int, default=10)
    ap.add_argument("--score_thresh", type=float, default=0.05)

    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_img_dir = os.path.join(args.data_root, args.train_img)
    val_img_dir = os.path.join(args.data_root, args.val_img)
    ann_train_path = os.path.join(args.data_root, args.ann_train)
    ann_val_path = os.path.join(args.data_root, args.ann_val)

    # datasets
    train_ds = COCODetection(train_img_dir, ann_train_path, img_size=args.img_size, training=True)
    val_ds = COCODetection(val_img_dir, ann_val_path, img_size=args.img_size, training=False)

    # IMPORTANT: category ids order used for mapping label->category_id
    # Ensure dataset has this attribute; if not, add it in src/dataset.py (see note below)
    cat_ids_sorted = getattr(val_ds, "cat_ids_sorted", None)
    if cat_ids_sorted is None:
        # fallback: rebuild from COCO object inside dataset
        cat_ids_sorted = sorted(val_ds.coco.getCatIds())

    train_loader = DataLoader(
        train_ds, batch_size=args.batch, shuffle=True, num_workers=4, pin_memory=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch, shuffle=False, num_workers=4, pin_memory=True, collate_fn=collate_fn
    )

    # model / loss
    model = RTDETR(num_classes=args.num_classes, hidden_dim=256, num_queries=args.num_queries).to(device)
    criterion = SetCriterion(num_classes=args.num_classes).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # load ckpt
    if os.path.exists(args.ckpt):
        ck = torch.load(args.ckpt, map_location="cpu")
        if "model" in ck:
            model.load_state_dict(ck["model"], strict=False)
        else:
            model.load_state_dict(ck, strict=False)

        if (not args.eval_only) and ("optim" in ck):
            optimizer.load_state_dict(ck["optim"])

    # eval only
    if args.eval_only:
        from pycocotools.coco import COCO

        coco_gt = COCO(ann_val_path)
        evaluate_coco(
            model=model,
            loader=val_loader,
            coco_gt=coco_gt,
            img_size=args.img_size,
            out_json="pred.json",
            device=device,
            cat_ids_sorted=cat_ids_sorted,
            score_thresh=args.score_thresh,
        )
        return

    # train loop
    for epoch in range(1, args.epochs + 1):
        train_one_epoch(model, criterion, train_loader, optimizer, device, epoch, args.epochs)

        torch.save({"model": model.state_dict(), "optim": optimizer.state_dict()}, args.ckpt)

        if args.eval_every > 0 and (epoch % args.eval_every == 0):
            from pycocotools.coco import COCO

            coco_gt = COCO(ann_val_path)
            evaluate_coco(
                model=model,
                loader=val_loader,
                coco_gt=coco_gt,
                img_size=args.img_size,
                out_json="pred.json",
                device=device,
                cat_ids_sorted=cat_ids_sorted,
                score_thresh=args.score_thresh,
            )


if __name__ == "__main__":
    main()