import os
import json
import argparse
import torch
import sys
import logging
from datetime import datetime
from torch.utils.data import DataLoader
from tqdm import tqdm

# Import các thành phần từ project của bạn
from models.rtdetr import RTDETR
from src.dataset import COCODetection, collate_fn
from src.criterion import SetCriterion, box_cxcywh_to_xyxy

# --- CẤU HÌNH LOGGING ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("training_debug.log"), # Ghi vào file này
        logging.StreamHandler(sys.stdout)          # Vẫn hiện ở terminal
    ]
)
logger = logging.getLogger(__name__)

def postprocess(outputs, img_size: int, score_thresh: float = 0.05):
    logits = outputs["pred_logits"]
    boxes = outputs["pred_boxes"]
    prob = logits.softmax(-1)
    scores, labels = prob[..., :-1].max(-1) 
    xyxy = box_cxcywh_to_xyxy(boxes).clamp(0, 1)
    xyxy = xyxy * img_size
    return scores, labels, xyxy

@torch.no_grad()
def evaluate_coco(model, loader, coco_gt, img_size, out_json, device, cat_ids_sorted, score_thresh=0.05):
    model.eval()
    results = []
    logger.info("Starting Evaluation...")
    
    try:
        for imgs, targets in tqdm(loader, desc="Eval"):
            imgs = imgs.to(device, non_blocking=True)
            outputs = model(imgs)
            scores, labels, boxes = postprocess(outputs, img_size, score_thresh=score_thresh)

            for b in range(imgs.size(0)):
                image_id = int(targets[b]["image_id"].item())
                s, l, bb = scores[b].cpu(), labels[b].cpu(), boxes[b].cpu()
                keep = s > score_thresh
                s, l, bb = s[keep], l[keep], bb[keep]

                for i in range(len(s)):
                    x1, y1, x2, y2 = bb[i].tolist()
                    w, h = max(0.0, x2 - x1), max(0.0, y2 - y1)
                    label_idx = int(l[i].item())
                    if label_idx < 0 or label_idx >= len(cat_ids_sorted): continue
                    results.append({
                        "image_id": image_id,
                        "category_id": int(cat_ids_sorted[label_idx]),
                        "bbox": [float(x1), float(y1), float(w), float(h)],
                        "score": float(s[i].item()),
                    })
        
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(results, f)

        from pycocotools.cocoeval import COCOeval
        coco_dt = coco_gt.loadRes(out_json)
        coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
        coco_eval.evaluate(); coco_eval.accumulate(); coco_eval.summarize()
        logger.info("Evaluation finished successfully.")
    except Exception as e:
        logger.error(f"Error during evaluation: {e}")

def train_one_epoch(model, criterion, loader, optimizer, device, epoch, epochs):
    model.train()
    criterion.train()
    pbar = tqdm(loader, desc=f"Epoch {epoch}/{epochs}")

    for i, (imgs, targets) in enumerate(pbar):
        try:
            # Heartbeat log: Cứ mỗi 50 steps ghi log một lần để biết máy chưa treo
            if i % 50 == 0:
                logger.info(f"Epoch {epoch} - Step {i} - GPU Memory: {torch.cuda.memory_allocated(0)/1024**2:.2f}MB")

            imgs = imgs.to(device, non_blocking=True)
            targets = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets]

            outputs = model(imgs)
            losses = criterion(outputs, targets)
            loss = losses["loss_total"]

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            optimizer.step()

            pbar.set_postfix({k: f"{v.item():.3f}" for k, v in losses.items() if "loss" in k})
            
        except Exception as e:
            logger.error(f"CRITICAL ERROR at Epoch {epoch}, Step {i}: {e}")
            # Nếu lỗi CUDA, thử giải phóng bộ nhớ ngay lập tức
            torch.cuda.empty_cache()
            raise e

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument("--train_img", type=str, default="train")
    ap.add_argument("--val_img", type=str, default="valid")
    ap.add_argument("--ann_train", type=str, default="train/_annotations.coco.json")
    ap.add_argument("--ann_val", type=str, default="valid/_annotations.coco.json")
    ap.add_argument("--num_classes", type=int, default=34)
    ap.add_argument("--img_size", type=int, default=640)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--num_queries", type=int, default=300)
    ap.add_argument("--eval_only", action="store_true")
    ap.add_argument("--ckpt", type=str, default="results/rtdetr_scratch.pth")
    ap.add_argument("--eval_every", type=int, default=10)
    ap.add_argument("--score_thresh", type=float, default=0.05)
    args = ap.parse_args()

    logger.info("--- Starting New Training Session ---")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")

    # Datasets
    try:
        train_ds = COCODetection(os.path.join(args.data_root, args.train_img), 
                                 os.path.join(args.data_root, args.ann_train), 
                                 img_size=args.img_size, training=True)
        val_ds = COCODetection(os.path.join(args.data_root, args.val_img), 
                               os.path.join(args.data_root, args.ann_val), 
                               img_size=args.img_size, training=False)
        logger.info("Datasets loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        return

    cat_ids_sorted = getattr(val_ds, "cat_ids_sorted", sorted(val_ds.coco.getCatIds()))

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, 
                              num_workers=0, pin_memory=False, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, 
                            num_workers=0, pin_memory=False, collate_fn=collate_fn)

    model = RTDETR(num_classes=args.num_classes, hidden_dim=256, num_queries=args.num_queries).to(device)
    criterion = SetCriterion(num_classes=args.num_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    os.makedirs(os.path.dirname(args.ckpt), exist_ok=True)
    if os.path.exists(args.ckpt):
        logger.info(f"Loading checkpoint from {args.ckpt}")
        ck = torch.load(args.ckpt, map_location="cpu")
        model.load_state_dict(ck["model"] if "model" in ck else ck, strict=False)
        if not args.eval_only and "optim" in ck:
            optimizer.load_state_dict(ck["optim"])

    if args.eval_only:
        from pycocotools.coco import COCO
        evaluate_coco(model, val_loader, COCO(os.path.join(args.data_root, args.ann_val)), 
                      args.img_size, "pred.json", device, cat_ids_sorted)
        return

    # --- TRAIN LOOP ---
    try:
        for epoch in range(1, args.epochs + 1):
            train_one_epoch(model, criterion, train_loader, optimizer, device, epoch, args.epochs)
            
            torch.save({"model": model.state_dict(), "optim": optimizer.state_dict(), "epoch": epoch}, args.ckpt)
            logger.info(f"Epoch {epoch} saved.")

            if args.eval_every > 0 and (epoch % args.eval_every == 0):
                from pycocotools.coco import COCO
                evaluate_coco(model, val_loader, COCO(os.path.join(args.data_root, args.ann_val)), 
                              args.img_size, "pred.json", device, cat_ids_sorted)
            
            if device == "cuda": torch.cuda.empty_cache()
            
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user (KeyboardInterrupt). Saving state...")
        torch.save({"model": model.state_dict(), "optim": optimizer.state_dict()}, "interrupted_ckpt.pth")
    except Exception as e:
        logger.error(f"Uncaught Exception: {e}")
    finally:
        logger.info("Session ended.")

if __name__ == "__main__":
    main()