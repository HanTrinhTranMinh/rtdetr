# Hungarian Matcher & Loss
# src/criterion.py
import torch
import torch.nn as nn
import torch.nn.functional as F

def box_cxcywh_to_xyxy(x):
    cx, cy, w, h = x.unbind(-1)
    b = [(cx - 0.5*w), (cy - 0.5*h),
         (cx + 0.5*w), (cy + 0.5*h)]
    return torch.stack(b, dim=-1)

def box_iou(boxes1, boxes2):
    # boxes: xyxy in [0,1]
    area1 = (boxes1[:,2]-boxes1[:,0]).clamp(min=0) * (boxes1[:,3]-boxes1[:,1]).clamp(min=0)
    area2 = (boxes2[:,2]-boxes2[:,0]).clamp(min=0) * (boxes2[:,3]-boxes2[:,1]).clamp(min=0)

    lt = torch.max(boxes1[:,None,:2], boxes2[:,:2])
    rb = torch.min(boxes1[:,None,2:], boxes2[:,2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:,:,0] * wh[:,:,1]
    union = area1[:,None] + area2 - inter + 1e-6
    iou = inter / union
    return iou, union

def generalized_box_iou(boxes1, boxes2):
    iou, _ = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:,None,:2], boxes2[:,:2])
    rb = torch.max(boxes1[:,None,2:], boxes2[:,2:])
    wh = (rb - lt).clamp(min=0)
    area_c = wh[:,:,0] * wh[:,:,1] + 1e-6

    area1 = (boxes1[:,2]-boxes1[:,0]).clamp(min=0) * (boxes1[:,3]-boxes1[:,1]).clamp(min=0)
    area2 = (boxes2[:,2]-boxes2[:,0]).clamp(min=0) * (boxes2[:,3]-boxes2[:,1]).clamp(min=0)

    lt2 = torch.max(boxes1[:,None,:2], boxes2[:,:2])
    rb2 = torch.min(boxes1[:,None,2:], boxes2[:,2:])
    wh2 = (rb2 - lt2).clamp(min=0)
    inter = wh2[:,:,0] * wh2[:,:,1]
    union = area1[:,None] + area2 - inter + 1e-6

    giou = iou - (area_c - union) / area_c
    return giou

def hungarian_match(cost):
    # cost: (Q, T) on CPU for simplicity
    # dùng linear_sum_assignment từ scipy nếu có; nếu không, fallback greedy.
    try:
        from scipy.optimize import linear_sum_assignment
        row_ind, col_ind = linear_sum_assignment(cost)
        return row_ind, col_ind
    except Exception:
        # greedy fallback (kém hơn, nhưng chạy được)
        q, t = cost.shape
        used_t = set()
        rows, cols = [], []
        for i in range(q):
            j = int(cost[i].argmin())
            while j in used_t and len(used_t) < t:
                cost[i, j] = cost[i].max() + 1
                j = int(cost[i].argmin())
            if j not in used_t:
                used_t.add(j)
                rows.append(i); cols.append(j)
        return rows, cols

class SetCriterion(nn.Module):
    def __init__(self, num_classes=80, lambda_cls=1.0, lambda_l1=5.0, lambda_giou=2.0):
        super().__init__()
        self.num_classes = num_classes
        self.lambda_cls = lambda_cls
        self.lambda_l1 = lambda_l1
        self.lambda_giou = lambda_giou
        self.empty_weight = torch.ones(num_classes + 1)
        self.empty_weight[-1] = 0.1  # down-weight no-object

    def forward(self, outputs, targets):
        pred_logits = outputs["pred_logits"]  # N,Q,K+1
        pred_boxes = outputs["pred_boxes"]    # N,Q,4

        device = pred_logits.device
        self.empty_weight = self.empty_weight.to(device)

        losses = {"loss_cls": 0.0, "loss_bbox": 0.0, "loss_giou": 0.0}
        bs, Q, _ = pred_logits.shape

        for b in range(bs):
            tgt_labels = targets[b]["labels"].to(device)
            tgt_boxes = targets[b]["boxes"].to(device)

            if tgt_boxes.numel() == 0:
                # all no-object
                target_classes = torch.full((Q,), self.num_classes, dtype=torch.long, device=device)
                losses["loss_cls"] += F.cross_entropy(pred_logits[b], target_classes, weight=self.empty_weight)
                continue

            # cost matrix
            out_prob = pred_logits[b].softmax(-1)[:, :self.num_classes]  # Q,K
            cost_class = -out_prob[:, tgt_labels]                        # Q,T

            cost_bbox = torch.cdist(pred_boxes[b], tgt_boxes, p=1)       # Q,T

            giou = generalized_box_iou(
                box_cxcywh_to_xyxy(pred_boxes[b]),
                box_cxcywh_to_xyxy(tgt_boxes)
            )                                                           # Q,T
            cost_giou = -giou

            C = (1.0 * cost_class + 5.0 * cost_bbox + 2.0 * cost_giou).detach().cpu().numpy()
            qi, ti = hungarian_match(C)

            qi = torch.as_tensor(qi, device=device, dtype=torch.long)
            ti = torch.as_tensor(ti, device=device, dtype=torch.long)

            # classification targets
            target_classes = torch.full((Q,), self.num_classes, dtype=torch.long, device=device)
            target_classes[qi] = tgt_labels[ti]

            losses["loss_cls"] += F.cross_entropy(pred_logits[b], target_classes, weight=self.empty_weight)

            # bbox losses on matched
            pb = pred_boxes[b][qi]
            tb = tgt_boxes[ti]
            losses["loss_bbox"] += F.l1_loss(pb, tb, reduction="mean")

            g = generalized_box_iou(box_cxcywh_to_xyxy(pb), box_cxcywh_to_xyxy(tb))
            losses["loss_giou"] += (1.0 - torch.diag(g)).mean()

        # average over batch
        for k in losses:
            losses[k] = losses[k] / bs

        total = self.lambda_cls*losses["loss_cls"] + self.lambda_l1*losses["loss_bbox"] + self.lambda_giou*losses["loss_giou"]
        losses["loss_total"] = total
        return losses