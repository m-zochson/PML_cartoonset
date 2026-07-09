"""Attribute classifier used as the evaluation instrument."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from cartoon_diffusion.data import build_dataloader


class AttributeClassifier(nn.Module):
    """Shared conv backbone + one linear head per attribute."""

    def __init__(self, attribute_dims, in_ch=3, base=32):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(in_ch, base, 3, stride=2, padding=1),
            nn.BatchNorm2d(base),
            nn.ReLU(),
            nn.Conv2d(base, base * 2, 3, stride=2, padding=1),
            nn.BatchNorm2d(base * 2),
            nn.ReLU(),
            nn.Conv2d(base * 2, base * 4, 3, stride=2, padding=1),
            nn.BatchNorm2d(base * 4),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.heads = nn.ModuleList([nn.Linear(base * 4, k) for k in attribute_dims])

    def forward(self, x):
        feat = self.backbone(x)
        return [head(feat) for head in self.heads]


def train_classifier(ds, attribute_dims, device, epochs=8, lr=1e-3, batch=128, workers=0):
    clf = AttributeClassifier(attribute_dims).to(device)
    loader = build_dataloader(ds, batch_size=batch, num_workers=workers)
    opt = torch.optim.Adam(clf.parameters(), lr=lr)
    clf.train()
    for ep in range(1, epochs + 1):
        total, correct, seen = 0.0, torch.zeros(len(attribute_dims)), 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = clf(x)
            loss = sum(
                F.cross_entropy(logits[i], y[:, i])
                for i in range(len(attribute_dims))
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * x.size(0)
            for i in range(len(attribute_dims)):
                correct[i] += (logits[i].argmax(1) == y[:, i]).sum().cpu()
            seen += x.size(0)
        acc = (correct / seen).tolist()
        print(
            f"  clf epoch {ep}/{epochs}  loss {total/seen:.3f}  "
            f"train acc {[round(a, 3) for a in acc]}"
        )
    clf.eval()
    return clf
