"""GPM — Gradient Projection Memory (Saha et al., NeurIPS 2021).

Key idea: after each task, compute the SVD of activations and store the top
principal components as a "memory space". Gradients are then projected into the
null-space of all previous tasks' memory spaces, preventing interference with
previously learned representations.

This is an exemplar-free method — no raw data is stored.
"""
from __future__ import annotations
from copy import deepcopy
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .base import CILMethod
from .ncm import SimpleEncoder
from benchmark.protocols.cil import Task


class GPM(CILMethod):
    name = "GPM"

    def __init__(self, hsi_channels, lidar_channels, num_classes, device,
                 d: int = 128,
                 epochs: int = 50,
                 lr: float = 1e-3,
                 threshold: float = 0.97,
                 eps: float = 0.1):
        """
        Args:
            d:          Feature embedding dimension.
            epochs:     Training epochs per task.
            lr:         Learning rate.
            threshold:  Fraction of variance to retain per layer (SVD cutoff).
            eps:        Slack for projection — allows small gradient leakage.
        """
        encoder = SimpleEncoder(hsi_channels, lidar_channels, d)
        super().__init__(encoder, device, num_classes)
        self.d = d
        self.epochs = epochs
        self.lr = lr
        self.threshold = threshold
        self.eps = eps

        self.head = nn.Linear(d, num_classes).to(device)

        # Basis matrices per parameter: name → (d_param, k) tensor
        self._memory: dict[str, torch.Tensor] = {}

    # ── Training ──────────────────────────────────────────────────

    def train_task(self, task: Task, train_loader: DataLoader):
        self.model.train()
        self.head.train()
        cids = sorted(self.seen_classes)
        g2l = {g: i for i, g in enumerate(cids)}

        params = list(self.model.parameters()) + list(self.head.parameters())
        opt = torch.optim.AdamW(params, lr=self.lr, weight_decay=1e-4)

        for ep in range(self.epochs):
            total, n = 0.0, 0
            for xh, xl, y in train_loader:
                xh, xl, y = xh.to(self.device), xl.to(self.device), y.to(self.device)
                feat   = self.model(xh, xl)
                logits = self.head(feat)[:, cids]
                mapped = torch.tensor([g2l[yi.item()] for yi in y], device=self.device)
                loss = F.cross_entropy(logits, mapped)

                opt.zero_grad()
                loss.backward()

                # Project gradients into null-space of memory
                self._project_gradients()

                opt.step()
                total += loss.item(); n += 1

            if (ep + 1) % 10 == 0:
                print(f"    [GPM] Epoch {ep+1}/{self.epochs}  loss={total/n:.4f}")

    def after_task(self, task: Task, train_loader: DataLoader):
        """Compute SVD of activations and update memory spaces."""
        self._update_memory(train_loader)

    # ── Gradient projection ───────────────────────────────────────

    def _project_gradients(self):
        """For each parameter with a stored basis, project out the memory directions."""
        for name, param in self.model.named_parameters():
            if param.grad is None or name not in self._memory:
                continue
            basis = self._memory[name].to(self.device)  # (flat_dim, k)
            g = param.grad.data.view(-1)                 # (flat_dim,)
            # Remove components in the memory space
            proj = basis @ (basis.t() @ g)               # (flat_dim,)
            param.grad.data -= proj.view_as(param.grad.data)

    # ── Memory update (SVD of activations) ───────────────────────

    @torch.no_grad()
    def _update_memory(self, loader: DataLoader):
        """
        Collect activation matrices per layer via forward hooks, then compute
        the top-k singular vectors and add them to the memory space.
        """
        activations: dict[str, list[torch.Tensor]] = {}
        hooks = []

        for name, module in self.model.named_modules():
            if not isinstance(module, (nn.Linear, nn.Conv2d)):
                continue
            # Capture input activations
            def make_hook(n):
                def hook(mod, inp, out):
                    x = inp[0].detach().cpu()
                    if x.ndim == 4:
                        # Conv: (B, C_in, H, W) → (B*H*W, C_in)
                        x = x.permute(0, 2, 3, 1).reshape(-1, x.size(1))
                    activations.setdefault(n, []).append(x)
                return hook
            hooks.append(module.register_forward_hook(make_hook(name)))

        self.model.eval()
        for xh, xl, _ in loader:
            self.model(xh.to(self.device), xl.to(self.device))

        for h in hooks:
            h.remove()

        for name, acts in activations.items():
            act_mat = torch.cat(acts, dim=0).float()  # (N, C_in)

            # SVD on activation matrix
            try:
                _, S, Vt = torch.linalg.svd(act_mat, full_matrices=False)
            except Exception:
                continue

            # Select top-k singular vectors that explain `threshold` variance
            var_explained = (S ** 2).cumsum(0) / (S ** 2).sum()
            k = int((var_explained < self.threshold).sum().item()) + 1
            k = max(1, min(k, S.size(0)))
            new_basis = Vt[:k].t()  # (C_in, k)

            # Find the corresponding param name (weight of this layer)
            param_name = name + ".weight"
            if param_name not in dict(self.model.named_parameters()):
                continue

            if param_name not in self._memory:
                self._memory[param_name] = new_basis
            else:
                # Merge old and new bases: QR to get orthonormal combined basis
                combined = torch.cat([self._memory[param_name], new_basis], dim=1)
                Q, _ = torch.linalg.qr(combined)
                self._memory[param_name] = Q

    # ── Inference ─────────────────────────────────────────────────

    @torch.no_grad()
    def predict(self, loader: DataLoader):
        self.model.eval()
        self.head.eval()
        cids = sorted(self.seen_classes)
        all_p, all_t = [], []
        for xh, xl, y in loader:
            xh, xl = xh.to(self.device), xl.to(self.device)
            logits = self.head(self.model(xh, xl))[:, cids]
            idx = logits.argmax(1)
            all_p.append(torch.tensor([cids[i] for i in idx.cpu().tolist()]))
            all_t.append(y)
        return torch.cat(all_p).numpy(), torch.cat(all_t).numpy()
