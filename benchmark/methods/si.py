"""SI — Synaptic Intelligence (Zenke et al., ICML 2017).

Online version of EWC: accumulates path-integral of gradient*delta_weight
during training to estimate parameter importance.
No additional forward passes needed after task completion.
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .base import CILMethod, register_method
from .ncm import SimpleEncoder
from benchmark.protocols.cil import Task


@register_method("si")
class SI(CILMethod):
    name = "SI"

    def __init__(self, hsi_channels, lidar_channels, num_classes, device,
                 d=128, epochs=50, lr=1e-3, si_lambda=1.0, epsilon=0.1, **kwargs):
        encoder = SimpleEncoder(hsi_channels, lidar_channels, d)
        super().__init__(encoder, device, num_classes)
        self.d = d
        self.epochs = epochs
        self.lr = lr
        self.si_lambda = si_lambda
        self.epsilon = epsilon
        self.head = nn.Linear(d, num_classes).to(device)

        all_params = dict(list(self.model.named_parameters()) +
                          list(self.head.named_parameters()))
        # Online accumulators (reset each task)
        self._W:        dict[str, torch.Tensor] = {n: torch.zeros_like(p)
                                                    for n, p in all_params.items()}
        # Cumulative importance
        self._omega:    dict[str, torch.Tensor] = {n: torch.zeros_like(p)
                                                    for n, p in all_params.items()}
        # Optimal params at end of each task
        self._theta_star: dict[str, torch.Tensor] = {n: p.detach().clone().cpu()
                                                       for n, p in all_params.items()}

    def _all_named_params(self):
        return list(self.model.named_parameters()) + list(self.head.named_parameters())

    def train_task(self, task: Task, train_loader: DataLoader):
        self.model.train(); self.head.train()
        opt = torch.optim.SGD(
            [p for _, p in self._all_named_params()],
            lr=self.lr, momentum=0.9, weight_decay=0)
        cids = sorted(self.seen_classes)
        g2l = {g: i for i, g in enumerate(cids)}

        # Save params at start of task
        theta_prev = {n: p.detach().clone() for n, p in self._all_named_params()}
        # Reset W
        self._W = {n: torch.zeros_like(p) for n, p in self._all_named_params()}

        for ep in range(self.epochs):
            total, n = 0.0, 0
            for xh, xl, y in train_loader:
                xh, xl, y = xh.to(self.device), xl.to(self.device), y.to(self.device)

                # Save old params for W update
                p_old = {n: p.detach().clone() for n, p in self._all_named_params()}

                logits = self.head(self.model(xh, xl))[:, cids]
                mapped = torch.tensor([g2l[yi.item()] for yi in y], device=self.device)
                loss_ce = F.cross_entropy(logits, mapped)

                # SI penalty
                loss_si = torch.tensor(0.0, device=self.device)
                for n, p in self._all_named_params():
                    if n in self._omega:
                        theta_s = self._theta_star[n].to(self.device)
                        loss_si += (self._omega[n].to(self.device) *
                                    (p - theta_s).pow(2)).sum()

                loss = loss_ce + self.si_lambda * loss_si
                opt.zero_grad(); loss.backward()

                # Accumulate W: ∑ g * Δθ
                for n, p in self._all_named_params():
                    if p.grad is not None:
                        delta = p.detach() - p_old[n]
                        self._W[n] -= p.grad.detach() * delta

                opt.step()
                total += loss.item(); n += 1

            if (ep + 1) % 10 == 0:
                print(f"    [SI] Epoch {ep+1}/{self.epochs}  loss={total/n:.4f}")

    def after_task(self, task: Task, train_loader: DataLoader):
        """Update omega and theta_star."""
        for n, p in self._all_named_params():
            delta_sq = (p.detach() - self._theta_star[n].to(self.device)).pow(2)
            self._omega[n] = (self._omega[n].to(self.device) +
                               self._W[n].to(self.device) /
                               (delta_sq + self.epsilon)).clamp(min=0).cpu()
            self._theta_star[n] = p.detach().clone().cpu()

    @torch.no_grad()
    def predict(self, loader):
        self.model.eval(); self.head.eval()
        cids = sorted(self.seen_classes)
        all_p, all_t = [], []
        for xh, xl, y in loader:
            xh, xl = xh.to(self.device), xl.to(self.device)
            logits = self.head(self.model(xh, xl))[:, cids]
            idx = logits.argmax(1)
            all_p.append(torch.tensor([cids[i] for i in idx.cpu().tolist()]))
            all_t.append(y)
        return torch.cat(all_p).numpy(), torch.cat(all_t).numpy()

    # ── checkpoint ──────────────────────────────────────────────
    def _method_state(self) -> dict:
        return {
            "head": self.head.state_dict(),
            "_omega": {n: t.cpu() for n, t in self._omega.items()},
            "_theta_star": {n: t.cpu() for n, t in self._theta_star.items()},
        }

    def _load_method_state(self, ckpt: dict):
        self.head.load_state_dict(ckpt["head"])
        self._omega = ckpt["_omega"]
        self._theta_star = ckpt["_theta_star"]
