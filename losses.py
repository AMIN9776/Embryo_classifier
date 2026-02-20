"""
losses.py
---------
All loss functions for the embryo development classification pipeline.

Losses:
  - InfoNCELoss           : Cross-focal contrastive (NT-Xent / InfoNCE)
  - CrossFocalInfoNCELoss : Treats different focal planes as positives
  - MAEReconstructionLoss : Masked patch reconstruction (MSE, optional norm)
  - TemporalGapLoss       : Cross-entropy over time-gap bins
  - TemporalOrderLoss     : Binary CE for temporal order prediction
  - SupervisedCELoss      : Cross-entropy for stage classification
  - MonotonicityLoss      : Penalise non-monotonic stage predictions over time
  - MoEDiversityLoss      : Encourage expert specialisation
  - CombinedPretrainLoss  : Weighted sum of pretraining losses
  - CombinedFinetuneLoss  : Weighted sum of fine-tuning losses
"""

from typing import Optional, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# InfoNCE / NT-Xent
# =============================================================================

class InfoNCELoss(nn.Module):
    """
    Standard InfoNCE (NT-Xent) loss for contrastive learning.

    Treats (z1[i], z2[i]) as a positive pair and all other samples as negatives.
    Both z1 and z2 should already be L2-normalised.
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """
        z1, z2: [B, D] L2-normalised projections.
        Returns: scalar loss.
        """
        B = z1.shape[0]
        z = torch.cat([z1, z2], dim=0)  # [2B, D]

        # Cosine similarity matrix
        sim = torch.mm(z, z.t()) / self.temperature  # [2B, 2B]

        # Mask self-similarities on diagonal
        mask = torch.eye(2 * B, dtype=torch.bool, device=z.device)
        sim.masked_fill_(mask, float("-inf"))

        # Positive pairs are (i, i+B) and (i+B, i)
        labels = torch.cat([
            torch.arange(B, 2 * B), torch.arange(0, B)
        ]).to(z.device)

        loss = F.cross_entropy(sim, labels)
        return loss


class CrossFocalInfoNCELoss(nn.Module):
    """
    Cross-focal contrastive loss.

    For each (embryo e, timepoint t), focal plane images of the SAME embryo/time
    are positives; different embryo/time combinations are negatives.

    Input:
        projs: [B * N_focal, D] projections
               arranged as [f0_b0, f0_b1, ..., fN_b0, fN_b1, ...]
               i.e., all focal planes for batch item 0 come after each other
               → shape is actually [B, N_focal, D] reshaped to [B*N, D]
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(
        self, projs: torch.Tensor, B: int, N: int
    ) -> torch.Tensor:
        """
        projs: [B*N, D] — B samples, N focal planes each, L2-normalised.
        Returns: scalar loss.
        """
        projs = projs.view(B, N, -1)  # [B, N, D]

        # Compute similarities across all B*N embeddings
        flat = projs.view(B * N, -1)  # [B*N, D]
        sim = torch.mm(flat, flat.t()) / self.temperature  # [B*N, B*N]

        # Build label mask: positives = same embryo (same batch item), different focal
        idx = torch.arange(B * N, device=projs.device)
        embryo_ids = idx // N  # which embryo each element belongs to
        focal_ids = idx % N    # which focal plane

        # Two elements are positive if: same embryo AND different focal plane
        pos_mask = (embryo_ids.unsqueeze(1) == embryo_ids.unsqueeze(0)) & \
                   (focal_ids.unsqueeze(1) != focal_ids.unsqueeze(0))

        # Self-similarity mask
        self_mask = torch.eye(B * N, dtype=torch.bool, device=projs.device)
        sim.masked_fill_(self_mask, float("-inf"))

        # For each anchor, compute log-softmax over all non-self pairs
        log_probs = F.log_softmax(sim, dim=1)  # [B*N, B*N]

        # Average loss over positive pairs
        pos_log_probs = log_probs[pos_mask]
        loss = -pos_log_probs.mean()
        return loss


# =============================================================================
# MAE Reconstruction Loss
# =============================================================================

class MAEReconstructionLoss(nn.Module):
    """
    Masked patch reconstruction loss (MSE).

    Can optionally normalise target patches per-patch (as in original MAE paper).
    Can compute loss on masked patches only or all patches.
    """

    def __init__(self, norm_pix_loss: bool = True, loss_on_masked_only: bool = True):
        super().__init__()
        self.norm_pix_loss = norm_pix_loss
        self.loss_on_masked_only = loss_on_masked_only

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        pred:   [B, N_patches, patch_dim]
        target: [B, N_patches, patch_dim]
        mask:   [B, N_patches] (1 = masked, 0 = visible)

        Returns: scalar loss.
        """
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1e-6).sqrt()

        loss = (pred - target) ** 2  # [B, N, patch_dim]
        loss = loss.mean(dim=-1)     # [B, N]

        if self.loss_on_masked_only:
            # Only compute loss on masked patches
            loss = (loss * mask).sum() / (mask.sum() + 1e-6)
        else:
            loss = loss.mean()

        return loss


# =============================================================================
# Temporal Losses
# =============================================================================

class TemporalGapLoss(nn.Module):
    """
    Cross-entropy loss for time-gap bin prediction.
    Used in temporal self-supervision pretraining.
    """

    def __init__(self, num_bins: int, label_smoothing: float = 0.0):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def forward(self, logits: torch.Tensor, gap_bins: torch.Tensor) -> torch.Tensor:
        """
        logits:   [B, num_bins]
        gap_bins: [B] int labels (0-indexed bin)
        """
        return self.ce(logits, gap_bins)


class TemporalOrderLoss(nn.Module):
    """
    Binary cross-entropy for temporal order prediction.
    Given two timepoints, predict which came first.
    """

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return F.binary_cross_entropy_with_logits(logits.squeeze(-1), labels.float())


# =============================================================================
# Supervised Classification Loss
# =============================================================================

class SupervisedCELoss(nn.Module):
    """
    Cross-entropy loss for stage classification.
    Supports label smoothing and class weighting.
    """

    def __init__(
        self,
        num_classes: int,
        label_smoothing: float = 0.1,
        class_weights: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=label_smoothing,
        )

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        logits: [B, num_classes] or [B, T, num_classes]
        labels: [B] or [B, T]
        """
        if logits.dim() == 3:
            B, T, C = logits.shape
            return self.ce(logits.view(B * T, C), labels.view(B * T))
        return self.ce(logits, labels)


# =============================================================================
# Monotonicity Loss
# =============================================================================

class MonotonicityLoss(nn.Module):
    """
    Soft penalty for non-monotonic stage predictions over time.

    Embryo development is monotonic: stage can only increase or stay the same.
    This loss penalises predicted stage decreases between consecutive frames.
    """

    def __init__(self, margin: float = 0.0):
        super().__init__()
        self.margin = margin

    def forward(self, logits_seq: torch.Tensor) -> torch.Tensor:
        """
        logits_seq: [B, T, num_classes]
        Returns: scalar penalty.
        """
        # Get predicted stage as weighted sum (soft argmax)
        num_classes = logits_seq.shape[-1]
        stage_vals = torch.arange(num_classes, dtype=logits_seq.dtype,
                                   device=logits_seq.device)
        probs = F.softmax(logits_seq, dim=-1)  # [B, T, C]
        pred_stages = (probs * stage_vals).sum(dim=-1)  # [B, T]

        # Penalise decreases: max(0, pred_t - pred_{t+1} + margin)
        diffs = pred_stages[:, :-1] - pred_stages[:, 1:]  # [B, T-1], positive = decrease
        penalty = F.relu(diffs + self.margin)
        return penalty.mean()


# =============================================================================
# MoE Diversity Loss
# =============================================================================

class MoEDiversityLoss(nn.Module):
    """
    Encourage router to use all experts (avoid collapse to one expert).

    Minimises the difference between actual expert usage and uniform usage.
    Also penalises high entropy within-sample router distributions to
    encourage specialisation.
    """

    def forward(self, router_weights: torch.Tensor) -> torch.Tensor:
        """
        router_weights: [B, num_experts] softmax probabilities.
        Returns: scalar loss.
        """
        # Mean usage per expert across the batch
        mean_usage = router_weights.mean(dim=0)  # [num_experts]
        num_experts = mean_usage.shape[0]

        # Encourage uniform usage across experts (load balancing)
        target_usage = torch.ones_like(mean_usage) / num_experts
        balance_loss = F.kl_div(
            mean_usage.log(), target_usage, reduction="sum"
        )

        return balance_loss


# =============================================================================
# Auxiliary Losses for Fine-tuning
# =============================================================================

class CellCountAuxLoss(nn.Module):
    """
    Auxiliary cross-entropy loss for cell count prediction (stages 1-4).
    Maps stage labels to cell count labels: stage 1→0, 2→1, 3→2, 4→3.
    """

    def __init__(self, early_stage_ids: list):
        super().__init__()
        self.early_ids = set(early_stage_ids)
        self.ce = nn.CrossEntropyLoss()

    def forward(
        self, cell_count_logits: torch.Tensor, stage_labels: torch.Tensor
    ) -> torch.Tensor:
        """
        cell_count_logits: [B, 4]
        stage_labels: [B] (0-indexed stages)
        """
        # Only compute on early-stage samples
        mask = torch.tensor(
            [l.item() in self.early_ids for l in stage_labels],
            dtype=torch.bool, device=stage_labels.device
        )
        if mask.sum() == 0:
            return torch.tensor(0.0, device=cell_count_logits.device,
                                requires_grad=True)

        # Map stage label to cell count label (just use position within early group)
        early_id_list = sorted(self.early_ids)
        count_labels = torch.tensor(
            [early_id_list.index(l.item()) if l.item() in self.early_ids else 0
             for l in stage_labels],
            dtype=torch.long, device=stage_labels.device
        )
        return self.ce(cell_count_logits[mask], count_labels[mask])


class BlastocystAuxLoss(nn.Module):
    """
    Auxiliary MSE loss for blastocyst expansion score (stages 9-16).
    Maps late stage index to a 0-4 expansion score.
    """

    def __init__(self, late_stage_ids: list):
        super().__init__()
        self.late_ids = set(late_stage_ids)
        self.mse = nn.MSELoss()
        self.late_id_list = sorted(late_stage_ids)

    def forward(
        self, blast_scores: torch.Tensor, stage_labels: torch.Tensor
    ) -> torch.Tensor:
        """
        blast_scores: [B] predicted expansion score
        stage_labels: [B] (0-indexed stages)
        """
        mask = torch.tensor(
            [l.item() in self.late_ids for l in stage_labels],
            dtype=torch.bool, device=stage_labels.device
        )
        if mask.sum() == 0:
            return torch.tensor(0.0, device=blast_scores.device, requires_grad=True)

        # Map late stage index to 0-4 score (linear mapping)
        n = len(self.late_id_list)
        target_scores = torch.tensor(
            [self.late_id_list.index(l.item()) * 4.0 / (n - 1)
             if l.item() in self.late_ids else 0.0
             for l in stage_labels],
            dtype=torch.float, device=stage_labels.device
        )
        return self.mse(blast_scores[mask], target_scores[mask])


# =============================================================================
# Combined Loss Wrappers
# =============================================================================

class CombinedPretrainLoss(nn.Module):
    """
    Combined pretraining loss with configurable weights.

    total = w_contrastive * L_contrastive
          + w_mae         * L_mae
          + w_temporal    * L_temporal
    """

    def __init__(self, cfg: dict):
        super().__init__()
        pt_cfg = cfg["pretrain"]
        weights = pt_cfg["loss_weights"]
        self.w_contrastive = weights.get("contrastive", 1.0)
        self.w_mae = weights.get("mae", 1.0)
        self.w_temporal = weights.get("temporal", 0.5)

        self.contrastive_loss = CrossFocalInfoNCELoss(
            temperature=pt_cfg["contrastive"]["temperature"]
        )
        self.global_contrastive = InfoNCELoss(
            temperature=pt_cfg["contrastive"]["temperature"]
        )

        mae_cfg = pt_cfg["mae"]
        self.mae_loss = MAEReconstructionLoss(
            norm_pix_loss=mae_cfg["norm_pix_loss"],
            loss_on_masked_only=mae_cfg["loss_on_masked_only"],
        )

        num_bins = len(pt_cfg["temporal_ssl"]["gap_bins"]) + 1
        self.temporal_loss = TemporalGapLoss(num_bins=num_bins)

    def forward(
        self,
        z1: Optional[torch.Tensor] = None,
        z2: Optional[torch.Tensor] = None,
        cross_focal_projs: Optional[torch.Tensor] = None,
        B: int = 0, N: int = 0,
        mae_pred: Optional[torch.Tensor] = None,
        mae_target: Optional[torch.Tensor] = None,
        mae_mask: Optional[torch.Tensor] = None,
        temporal_logits: Optional[torch.Tensor] = None,
        gap_bins: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Returns: (total_loss, loss_dict)
        """
        losses = {}
        total = torch.tensor(0.0, device=self._get_device(z1, mae_pred, temporal_logits))

        # Global contrastive (augmented views)
        if z1 is not None and z2 is not None and self.w_contrastive > 0:
            l = self.global_contrastive(z1, z2)
            losses["contrastive_global"] = l.item()
            total = total + self.w_contrastive * l

        # Cross-focal contrastive
        if cross_focal_projs is not None and self.w_contrastive > 0:
            l = self.contrastive_loss(cross_focal_projs, B, N)
            losses["contrastive_focal"] = l.item()
            total = total + self.w_contrastive * l

        # MAE reconstruction
        if mae_pred is not None and self.w_mae > 0:
            l = self.mae_loss(mae_pred, mae_target, mae_mask)
            losses["mae"] = l.item()
            total = total + self.w_mae * l

        # Temporal gap prediction
        if temporal_logits is not None and gap_bins is not None and self.w_temporal > 0:
            l = self.temporal_loss(temporal_logits, gap_bins)
            losses["temporal_gap"] = l.item()
            total = total + self.w_temporal * l

        losses["total"] = total.item()
        return total, losses

    @staticmethod
    def _get_device(*tensors) -> torch.device:
        for t in tensors:
            if t is not None:
                return t.device
        return torch.device("cpu")


class CombinedFinetuneLoss(nn.Module):
    """
    Combined fine-tuning loss.

    total = w_ce      * L_ce
          + w_cell    * L_cell_count
          + w_blast   * L_blastocyst
          + w_mono    * L_monotonicity
          + w_moe_div * L_moe_diversity
    """

    def __init__(self, cfg: dict, class_weights: Optional[torch.Tensor] = None):
        super().__init__()
        ft_cfg = cfg["finetune"]
        weights = ft_cfg["loss_weights"]
        self.w_ce = weights.get("classification", 1.0)
        self.w_cell = weights.get("cell_count_aux", 0.3)
        self.w_blast = weights.get("blastocyst_aux", 0.2)
        self.w_mono = weights.get("monotonicity", 0.1)
        self.w_moe = weights.get("moe_diversity", 0.05)

        self.ce_loss = SupervisedCELoss(
            num_classes=cfg["data"]["num_stages"],
            label_smoothing=ft_cfg.get("label_smoothing", 0.1),
            class_weights=class_weights,
        )
        stage_groups = cfg["data"]["stage_groups"]
        early_ids = [s - 1 for s in stage_groups["early"]]
        late_ids = [s - 1 for s in stage_groups["late"]]

        self.cell_count_loss = CellCountAuxLoss(early_ids)
        self.blast_loss = BlastocystAuxLoss(late_ids)
        self.monotonicity_loss = MonotonicityLoss()
        self.moe_diversity_loss = MoEDiversityLoss()

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        cell_count_logits: Optional[torch.Tensor] = None,
        blastocyst_scores: Optional[torch.Tensor] = None,
        router_weights: Optional[torch.Tensor] = None,
        is_sequence: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        logits: [B, C] or [B, T, C]
        labels: [B] or [B, T]
        """
        losses = {}
        device = logits.device
        total = torch.tensor(0.0, device=device)

        # Main CE loss
        l_ce = self.ce_loss(logits, labels)
        losses["ce"] = l_ce.item()
        total = total + self.w_ce * l_ce

        # Flatten for auxiliary losses
        if is_sequence:
            B, T, C = logits.shape
            flat_labels = labels.view(B * T)
        else:
            flat_labels = labels

        # Cell count auxiliary loss
        if cell_count_logits is not None and self.w_cell > 0:
            if is_sequence:
                flat_cc = cell_count_logits.view(-1, cell_count_logits.shape[-1])
            else:
                flat_cc = cell_count_logits
            l_cell = self.cell_count_loss(flat_cc, flat_labels)
            losses["cell_count"] = l_cell.item()
            total = total + self.w_cell * l_cell

        # Blastocyst auxiliary loss
        if blastocyst_scores is not None and self.w_blast > 0:
            flat_blast = blastocyst_scores.view(-1) if is_sequence else blastocyst_scores
            l_blast = self.blast_loss(flat_blast, flat_labels)
            losses["blastocyst"] = l_blast.item()
            total = total + self.w_blast * l_blast

        # Monotonicity loss (sequence only)
        if is_sequence and self.w_mono > 0:
            l_mono = self.monotonicity_loss(logits)
            losses["monotonicity"] = l_mono.item()
            total = total + self.w_mono * l_mono

        # MoE diversity loss
        if router_weights is not None and self.w_moe > 0:
            flat_rw = router_weights.view(-1, router_weights.shape[-1])
            l_moe = self.moe_diversity_loss(flat_rw)
            losses["moe_diversity"] = l_moe.item()
            total = total + self.w_moe * l_moe

        losses["total"] = total.item()
        return total, losses
