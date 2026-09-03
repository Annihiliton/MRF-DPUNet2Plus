import logging
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader

import config as cfg
from dataset import EvalTransform, SegmentationDataset, TrainTransform
from model import MRFDPUnet2Plus


class BalancedBinaryLoss(nn.Module):
    def __init__(self, weights=(0.5, 0.5)):
        super().__init__()
        self.weights = weights

    def forward(self, prediction, target):
        prediction = prediction.reshape(-1)
        target = target.reshape(-1)
        if prediction.shape != target.shape:
            raise ValueError(f"Prediction/label shape mismatch: {prediction.shape} vs {target.shape}")
        values = F.binary_cross_entropy_with_logits(prediction, target, reduction="none")
        positive = (target > 0.5).float()
        negative = (target < 0.5).float()
        positive_count = positive.sum().item() + 1e-12
        negative_count = negative.sum().item() + 1e-12
        return (
            self.weights[0] * positive * values / positive_count
            + self.weights[1] * negative * values / negative_count
        ).sum()


class SoftDiceLoss(nn.Module):
    def __init__(self, weights=(0.5, 0.5)):
        super().__init__()
        self.weights = weights

    def forward(self, prediction, target, smooth=1e-5):
        count = prediction.shape[0]
        estimate = prediction.reshape(count, -1)
        truth = target.reshape(count, -1)
        if estimate.shape != truth.shape:
            raise ValueError(f"Prediction/label shape mismatch: {estimate.shape} vs {truth.shape}")
        scale = target.detach().reshape(count, -1)
        scale = scale * (self.weights[1] - self.weights[0]) + self.weights[0]
        estimate = scale * estimate
        truth = scale * truth
        overlap = (estimate * truth).sum(-1)
        magnitude = (estimate * estimate).sum(-1) + (truth * truth).sum(-1)
        return (1.0 - (2.0 * overlap + smooth) / (magnitude + smooth)).mean()


class HybridSegLoss(nn.Module):
    def __init__(self, dice_ratio=0.5, bce_ratio=0.5):
        super().__init__()
        self.binary_term = BalancedBinaryLoss()
        self.dice_term = SoftDiceLoss()
        self.dice_ratio = dice_ratio
        self.bce_ratio = bce_ratio

    def forward(self, prediction, target):
        return self.dice_ratio * self.dice_term(prediction, target) + self.bce_ratio * self.binary_term(prediction, target)


def dice_value(prediction, target):
    count = prediction.shape[0]
    estimate = (prediction >= 0.5).float().reshape(count, -1)
    truth = (target > 0).float().reshape(count, -1)
    overlap = (estimate * truth).sum(dim=1)
    magnitude = (estimate * estimate).sum(dim=1) + (truth * truth).sum(dim=1)
    return ((2.0 * overlap + 1e-5) / (magnitude + 1e-5)).mean().item()


def iou_value(prediction, target):
    count = prediction.shape[0]
    estimate = (torch.sigmoid(prediction) >= 0.5).reshape(count, -1)
    truth = (target > 0).reshape(count, -1)
    overlap = (estimate & truth).sum(dim=1).float()
    joined = (estimate | truth).sum(dim=1).float()
    score = torch.where(joined > 0, overlap / joined, torch.ones_like(joined))
    return score.mean().item()


def open_log(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    stream = logging.getLogger(cfg.model_name)
    stream.setLevel(logging.INFO)
    stream.handlers.clear()
    layout = logging.Formatter("%(message)s")
    disk = logging.FileHandler(path, encoding="utf-8", mode="w")
    disk.setFormatter(layout)
    stream.addHandler(disk)
    terminal = logging.StreamHandler()
    terminal.setFormatter(layout)
    stream.addHandler(terminal)
    return stream


def seed_everything(value):
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(value)
        torch.cuda.manual_seed_all(value)
    cudnn.benchmark = False
    cudnn.deterministic = True


def seed_worker(worker_id):
    random.seed(cfg.seed + worker_id)
    np.random.seed(cfg.seed + worker_id)


def new_network():
    return MRFDPUnet2Plus(
        num_classes=cfg.n_labels,
        input_channels=cfg.n_channels,
    )


def pass_over(loader, network, objective, update_rule, amp_scaler, device, training, epoch, log):
    network.train(training)
    loss_total = 0.0
    dice_total = 0.0
    iou_total = 0.0
    sample_total = 0
    start_time = time.time()

    for step, (batch, _) in enumerate(loader, 1):
        inputs = batch["image"].to(device, non_blocking=True)
        targets = batch["label"].to(device, non_blocking=True).float()
        current_size = inputs.shape[0]
        if training:
            update_rule.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
                outputs = network(inputs)
                loss = objective(outputs, targets)
            if training:
                if amp_scaler is not None:
                    amp_scaler.scale(loss).backward()
                    amp_scaler.unscale_(update_rule)
                    norm = torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=1.0)
                    if norm > 10.0:
                        log.warning(f"WARNING: grad norm = {float(norm):.2f}")
                    amp_scaler.step(update_rule)
                    amp_scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=1.0)
                    update_rule.step()
        dice = dice_value(outputs.detach(), targets)
        iou = iou_value(outputs.detach(), targets)
        loss_total += loss.item() * current_size
        dice_total += dice * current_size
        iou_total += iou * current_size
        sample_total += current_size
        phase = "Train" if training else "Val"
        log.info(
            f"[{phase}] Epoch {epoch:03d} | Batch {step:04d}/{len(loader):04d} | "
            f"Loss {loss.item():.4f} | Dice {dice:.4f} | IoU {iou:.4f}"
        )

    mean_loss = loss_total / sample_total
    mean_dice = dice_total / sample_total
    mean_iou = iou_total / sample_total
    duration = time.time() - start_time
    phase = "Train" if training else "Val"
    log.info(
        f"[{phase}] Epoch {epoch:03d} summary | Loss {mean_loss:.4f} | "
        f"Dice {mean_dice:.4f} | IoU {mean_iou:.4f} | Time {duration:.1f}s"
    )
    return mean_loss, mean_dice


def write_best(network, update_rule, epoch, valid_loss, valid_dice):
    cfg.model_path.mkdir(parents=True, exist_ok=True)
    package = {
        "epoch": epoch,
        "best_model": True,
        "model": cfg.model_name,
        "state_dict": network.state_dict(),
        "val_loss": valid_loss,
        "val_dice": valid_dice,
        "optimizer": update_rule.state_dict(),
    }
    torch.save(package, cfg.best_model_path)


def main():
    seed_everything(cfg.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required by this training configuration.")

    device = torch.device("cuda")
    cfg.save_path.mkdir(parents=True, exist_ok=True)
    log = open_log(cfg.logger_path)

    training_set = SegmentationDataset(cfg.train_dataset, TrainTransform(), cfg.img_size)
    validation_set = SegmentationDataset(cfg.val_dataset, EvalTransform(), cfg.img_size)
    training_batches = DataLoader(
        training_set,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        worker_init_fn=seed_worker,
        pin_memory=False,
    )
    validation_batches = DataLoader(
        validation_set,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        worker_init_fn=seed_worker,
        pin_memory=False,
    )

    network = new_network().to(device)
    update_rule = torch.optim.Adam(filter(lambda p: p.requires_grad, network.parameters()), lr=cfg.learning_rate)
    objective = HybridSegLoss(dice_ratio=0.5, bce_ratio=0.5)
    amp_scaler = torch.amp.GradScaler("cuda")

    log.info(f"Model: {cfg.model_name}")
    log.info(f"GPU: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    log.info(f"Train images: {len(training_set)} | Val images: {len(validation_set)}")
    log.info(f"Epochs: {cfg.epochs} | Batch size: {cfg.batch_size} | LR: {cfg.learning_rate} | Seed: {cfg.seed}")

    top_dice = 0.0
    top_epoch = 0
    for epoch in range(1, cfg.epochs + 1):
        log.info(f"\n========= Epoch [{epoch}/{cfg.epochs}] =========")
        pass_over(training_batches, network, objective, update_rule, amp_scaler, device, True, epoch, log)
        with torch.no_grad():
            valid_loss, valid_dice = pass_over(
                validation_batches, network, objective, update_rule, amp_scaler, device, False, epoch, log
            )
        if valid_dice > top_dice:
            log.info(f"Saving best model: Dice improved from {top_dice:.4f} to {valid_dice:.4f}")
            top_dice = valid_dice
            top_epoch = epoch
            write_best(network, update_rule, epoch, valid_loss, valid_dice)
        else:
            log.info(f"Validation Dice did not improve. Best: {top_dice:.4f} at epoch {top_epoch}")
        inactive_epochs = epoch - top_epoch
        log.info(f"early_stopping_count: {inactive_epochs}/{cfg.early_stopping_patience}")
        if inactive_epochs > cfg.early_stopping_patience:
            log.info("Early stopping.")
            break

    log.info(f"Training completed. Best Dice: {top_dice:.4f} at epoch {top_epoch}")
    log.info(f"Best checkpoint: {cfg.best_model_path}")


if __name__ == "__main__":
    main()
