import argparse
import gc
import os
import tempfile
import time
from datetime import datetime

import cv2
import numpy as np
import torch
from scipy.ndimage import distance_transform_edt
from torch.utils.data import DataLoader
from tqdm import tqdm

import config as cfg
from dataset import EvalTransform, SegmentationDataset
from model import MRFDPUnet2Plus


profile_warmup = 50
profile_runs = 500
mask_threshold = 0.5


def new_network():
    return MRFDPUnet2Plus(
        num_classes=cfg.n_labels,
        input_channels=cfg.n_channels,
    )


def restore_weights(network, path):
    path = os.fspath(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    package = torch.load(path, map_location="cpu")
    weights = package.get("state_dict", package) if isinstance(package, dict) else package
    try:
        network.load_state_dict(weights, strict=True)
    except RuntimeError:
        weights = {(key[7:] if key.startswith("module.") else key): value for key, value in weights.items()}
        network.load_state_dict(weights, strict=True)
    return network


def overlap_scores(estimate, truth):
    estimate = estimate.astype(np.uint8)
    truth = truth.astype(np.uint8)
    if truth.sum() == 0 and estimate.sum() == 0:
        return 1.0, 1.0, 1.0, 1.0
    true_positive = np.sum((estimate == 1) & (truth == 1))
    false_positive = np.sum((estimate == 1) & (truth == 0))
    false_negative = np.sum((estimate == 0) & (truth == 1))
    eps = 1e-7
    dice = 2.0 * true_positive / (2.0 * true_positive + false_positive + false_negative + eps)
    iou = true_positive / (true_positive + false_positive + false_negative + eps)
    recall = true_positive / (true_positive + false_negative + eps)
    precision = true_positive / (true_positive + false_positive + eps)
    return float(dice), float(iou), float(recall), float(precision)


def percentile_hausdorff(estimate, truth):
    estimate = np.asarray(estimate).astype(bool)
    truth = np.asarray(truth).astype(bool)
    if estimate.sum() == 0 or truth.sum() == 0:
        return 0.0
    distance_to_estimate = distance_transform_edt(~estimate)
    distance_to_truth = distance_transform_edt(~truth)
    a = np.percentile(distance_to_estimate[truth], 95)
    b = np.percentile(distance_to_truth[estimate], 95)
    return float(max(a, b))


def infer(network, device, weight_path, save_masks=True):
    test_set = SegmentationDataset(cfg.test_dataset, EvalTransform(), cfg.img_size)
    batches = DataLoader(test_set, batch_size=1, shuffle=False)
    if save_masks:
        cfg.visualize_path.mkdir(parents=True, exist_ok=True)
    cfg.save_path.mkdir(parents=True, exist_ok=True)
    totals = {"dice": 0.0, "iou": 0.0, "recall": 0.0, "precision": 0.0, "hd95": 0.0}

    network.eval()
    with torch.inference_mode():
        for batch, names in tqdm(batches, desc="Evaluate", unit="img"):
            frame = batch["image"].to(device)
            truth = batch["label"].numpy()[0]
            raw = network(frame)[0, 0].detach().cpu().numpy()
            estimate = (raw >= mask_threshold).astype(np.uint8)
            dice, iou, recall, precision = overlap_scores(estimate, truth)
            hd95 = percentile_hausdorff(estimate, truth)
            totals["dice"] += dice
            totals["iou"] += iou
            totals["recall"] += recall
            totals["precision"] += precision
            totals["hd95"] += hd95
            if save_masks:
                filename = f"{os.path.splitext(names[0])[0]}.png"
                cv2.imwrite(os.fspath(cfg.visualize_path / filename), estimate * 255)

    count = len(test_set)
    summary = {name: value / count for name, value in totals.items()}
    print("Segmentation test results")
    print(f"Checkpoint : {weight_path}")
    print(f"Test images: {count}")
    print(f"Dice       : {summary['dice']:.6f}")
    print(f"IoU        : {summary['iou']:.6f}")
    print(f"Recall     : {summary['recall']:.6f}")
    print(f"Precision  : {summary['precision']:.6f}")
    print(f"HD95       : {summary['hd95']:.6f}")
    with open(cfg.result_path, "a", encoding="utf-8") as handle:
        handle.write(str(datetime.now()) + "\n")
        handle.write(f"model: {weight_path}\n")
        for name in ("dice", "iou", "recall", "precision", "hd95"):
            handle.write(f"{name}: {summary[name]}\n")
        handle.write("\n")
    return summary


def parameter_count(network):
    return sum(item.numel() for item in network.parameters())


def operation_count(network, sample):
    try:
        from thop import profile
    except ImportError:
        print("[Warning] thop is not installed; MACs/FLOPs will be skipped.")
        return None, None
    with torch.inference_mode():
        macs, _ = profile(network, inputs=(sample,), verbose=False)
    return float(macs), float(2.0 * macs)


def serialized_size(network):
    filename = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as handle:
            filename = handle.name
        torch.save(network.state_dict(), filename)
        return os.path.getsize(filename) / (1024 ** 2)
    finally:
        if filename and os.path.exists(filename):
            os.remove(filename)


def peak_memory(network, sample, device):
    with torch.inference_mode():
        for _ in range(10):
            network(sample)
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    with torch.inference_mode():
        network(sample)
    torch.cuda.synchronize(device)
    return torch.cuda.max_memory_allocated(device) / (1024 ** 2)


def timing(network, sample, device, warmup=50, runs=500):
    torch.backends.cudnn.benchmark = True
    with torch.inference_mode():
        for _ in range(warmup):
            network(sample)
    torch.cuda.synchronize(device)
    durations = []
    with torch.inference_mode():
        for _ in range(runs):
            torch.cuda.synchronize(device)
            begin = time.perf_counter()
            network(sample)
            torch.cuda.synchronize(device)
            durations.append((time.perf_counter() - begin) * 1000.0)
    durations = np.asarray(durations, dtype=np.float64)
    average = float(np.mean(durations))
    deviation = float(np.std(durations, ddof=1))
    return average, deviation, 1000.0 / average


def profile_network(network, device, weight_path):
    shape = (1, cfg.n_channels, cfg.img_size, cfg.img_size)
    sample = torch.randn(*shape, dtype=torch.float32, device=device)
    params = parameter_count(network)
    macs, flops = operation_count(network, sample)
    size_mb = serialized_size(network)
    memory_mb = peak_memory(network, sample, device)
    latency, latency_std, fps = timing(network, sample, device, profile_warmup, profile_runs)
    print("Model efficiency")
    print(f"Model           : {cfg.model_name}")
    print(f"Checkpoint      : {weight_path}")
    print(f"GPU             : {torch.cuda.get_device_name(device)}")
    print(f"Input shape     : {shape}")
    print("Batch size      : 1")
    print("Precision       : FP32")
    print(f"Params          : {params / 1e6:.4f} M")
    if macs is None:
        print("MACs            : N/A")
        print("FLOPs           : N/A")
    else:
        print(f"MACs            : {macs / 1e9:.4f} G")
        print(f"FLOPs (2xMACs)  : {flops / 1e9:.4f} G")
    print(f"Model size      : {size_mb:.2f} MB")
    print(f"Peak GPU memory : {memory_mb:.2f} MB")
    print(f"Latency         : {latency:.3f} +/- {latency_std:.3f} ms/image")
    print(f"FPS             : {fps:.2f} images/s")
    del sample
    gc.collect()
    torch.cuda.empty_cache()


def read_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("evaluate", "efficiency", "all"), default="evaluate")
    parser.add_argument("--checkpoint", default=os.fspath(cfg.best_model_path))
    parser.add_argument("--no-save-predictions", action="store_true")
    return parser.parse_args()


def main():
    args = read_cli()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for this test script.")
    device = torch.device("cuda")
    network = restore_weights(new_network(), args.checkpoint).to(device).eval()
    if args.mode in ("evaluate", "all"):
        infer(network, device, args.checkpoint, save_masks=not args.no_save_predictions)
    if args.mode in ("efficiency", "all"):
        profile_network(network, device, args.checkpoint)


if __name__ == "__main__":
    main()
