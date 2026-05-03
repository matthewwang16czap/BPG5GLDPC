import json
import subprocess
from PIL import Image
import os
import numpy as np
from glob import glob
import torch
from torchvision import transforms
from torch.utils.data import Dataset
from torchmetrics.image import (
    StructuralSimilarityIndexMeasure as SSIM,
    MultiScaleStructuralSimilarityIndexMeasure as MS_SSIM,
)
import lpips
from utils import *


class RandomResizedCropImageDataset(Dataset):
    def __init__(self, dirs, image_dims, max_samples=None):
        """
        image_dims: (C,H,W)
        train: True = Random crop, False = Center crop or resize
        """
        self.paths = []
        for d in dirs:
            self.paths += glob(os.path.join(d, "*.png"))
            self.paths += glob(os.path.join(d, "*.jpg"))
        self.paths.sort()
        if max_samples is not None:
            self.paths = self.paths[:max_samples]
        _, H, W = image_dims
        self.transform = transforms.Compose(
            [
                transforms.Resize(min(H, W)),  # preserves aspect ratio
                transforms.CenterCrop((H, W)),  # exact final size
                transforms.ToTensor(),
            ]
        )

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        img_tensor = self.transform(img)
        return img_tensor


def get_dataset(data_dirs, config):
    return RandomResizedCropImageDataset(
        dirs=data_dirs,
        image_dims=config.image_dims,
        max_samples=config.max_test_samples,
    )


def preprocess_dataset(data_dirs, config, temp_dir="./temp/images"):
    """
    Saves preprocessed images as PNG to temp_dir for bpgenc.
    Returns list of saved PNG paths.
    """
    os.makedirs(temp_dir, exist_ok=True)
    dataset = get_dataset(data_dirs=data_dirs, config=config)
    saved_paths = []
    for i, item in enumerate(dataset):
        img_np = (item.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
        out_path = os.path.join(temp_dir, f"{i:05d}.png")
        Image.fromarray(img_np).save(out_path)
        saved_paths.append(out_path)
    print(f"Saved {len(saved_paths)} preprocessed images to {temp_dir}")
    return saved_paths


def load_image_tensor(path, device):
    img = Image.open(path).convert("RGB")
    t = torch.tensor(np.array(img)).float() / 255.0
    return t.permute(2, 0, 1).unsqueeze(0).to(device)


def file_to_bitstream(path):
    with open(path, "rb") as f:
        file_bytes = f.read()
    bitstream = np.unpackbits(np.frombuffer(file_bytes, dtype=np.uint8))
    return bitstream


def bitstream_to_file(bitstream, path):
    file_bytes = np.packbits(bitstream)
    with open(path, "wb") as f:
        f.write(file_bytes)


def encode_bpg(image_paths, q_list, temp_dir="./temp/bpg"):
    """
    For each q, encodes all images with bpgenc.
    """
    for q in q_list:
        q_dir_bpg = os.path.join(temp_dir, f"q{q}", "bpg")
        os.makedirs(q_dir_bpg, exist_ok=True)
        success_count = 0
        for img_path in image_paths:
            file_name = os.path.splitext(os.path.basename(img_path))[0]
            bpg_path = os.path.join(q_dir_bpg, f"{file_name}.bpg")
            # Encode
            subprocess.run(
                ["bpgenc", "-q", str(q), img_path, "-o", bpg_path],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            if not os.path.exists(bpg_path):
                print(f"[Warning] bpgenc failed for {img_path} at q={q}")
                continue
            success_count += 1
        print(f"q={q}: encoded {success_count}/{len(image_paths)} images")


def decode_bpg(image_paths, q_list, temp_dir="./temp/bpg", file_name_postfix=""):
    """
    For each q, decodes with bpgdec.
    Returns dict: { q -> [ {orig_path, rec_path, bpp} ] }
    """
    results = {}
    for q in q_list:
        q_dir_bpg = os.path.join(temp_dir, f"q{q}", "bpg")
        q_dir_rec = os.path.join(temp_dir, f"q{q}", "rec")
        os.makedirs(q_dir_rec, exist_ok=True)
        q_results = []
        for img_path in image_paths:
            file_name = os.path.splitext(os.path.basename(img_path))[0]
            bpg_path = os.path.join(q_dir_bpg, f"{file_name}{file_name_postfix}.bpg")
            rec_path = os.path.join(q_dir_rec, f"{file_name}{file_name_postfix}.png")
            orig = Image.open(img_path)
            H, W = orig.size[1], orig.size[0]
            bpp = os.path.getsize(bpg_path) * 8 / (H * W)
            # Decode
            subprocess.run(
                ["bpgdec", bpg_path, "-o", rec_path],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            if not os.path.exists(rec_path):
                print(f"[Warning] bpgdec failed for {bpg_path}")
                continue
            q_results.append({"orig_path": img_path, "rec_path": rec_path, "bpp": bpp})
        results[q] = q_results
        with open(
            os.path.join(temp_dir, f"results{file_name_postfix}.json"), "w"
        ) as fp:
            json.dump(results, fp, indent=2)
        print(f"q={q}: decoded {len(q_results)}/{len(image_paths)} images")
    return results


def compute_metrics(bpg_results, device="cuda", log_dir="./logs", file_name_postfix=""):
    """
    For each q, computes mean CBR, PSNR, SSIM, MS-SSIM, LPIPS.
    Saves results to JSON.
    """
    os.makedirs(log_dir, exist_ok=True)
    ssim_fn = SSIM(data_range=1.0).to(device)
    msssim_fn = MS_SSIM(data_range=1.0).to(device)
    lpips_fn = lpips.LPIPS(net="alex").to(device)
    all_results = []
    for q, items in bpg_results.items():
        psnr_list, ssim_list, msssim_list, lpips_list, bpp_list = [], [], [], [], []
        for item in items:
            orig_t = load_image_tensor(
                item["orig_path"], device
            )  # (1,C,H,W) float [0,1]
            rec_t = load_image_tensor(item["rec_path"], device)
            if rec_t.shape != orig_t.shape:
                print(
                    f"[Warning] shape mismatch for {item['rec_path']}, skipping metrics"
                )
                continue
            bpp = item["bpp"]
            psnr = compute_psnr(orig_t, rec_t).item()
            ssim = ssim_fn(rec_t, orig_t).item()
            msssim = msssim_fn(rec_t, orig_t).item()
            lp = (
                lpips_fn(rec_t * 2 - 1, orig_t * 2 - 1).mean().item()
            )  # lpips expects [-1,1]
            bpp_list.append(bpp)
            psnr_list.append(psnr)
            ssim_list.append(ssim)
            msssim_list.append(msssim)
            lpips_list.append(lp)
        all_results.append(
            {
                "q": q,
                "bpp": float(np.mean(bpp_list)),
                "psnr": float(np.mean(psnr_list)),
                "ssim": float(np.mean(ssim_list)),
                "msssim": float(np.mean(msssim_list)),
                "lpips": float(np.mean(lpips_list)),
            }
        )
        print(
            f"q={q} | BPP={all_results[-1]['bpp']:.4f} | PSNR={all_results[-1]['psnr']:.2f}"
        )
    out_path = os.path.join(log_dir, f"bpg_metrics{file_name_postfix}.json")
    with open(out_path, "w") as fp:
        json.dump(all_results, fp, indent=2)
    print(f"bpg metrics saved to {out_path}")
    return all_results
