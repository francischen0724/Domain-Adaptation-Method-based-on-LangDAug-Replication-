import os
import torch
import matplotlib.pyplot as plt
from model.stylegan1 import EBM_CAttn
from vqvae import VQVAE
from torchvision import transforms
from PIL import Image
import numpy as np
import torch.optim as optim
import kornia.color as color
import math

# ======== 从 adapt 文件复制的函数 ========
def normalize_to_unit_range(tensor):
    return (tensor + 1) / 2

def normalize_to_neg_one_one(tensor):
    return tensor * 2 - 1

def normalize_lab_tensor(tensor):
    assert tensor.size(1) == 3
    tensor[:, 0, :, :] = (tensor[:, 0, :, :] / 50.0) - 1.0
    tensor[:, 1, :, :] = (tensor[:, 1, :, :] + 128) / 127.5 - 1.0
    tensor[:, 2, :, :] = (tensor[:, 2, :, :] + 128) / 127.5 - 1.0
    return tensor

def denormalize_lab_tensor(tensor):
    assert tensor.size(1) == 3
    tensor[:, 0, :, :] = (tensor[:, 0, :, :] + 1.0) * 50.0
    tensor[:, 1, :, :] = (tensor[:, 1, :, :] + 1.0) * 127.5 - 128
    tensor[:, 2, :, :] = (tensor[:, 2, :, :] + 1.0) * 127.5 - 128
    return tensor

def rgb_to_lab(rgb_tensor):
    rgb_tensor_unit = normalize_to_unit_range(rgb_tensor)
    return color.rgb_to_lab(rgb_tensor_unit)

def lab_to_rgb(lab_tensor):
    rgb_tensor_unit = color.lab_to_rgb(lab_tensor)
    return normalize_to_neg_one_one(rgb_tensor_unit)

def langvin_sampler(model, x, y, langevin_steps=20, lr=1.0):
    x = x.clone().detach()
    y = y.clone().detach()
    x.requires_grad_(True)
    sgd = optim.SGD([x], lr=lr)
    for _ in range(langevin_steps):
        model.zero_grad()
        sgd.zero_grad()
        energy = model(x, y).sum()
        (-energy).backward()
        sgd.step()
    return x.clone().detach()

# ======== CONFIG ========
CHECKPOINT_DIR = "./results/fundus-123/00001-ls40-llr1.0000-lr0.0010-embed32-nembed256-attn-sn-chm8-beta0.5_0.999-123"
AE_CKPT = "./logs/fundus-123-embed32-nembed256-noisein0.03-LAB/checkpoint/vqvae_best.pt"
DATA_DIR = "./datasets/fundus/test/Domain1/image"

EMBED_DIM = 32
N_EMBED = 256
CHANNEL_MUL = 8
IMG_SIZE = 256
N_SAMPLES = 4
LANGEVIN_STEP = 40
LANGEVIN_LR = 1.0
DEVICE = "cuda"
CHECKPOINTS_PER_PAGE = 4  # 每页显示多少个 checkpoint

# ======== LOAD MODELS ========
def load_ae():
    ckpt = torch.load(AE_CKPT, map_location=DEVICE)
    model = VQVAE(embed_dim=EMBED_DIM, n_embed=N_EMBED, noise=False)
    model.load_state_dict({k.replace("module.", ""): v for k, v in ckpt.items()})
    model.to(DEVICE)
    model.eval()
    return model

def load_ebm(path):
    model = EBM_CAttn(
        size=64, channel_multiplier=CHANNEL_MUL,
        input_channel=EMBED_DIM * 2, add_attention=True,
        spectral=True, cam=False, dataset='fundus'
    ).to(DEVICE)
    ckpt = torch.load(path, map_location=DEVICE)
    model.load_state_dict(ckpt)
    model.eval()
    return model

# ======== LOAD TEST IMAGES ========
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
valid_ext = (".jpg", ".jpeg", ".png", ".bmp", ".tif")
img_paths = sorted([
    os.path.join(DATA_DIR, f)
    for f in os.listdir(DATA_DIR)
    if f.lower().endswith(valid_ext)
])[:N_SAMPLES]

print(f"✅ Found {len(img_paths)} images for comparison")
images = torch.stack([transform(Image.open(p).convert("RGB")) for p in img_paths]).to(DEVICE)

# ======== SCAN CHECKPOINTS ========
all_checkpoints = sorted([f for f in os.listdir(CHECKPOINT_DIR) if f.startswith("ebm_") and f.endswith(".pt")])
num_pages = math.ceil(len(all_checkpoints) / CHECKPOINTS_PER_PAGE)

# ======== INFERENCE & SAVE ========
ae = load_ae()
with torch.no_grad():
    LAB_img = normalize_lab_tensor(rgb_to_lab(images))
    latent_img = ae.latent(LAB_img)
    label_latent = latent_img.clone()

for page in range(num_pages):
    ckpts = all_checkpoints[page*CHECKPOINTS_PER_PAGE:(page+1)*CHECKPOINTS_PER_PAGE]
    results = []
    for ckpt in ckpts:
        ebm_model = load_ebm(os.path.join(CHECKPOINT_DIR, ckpt))
        latent_q = langvin_sampler(ebm_model, latent_img.clone(), label_latent,
                                   langevin_steps=LANGEVIN_STEP, lr=LANGEVIN_LR)
        out = denormalize_lab_tensor(ae.dec(latent_q))
        out = lab_to_rgb(out)
        results.append(out)

    fig, axes = plt.subplots(N_SAMPLES, len(ckpts) + 1, figsize=(4*(len(ckpts)+1), 3*N_SAMPLES))
    for row in range(N_SAMPLES):
        img_input = (images[row].cpu().numpy().transpose(1, 2, 0) + 1) / 2
        axes[row, 0].imshow(np.clip(img_input, 0, 1))
        axes[row, 0].axis("off")
        if row == 0:
            axes[row, 0].set_title("Input", fontsize=10)

        for col, out in enumerate(results):
            img = (out[row].detach().cpu().numpy().transpose(1, 2, 0) + 1) / 2
            axes[row, col + 1].imshow(np.clip(img, 0, 1))
            axes[row, col + 1].axis("off")
            if row == 0:
                axes[row, col + 1].set_title(ckpts[col].replace("ebm_", "").replace(".pt", ""), fontsize=10)

    plt.tight_layout()
    save_name = f"ebm_comparison_page_{page+1}.png"
    plt.savefig(save_name, dpi=300)
    plt.close()
    print(f"✅ Saved page {page+1}/{num_pages}: {save_name}")
