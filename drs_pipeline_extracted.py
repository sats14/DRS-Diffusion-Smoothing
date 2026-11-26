# Extracted from certified_smoothing_kaggle_v2.ipynb (code cells in order)

# ===== cell 2 =====
!pip install -q diffusers accelerate statsmodels

import torch, torchvision
import numpy as np
import pandas as pd
import os, json, random
from statsmodels.stats.proportion import proportion_confint
from scipy.stats import norm
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)
assert device == "cuda", "Set Accelerator to GPU T4 x2 in the settings panel, then restart the session."

WORK_DIR = "/kaggle/working"
CHECKPOINT_PATH = os.path.join(WORK_DIR, "certify_checkpoint.csv")
FT_WEIGHTS_PATH = os.path.join(WORK_DIR, "resnet20_noise_ft.pt")


# ===== cell 4 =====
transform = torchvision.transforms.ToTensor()
testset = torchvision.datasets.CIFAR10(root=os.path.join(WORK_DIR, "data"), train=False,
                                        download=True, transform=transform)
classes = testset.classes
print(len(testset), "test images. Classes:", classes)


# ===== cell 6 =====
base_model_raw = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet20", pretrained=True)

CIFAR10_MEAN = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
CIFAR10_STD  = torch.tensor([0.2470, 0.2435, 0.2616]).view(1, 3, 1, 1)

class NormalizedModel(torch.nn.Module):
    def __init__(self, model, mean, std):
        super().__init__()
        self.model = model
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)
    def forward(self, x):
        return self.model((x - self.mean) / self.std)

base_model = NormalizedModel(base_model_raw, CIFAR10_MEAN, CIFAR10_STD).to(device).eval()

# Sanity check: clean accuracy should be close to the model's published 92.6%.
loader = torch.utils.data.DataLoader(testset, batch_size=200, shuffle=False)
x_check, y_check = next(iter(loader))
x_check, y_check = x_check.to(device), y_check.to(device)
with torch.no_grad():
    pred_check = base_model(x_check).argmax(1)
print("clean top-1 accuracy on 200 test images:", (pred_check == y_check).float().mean().item())


# ===== cell 8 =====
import torch.optim as optim

FT_SIGMA = 0.25       # must match the sigma you certify at in Section 9
FT_EPOCHS = 5
FT_LR = 1e-3
FT_BATCH_SIZE = 128

trainset = torchvision.datasets.CIFAR10(root=os.path.join(WORK_DIR, "data"), train=True,
                                         download=True, transform=torchvision.transforms.ToTensor())
train_loader = torch.utils.data.DataLoader(trainset, batch_size=FT_BATCH_SIZE, shuffle=True, num_workers=2)

raw_model = base_model.model
raw_model.train()
optimizer = optim.SGD(raw_model.parameters(), lr=FT_LR, momentum=0.9, weight_decay=5e-4)
criterion = torch.nn.CrossEntropyLoss()

for epoch in range(FT_EPOCHS):
    total, correct, running_loss = 0, 0, 0.0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        noise = torch.randn_like(x) * FT_SIGMA
        x_noisy = (x + noise).clamp(0, 1)
        x_norm = (x_noisy - CIFAR10_MEAN.to(device)) / CIFAR10_STD.to(device)

        optimizer.zero_grad()
        logits = raw_model(x_norm)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * x.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += x.size(0)
    print(f"epoch {epoch+1}/{FT_EPOCHS}  loss={running_loss/total:.4f}  noisy-train-acc={correct/total:.4f}")

raw_model.eval()
base_model = NormalizedModel(raw_model, CIFAR10_MEAN, CIFAR10_STD).to(device).eval()

torch.save(raw_model.state_dict(), FT_WEIGHTS_PATH)
print("saved fine-tuned weights to", FT_WEIGHTS_PATH)

# Verify under the SAME kind of noise it will be certified against -- not clean accuracy.
verify_loader = torch.utils.data.DataLoader(testset, batch_size=200, shuffle=False)
x_v, y_v = next(iter(verify_loader))
x_v, y_v = x_v.to(device), y_v.to(device)
noise = torch.randn_like(x_v) * FT_SIGMA
x_v_noisy = (x_v + noise).clamp(0, 1)
with torch.no_grad():
    pred_v = base_model(x_v_noisy).argmax(1)
print("NOISY (sigma=%.2f) top-1 accuracy:" % FT_SIGMA, (pred_v == y_v).float().mean().item())
print("unique predicted classes under noise:", pred_v.unique())
print(">> Check before continuing: most/all of classes 0-9 should appear above, and")
print(">> accuracy should be meaningfully above random (0.10). If not, the fine-tuning")
print(">> did not take -- do not proceed to certification with this model.")


# ===== cell 10 =====
if os.path.exists(FT_WEIGHTS_PATH):
    raw_model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet20", pretrained=False)
    raw_model.load_state_dict(torch.load(FT_WEIGHTS_PATH))
    raw_model = raw_model.to(device).eval()
    base_model = NormalizedModel(raw_model, CIFAR10_MEAN, CIFAR10_STD).to(device).eval()
    print("loaded fine-tuned model from disk:", FT_WEIGHTS_PATH)
else:
    print("No saved weights found at", FT_WEIGHTS_PATH, "-- run Section 4 (fine-tuning) instead.")


# ===== cell 12 =====
from diffusers import UNet2DModel, DDPMScheduler

diffusion_model = UNet2DModel.from_pretrained("google/ddpm-cifar10-32").to(device).eval()
scheduler = DDPMScheduler.from_pretrained("google/ddpm-cifar10-32")
alphas_cumprod = scheduler.alphas_cumprod.to(device)

def sigma_to_timestep(sigma):
    target = sigma * 2  # diffusers images live in [-1,1]; sigma is defined in [0,1] pixel space
    noise_ratio = torch.sqrt(1 - alphas_cumprod) / torch.sqrt(alphas_cumprod)
    return torch.argmin(torch.abs(noise_ratio - target)).item()

@torch.no_grad()
def diffusion_purify(x_noisy_01, sigma):
    t = sigma_to_timestep(sigma)
    x = x_noisy_01 * 2 - 1
    t_batch = torch.full((x.shape[0],), t, device=device, dtype=torch.long)
    eps_pred = diffusion_model(x, t_batch).sample
    alpha_bar_t = alphas_cumprod[t]
    x0_pred = (x - torch.sqrt(1 - alpha_bar_t) * eps_pred) / torch.sqrt(alpha_bar_t)
    x0_pred = x0_pred.clamp(-1, 1)
    return (x0_pred + 1) / 2

print("Example: sigma=0.25 maps to diffusion timestep", sigma_to_timestep(0.25))


# ===== cell 14 =====
def clopper_pearson_lower(k, n, alpha):
    if k == 0:
        return 0.0
    return proportion_confint(k, n, alpha=2 * alpha, method="beta")[0]

@torch.no_grad()
def sample_predictions(classify_fn, x, num_samples, sigma, batch_size=400):
    counts = np.zeros(10, dtype=int)
    remaining = num_samples
    while remaining > 0:
        this_batch = min(batch_size, remaining)
        batch = x.repeat(this_batch, 1, 1, 1)
        noise = torch.randn_like(batch) * sigma
        noisy = (batch + noise).clamp(0, 1)
        preds = classify_fn(noisy)
        counts += np.bincount(preds.cpu().numpy(), minlength=10)
        remaining -= this_batch
    return counts

def certify(classify_fn, x, sigma, n0=100, n=1000, alpha=0.001):
    guess_counts = sample_predictions(classify_fn, x, n0, sigma)
    c_hat = int(np.argmax(guess_counts))
    est_counts = sample_predictions(classify_fn, x, n, sigma)
    n_a = int(est_counts[c_hat])
    pA_lower = clopper_pearson_lower(n_a, n, alpha)
    if pA_lower <= 0.5:
        return "ABSTAIN", 0.0, c_hat
    radius = sigma * norm.ppf(pA_lower)
    return c_hat, radius, c_hat


# ===== cell 16 =====
@torch.no_grad()
def classify_baseline(x_01):
    return base_model(x_01.to(device)).argmax(1).cpu()

@torch.no_grad()
def classify_purified(x_01, sigma):
    purified = diffusion_purify(x_01.to(device), sigma)
    return base_model(purified).argmax(1).cpu()


# ===== cell 18 =====
if os.path.exists(CHECKPOINT_PATH):
    os.remove(CHECKPOINT_PATH)
    print("checkpoint cleared")
else:
    print("no existing checkpoint to clear")


# ===== cell 20 =====
SEED = 0
SIGMA = 0.25
NUM_IMAGES = 10
N0, N, ALPHA = 50, 100, 0.001   # pilot scale -- raise per the table above for a real result

random.seed(SEED)
torch.manual_seed(SEED)
indices = random.sample(range(len(testset)), NUM_IMAGES)

if os.path.exists(CHECKPOINT_PATH):
    done = pd.read_csv(CHECKPOINT_PATH)
    done_this_run = done[(done.seed == SEED) & (done.sigma == SIGMA)]
    completed_indices = set(done_this_run["index"].tolist())
    print(f"resuming: {len(completed_indices)} images already done for seed={SEED}, sigma={SIGMA}")
else:
    completed_indices = set()
    pd.DataFrame(columns=["seed", "sigma", "index", "true_label", "method",
                           "pred", "radius", "correct"]).to_csv(CHECKPOINT_PATH, index=False)

for idx in indices:
    if idx in completed_indices:
        continue
    x, img_label = testset[idx]
    x = x.unsqueeze(0)

    rows = []
    pred_b, r_b, _ = certify(classify_baseline, x, SIGMA, N0, N, ALPHA)
    rows.append({"seed": SEED, "sigma": SIGMA, "index": idx, "true_label": img_label,
                 "method": "baseline", "pred": pred_b, "radius": r_b,
                 "correct": int(pred_b == img_label)})

    pred_p, r_p, _ = certify(lambda z: classify_purified(z, SIGMA), x, SIGMA, N0, N, ALPHA)
    rows.append({"seed": SEED, "sigma": SIGMA, "index": idx, "true_label": img_label,
                 "method": "purified", "pred": pred_p, "radius": r_p,
                 "correct": int(pred_p == img_label)})

    pd.DataFrame(rows).to_csv(CHECKPOINT_PATH, mode="a", header=False, index=False)
    print(f"idx={idx:5d} true={classes[img_label]:>10s} | baseline: {pred_b} r={r_b:.3f} | purified: {pred_p} r={r_p:.3f}")

print("done, results checkpointed to", CHECKPOINT_PATH)


# ===== cell 22 =====
results_df = pd.read_csv(CHECKPOINT_PATH)
run = results_df[(results_df.seed == SEED) & (results_df.sigma == SIGMA)]

summary = run.groupby("method")[["correct", "radius"]].apply(
    lambda d: pd.Series({
        "accuracy": d["correct"].mean(),
        "abstain_rate": (d["radius"] == 0).mean(),
        "mean_radius_when_correct": d.loc[d["radius"] > 0, "radius"].mean() if (d["radius"] > 0).any() else 0.0
    })
)
print(summary)
print()
print(run[["index", "method", "true_label", "pred", "radius", "correct"]].sort_values("index"))

def certified_accuracy_curve(df, method, thresholds):
    sub = df[df.method == method]
    radii = np.where(sub.correct == 1, sub.radius, 0.0)
    return [(radii >= t).mean() for t in thresholds]

thresholds = np.linspace(0, SIGMA * 2, 50)
plt.figure(figsize=(6, 4))
plt.plot(thresholds, certified_accuracy_curve(run, "baseline", thresholds), label="Baseline RS")
plt.plot(thresholds, certified_accuracy_curve(run, "purified", thresholds), label="Diffusion-purified RS")
plt.xlabel("Certified radius R")
plt.ylabel("Certified accuracy (fraction with radius >= R)")
plt.title(f"seed={SEED}, sigma={SIGMA}, n_images={NUM_IMAGES}, n_certify={N}")
plt.legend()
plt.grid(alpha=0.3)
plt.savefig(os.path.join(WORK_DIR, f"certified_accuracy_sigma{SIGMA}_seed{SEED}.png"), dpi=150)
plt.show()

