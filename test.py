"""
test.py — Trains two CIFAR-10 ResNet50 models (if not already saved),
          then runs DeepXplore differential testing on them.

Usage:
    conda activate trustworthy_ai_2
    python test.py

Outputs saved to results/:
  - disagreement_XX_m1_<c1>_m2_<c2>.png : individual disagreement images

DeepXplore reference:
  "DeepXplore: Automated Whitebox Testing of Deep Learning Systems"
  Pei et al., SOSP 2017

Modifications from the original DeepXplore implementation:
  - PyTorch instead of Keras/TensorFlow
  - CIFAR-10 (10 classes) instead of ImageNet (1000 classes)
  - Two ResNet50 models instead of VGG16/VGG19/ResNet50
  - Forward hooks (register_forward_hook) for neuron coverage tracking
  - Gradient ascent via PyTorch autograd
"""

import os
import random

import matplotlib
matplotlib.use('Agg')  # non-interactive backend for saving figures
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import ResNet50_Weights, resnet50

# Configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL1_PATH = 'models/resnet50_cifar10_model1.pth'
MODEL2_PATH = 'models/resnet50_cifar10_model2.pth'
RESULTS_DIR = 'results'

# CIFAR-10 normalization statistics
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2023, 0.1994, 0.2010)
IMG_SIZE     = 224  # standard ResNet50 input size

CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck',
]

# Training hyperparameters
TRAIN_EPOCHS     = 10
TRAIN_BATCH_SIZE = 128

# DeepXplore hyperparameters (mirroring original paper defaults)
NUM_SEEDS       = 100         # number of seed images to test
GRAD_ITERATIONS = 100         # max gradient ascent steps per seed
TRANSFORMATION  = 'light'     # 'light' | 'occl' | 'blackout'
WEIGHT_DIFF     = 2.0         # weight for differential-behavior loss
WEIGHT_NC       = 0.1         # weight for neuron-coverage loss
STEP            = 1.0         # gradient ascent step size
THRESHOLD       = 0.5         # neuron activation threshold for coverage
TARGET_MODEL    = 0           # 0 → model1 is the target model


# Training utilities  (originally train_models.py)
def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def get_dataloaders(batch_size: int = 64):
    """Return CIFAR-10 train/test DataLoaders with 224×224 resize."""
    transform_train = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    transform_test = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    trainset = torchvision.datasets.CIFAR10(
        './data', train=True,  download=True, transform=transform_train)
    testset  = torchvision.datasets.CIFAR10(
        './data', train=False, download=True, transform=transform_test)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True)
    testloader  = torch.utils.data.DataLoader(
        testset,  batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True)
    return trainloader, testloader


def build_pretrained_resnet50() -> nn.Module:
    """ResNet50 with ImageNet pretrained weights, FC replaced for 10 classes."""
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(2048, 10)
    return model


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct += outputs.argmax(dim=1).eq(targets).sum().item()
        total   += targets.size(0)
    return total_loss / len(loader), 100.0 * correct / total


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            correct += outputs.argmax(dim=1).eq(targets).sum().item()
            total   += targets.size(0)
    return total_loss / len(loader), 100.0 * correct / total


def train_model(model_id: int, optimizer_type: str, seed: int,
                epochs: int = 10, lr: float = 0.01,
                batch_size: int = 64) -> str:
    """Train a ResNet50 model on CIFAR-10 and save the best checkpoint."""
    set_seed(seed)

    print(f"\n{'='*60}")
    print(f"Training Model {model_id}")
    print(f"  Optimizer : {optimizer_type}")
    print(f"  LR        : {lr}")
    print(f"  Seed      : {seed}")
    print(f"  Epochs    : {epochs}")
    print(f"  Device    : {DEVICE}")
    print('='*60)

    trainloader, testloader = get_dataloaders(batch_size)
    model     = build_pretrained_resnet50().to(DEVICE)
    criterion = nn.CrossEntropyLoss()

    if optimizer_type == 'sgd':
        # Model 1: SGD with momentum and cosine LR schedule
        optimizer = optim.SGD(
            model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    else:
        # Model 2: Adam with step decay
        optimizer = optim.Adam(
            model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    os.makedirs('models', exist_ok=True)
    save_path = f'models/resnet50_cifar10_model{model_id}.pth'
    best_acc  = 0.0

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_epoch(
            model, trainloader, optimizer, criterion, DEVICE)
        test_loss, test_acc = evaluate(
            model, testloader, criterion, DEVICE)
        scheduler.step()

        print(f"Epoch {epoch:02d}/{epochs} | "
              f"Train Loss={train_loss:.4f} Acc={train_acc:.2f}% | "
              f"Test Loss={test_loss:.4f} Acc={test_acc:.2f}%")

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), save_path)
            print(f"  -> Saved checkpoint (best acc={best_acc:.2f}%)")

    print(f"\nModel {model_id} done. Best test accuracy: {best_acc:.2f}%")
    return save_path


# DeepXplore core  (originally deepxplore_cifar.py)
class NeuronCoverageTracker:
    """
    Tracks neuron coverage for a ResNet50 model using forward hooks.

    Coverage is measured at the channel level: a channel is "covered" if
    its mean activation (across spatial dims) exceeds `threshold` after
    min-max normalization to [0, 1].

    Hooks are registered on the four main layer blocks (layer1–4) of
    ResNet50. This gives ~3840 neurons to track (256+512+1024+2048 channels),
    avoiding the complication of shared inplace-ReLU modules.
    """

    HOOK_TARGETS = ['layer1', 'layer2', 'layer3', 'layer4']

    def __init__(self, model: nn.Module, threshold: float = 0.5):
        self.threshold = threshold
        self.coverage: dict[tuple, bool] = {}
        self._act_buffer: dict[str, torch.Tensor] = {}
        self._hooks: list = []
        self._register_hooks(model)

    def _register_hooks(self, model: nn.Module):
        for name, module in model.named_modules():
            if name in self.HOOK_TARGETS:
                def make_hook(layer_name):
                    def hook(mod, inp, out):
                        self._act_buffer[layer_name] = out
                        self._update_coverage(layer_name, out.detach())
                    return hook
                self._hooks.append(
                    module.register_forward_hook(make_hook(name)))

    def _update_coverage(self, layer_name: str, act: torch.Tensor):
        """Mark channels as covered if their mean activation > threshold."""
        if act.dim() < 2:
            return
        mean_act = act[0].mean(dim=list(range(1, act[0].dim()))) \
            if act[0].dim() > 1 else act[0]

        lo, hi = mean_act.min().item(), mean_act.max().item()
        scaled  = (mean_act - lo) / (hi - lo) if hi > lo else torch.zeros_like(mean_act)

        for idx in range(len(scaled)):
            key = (layer_name, idx)
            if key not in self.coverage:
                self.coverage[key] = False
            if scaled[idx].item() > self.threshold:
                self.coverage[key] = True

    def get_coverage_rate(self) -> float:
        if not self.coverage:
            return 0.0
        covered = sum(1 for v in self.coverage.values() if v)
        return covered / len(self.coverage)

    def get_uncovered_neuron(self) -> tuple | None:
        """Return a random uncovered (layer_name, channel_idx) pair."""
        uncovered = [k for k, v in self.coverage.items() if not v]
        if uncovered:
            return random.choice(uncovered)
        if self.coverage:
            return random.choice(list(self.coverage.keys()))
        return None

    def get_neuron_activation(self, layer_name: str, channel_idx: int) -> torch.Tensor | None:
        """Return the mean activation of a specific channel (scalar with grad_fn)."""
        act = self._act_buffer.get(layer_name)
        if act is None:
            return None
        if act.dim() == 4:
            return act[0, channel_idx, :, :].mean()
        elif act.dim() == 2:
            return act[0, channel_idx]
        return None

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


def load_model(path: str, device) -> nn.Module:
    """Load a saved CIFAR-10 ResNet50 checkpoint."""
    model = resnet50(weights=None)
    model.fc = nn.Linear(2048, 10)
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    return model


def constraint_light(grad: torch.Tensor) -> torch.Tensor:
    """Global lighting: uniform gradient scaled by mean (simulates brightness shift)."""
    grad_mean = 1e4 * grad.mean()
    return torch.ones_like(grad) * grad_mean


def constraint_occl(grad: torch.Tensor,
                    start_point: tuple = (0, 0),
                    rect_shape: tuple = (32, 32)) -> torch.Tensor:
    """Occlusion: restrict gradient update to a rectangular patch."""
    new_grad = torch.zeros_like(grad)
    h0, w0 = start_point
    h, w   = rect_shape
    new_grad[:, :, h0:h0 + h, w0:w0 + w] = grad[:, :, h0:h0 + h, w0:w0 + w]
    return new_grad


def constraint_blackout(grad: torch.Tensor,
                        rect_shape: tuple = (16, 16)) -> torch.Tensor:
    """Blackout: apply a small black patch in a random location."""
    H, W   = grad.shape[2], grad.shape[3]
    h, w   = rect_shape
    h0     = random.randint(0, max(0, H - h))
    w0     = random.randint(0, max(0, W - w))
    new_grad = torch.zeros_like(grad)
    patch  = grad[:, :, h0:h0 + h, w0:w0 + w]
    if patch.mean() < 0:
        new_grad[:, :, h0:h0 + h, w0:w0 + w] = -torch.ones_like(patch)
    return new_grad


def normalize_grad(grad: torch.Tensor) -> torch.Tensor:
    """Normalize gradient by its L2 norm."""
    return grad / (grad.norm() + 1e-8)


def run_deepxplore(
    model1: nn.Module,
    model2: nn.Module,
    tracker1: NeuronCoverageTracker,
    tracker2: NeuronCoverageTracker,
    seed_images: list,
    transformation: str = 'light',
    weight_diff: float = 1.0,
    weight_nc: float = 0.1,
    step: float = 1.0,
    grad_iterations: int = 50,
    threshold: float = 0.5,
    target_model: int = 0,
    device=None,
    start_point: tuple = (0, 0),
    occl_size: tuple = (32, 32),
    verbose: bool = True,
) -> tuple[list, tuple[float, float]]:
    """
    Run DeepXplore differential testing on two ResNet50 CIFAR-10 models.

    For each seed image:
      - If models already disagree → record immediately.
      - Otherwise run gradient ascent to maximise:
          (-weight_diff * logit_target[orig]) + logit_other[orig]
          + weight_nc * (neuron_act_1 + neuron_act_2)
        A transformation constraint restricts the allowed perturbation type.

    Returns:
        disagreement_inputs : list of (gen_img_cpu, label1, label2)
        coverage_rates      : (model1_coverage_rate, model2_coverage_rate)
    """
    if device is None:
        device = next(model1.parameters()).device

    model1.eval()
    model2.eval()
    disagreement_inputs: list = []

    for seed_idx, seed_img in enumerate(seed_images):
        seed_img = seed_img.to(device)

        with torch.no_grad():
            pred1 = model1(seed_img)
            pred2 = model2(seed_img)

        label1 = pred1.argmax(dim=1).item()
        label2 = pred2.argmax(dim=1).item()

        if verbose:
            print(f"[Seed {seed_idx + 1:03d}/{len(seed_images)}] "
                  f"M1={CIFAR10_CLASSES[label1]}, M2={CIFAR10_CLASSES[label2]}", end='')

        # Already disagree on the raw seed
        if label1 != label2:
            if verbose:
                print(f"  → already disagrees!")
            disagreement_inputs.append((seed_img.clone().cpu(), label1, label2))
            with torch.no_grad():
                model1(seed_img)
                model2(seed_img)
            continue

        # Gradient ascent to induce disagreement
        orig_label = label1
        gen_img    = seed_img.clone().detach()
        found      = False

        for iteration in range(grad_iterations):
            gen_img_var = gen_img.clone().detach().requires_grad_(True)

            out1 = model1(gen_img_var)
            out2 = model2(gen_img_var)

            cur_label1 = out1.argmax(dim=1).item()
            cur_label2 = out2.argmax(dim=1).item()

            if cur_label1 != cur_label2:
                if verbose:
                    print(f"  → disagreement at iter {iteration}: "
                          f"M1={CIFAR10_CLASSES[cur_label1]}, "
                          f"M2={CIFAR10_CLASSES[cur_label2]}")
                disagreement_inputs.append(
                    (gen_img_var.detach().cpu(), cur_label1, cur_label2))
                found = True
                break

            # Loss: push target model away from orig_label, keep other stable
            if target_model == 0:
                loss_diff = -weight_diff * out1[0, orig_label] + out2[0, orig_label]
            else:
                loss_diff =  out1[0, orig_label] - weight_diff * out2[0, orig_label]

            # Neuron coverage loss: activate uncovered channels
            nc1 = tracker1.get_uncovered_neuron()
            nc2 = tracker2.get_uncovered_neuron()
            loss_nc = torch.tensor(0.0, device=device)
            if nc1 is not None:
                act1 = tracker1.get_neuron_activation(*nc1)
                if act1 is not None:
                    loss_nc = loss_nc + act1
            if nc2 is not None:
                act2 = tracker2.get_neuron_activation(*nc2)
                if act2 is not None:
                    loss_nc = loss_nc + act2

            total_loss = loss_diff + weight_nc * loss_nc
            total_loss.backward()

            grad = gen_img_var.grad.detach()
            if transformation == 'light':
                grad = constraint_light(grad)
            elif transformation == 'occl':
                grad = constraint_occl(grad, start_point, occl_size)
            elif transformation == 'blackout':
                grad = constraint_blackout(grad)

            grad    = normalize_grad(grad)
            gen_img = (gen_img + step * grad).detach()

        if not found and verbose:
            print(f"  → no disagreement after {grad_iterations} iters")

        with torch.no_grad():
            model1(gen_img.to(device))
            model2(gen_img.to(device))

        if verbose and (seed_idx + 1) % 10 == 0:
            r1 = tracker1.get_coverage_rate()
            r2 = tracker2.get_coverage_rate()
            print(f"    Coverage so far: M1={r1:.3f}, M2={r2:.3f} | "
                  f"Disagreements: {len(disagreement_inputs)}")

    rate1 = tracker1.get_coverage_rate()
    rate2 = tracker2.get_coverage_rate()
    return disagreement_inputs, (rate1, rate2)


# Visualization & reporting  (originally test.py helpers)
def denormalize(tensor: torch.Tensor) -> np.ndarray:
    """Convert a normalized (C, H, W) tensor to a uint8 HWC numpy array."""
    mean = torch.tensor(CIFAR10_MEAN).view(3, 1, 1)
    std  = torch.tensor(CIFAR10_STD).view(3, 1, 1)
    img  = torch.clamp(tensor.cpu().squeeze(0) * std + mean, 0.0, 1.0)
    return (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)


def get_seed_images(n: int = 50) -> list:
    """Sample n random images from the CIFAR-10 test set."""
    transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    testset = torchvision.datasets.CIFAR10(
        './data', train=False, download=True, transform=transform)
    indices = random.sample(range(len(testset)), n)
    return [testset[i][0].unsqueeze(0) for i in indices]


def visualize_disagreements(disagreement_inputs: list, save_dir: str):
    """Save individual plots of disagreement-inducing inputs."""
    os.makedirs(save_dir, exist_ok=True)

    if not disagreement_inputs:
        print("No disagreement inputs to visualize.")
        return

    n_show = min(len(disagreement_inputs), 10)

    # Individual images
    for i, (img_tensor, l1, l2) in enumerate(disagreement_inputs[:n_show]):
        img    = denormalize(img_tensor)
        c1, c2 = CIFAR10_CLASSES[l1], CIFAR10_CLASSES[l2]
        fig, ax = plt.subplots(figsize=(3, 3))
        ax.imshow(img)
        ax.set_title(f'M1: {c1} | M2: {c2}', fontsize=9)
        ax.axis('off')
        fig.savefig(
            os.path.join(save_dir, f'disagreement_{i + 1:02d}_m1_{c1}_m2_{c2}.png'),
            dpi=120, bbox_inches='tight')
        plt.close(fig)


def print_summary(disagreement_inputs: list, rates: tuple):
    """Print a structured results summary."""
    r1, r2 = rates
    print()
    print('=' * 60)
    print('DEEPXPLORE RESULTS SUMMARY')
    print('=' * 60)
    print(f'Total disagreement-inducing inputs : {len(disagreement_inputs)}')
    print(f'Neuron coverage – Model 1          : {r1:.4f} ({r1 * 100:.1f}%)')
    print(f'Neuron coverage – Model 2          : {r2:.4f} ({r2 * 100:.1f}%)')
    print(f'Average neuron coverage            : '
          f'{(r1 + r2) / 2:.4f} ({(r1 + r2) / 2 * 100:.1f}%)')

    if disagreement_inputs:
        print()
        print('Disagreement class pairs:')
        pairs: dict = {}
        for _, l1, l2 in disagreement_inputs:
            key = (CIFAR10_CLASSES[l1], CIFAR10_CLASSES[l2])
            pairs[key] = pairs.get(key, 0) + 1
        for (c1, c2), cnt in sorted(pairs.items(), key=lambda x: -x[1]):
            print(f'  Model1={c1:12s} | Model2={c2:12s} : {cnt}')
    print('=' * 60)



def main():
    print(f"Device: {DEVICE}")
    print(f"PyTorch version: {torch.__version__}")

    # Train models if checkpoints are missing
    if not os.path.exists(MODEL1_PATH) or not os.path.exists(MODEL2_PATH):
        print("\nModel checkpoints not found. Training from scratch...")
        train_model(
            model_id=1, optimizer_type='sgd', seed=42,
            epochs=TRAIN_EPOCHS, lr=0.01, batch_size=TRAIN_BATCH_SIZE,
        )
        train_model(
            model_id=2, optimizer_type='adam', seed=123,
            epochs=TRAIN_EPOCHS, lr=0.001, batch_size=TRAIN_BATCH_SIZE,
        )
        print("\nBoth models trained and saved to models/")
    else:
        print(f"\nFound existing checkpoints — skipping training.")

    # Load models 
    print(f"\nLoading Model 1 from {MODEL1_PATH}")
    model1 = load_model(MODEL1_PATH, DEVICE)
    print(f"Loading Model 2 from {MODEL2_PATH}")
    model2 = load_model(MODEL2_PATH, DEVICE)

    # Set up neuron coverage trackers
    print("\nRegistering neuron coverage hooks...")
    tracker1 = NeuronCoverageTracker(model1, threshold=THRESHOLD)
    tracker2 = NeuronCoverageTracker(model2, threshold=THRESHOLD)
    print(f"  Tracker 1 monitors {len(tracker1.HOOK_TARGETS)} ResNet layers")
    print(f"  Tracker 2 monitors {len(tracker2.HOOK_TARGETS)} ResNet layers")

    # Load seed images 
    print(f"\nSampling {NUM_SEEDS} seed images from CIFAR-10 test set...")
    random.seed(0)
    seed_images = get_seed_images(NUM_SEEDS)
    print(f"  Loaded {len(seed_images)} seed images (224×224, normalized)")

    # Run DeepXplore 
    print(f"\nRunning DeepXplore (transformation='{TRANSFORMATION}')...")
    print(f"  weight_diff={WEIGHT_DIFF}, weight_nc={WEIGHT_NC}, "
          f"step={STEP}, iterations={GRAD_ITERATIONS}")

    disagreements, rates = run_deepxplore(
        model1, model2, tracker1, tracker2,
        seed_images=seed_images,
        transformation=TRANSFORMATION,
        weight_diff=WEIGHT_DIFF,
        weight_nc=WEIGHT_NC,
        step=STEP,
        grad_iterations=GRAD_ITERATIONS,
        threshold=THRESHOLD,
        target_model=TARGET_MODEL,
        device=DEVICE,
        verbose=True,
    )

    print_summary(disagreements, rates)
    print(f"\nSaving visualizations to '{RESULTS_DIR}/'...")
    visualize_disagreements(disagreements, RESULTS_DIR)

    tracker1.remove_hooks()
    tracker2.remove_hooks()
    print("\nDone.")


if __name__ == '__main__':
    main()
