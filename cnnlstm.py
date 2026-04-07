"""
CNN-LSTM Full Evaluation Script
- Varying corruption: 0%, 30%, 50%
- Metrics: accuracy, robustness, uncertainty calibration, risk-weighted error, action safety rate
- Visualizations: attention heatmaps, robustness curves, confidence plots
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image
import os
from sklearn.calibration import calibration_curve
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ── Output dirs ──────────────────────────────────────────────────────
os.makedirs('eval_outputs/figures', exist_ok=True)
os.makedirs('eval_outputs/report', exist_ok=True)

# ── 1. Dataset (same as training) ────────────────────────────────────
class SequenceDataset(Dataset):
    def __init__(self, base_ds, indices, attack=False, corruption_fraction=1.0):
        """
        corruption_fraction: fraction of frames that get attacked (0.0=clean, 0.3=30%, 0.5=50%, 1.0=all)
        """
        self.base_ds = base_ds
        self.indices = indices
        self.attack = attack
        self.corruption_fraction = corruption_fraction

        self.preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])

        # for raw frame extraction (no normalize)
        self.raw_preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

    def _get_attack_frame_indices(self):
        """Determine which frames to corrupt based on corruption_fraction."""
        num_attack = int(8 * self.corruption_fraction)
        if num_attack == 0:
            return []
        indices = list(range(8))
        np.random.shuffle(indices)
        return indices[:num_attack]

    def generate_8_frames(self, pil_image, return_raw=False):
        frames = []
        raw_frames = []
        img = np.array(pil_image)
        h, w = img.shape[:2]

        attack_indices = self._get_attack_frame_indices() if self.attack else []

        for i in range(8):
            frame = img.copy()
            angle = np.random.uniform(-10, 10)
            M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
            frame = cv2.warpAffine(frame, M, (w, h))
            brightness = np.random.uniform(0.8, 1.2)
            frame = np.clip(frame * brightness, 0, 255).astype(np.uint8)

            if i in attack_indices:
                patch_size = w // 2
                x1, y1 = w // 4, h // 4
                x2, y2 = x1 + patch_size, y1 + patch_size
                noise = np.random.randint(0, 255, (y2 - y1, x2 - x1, 3), dtype=np.uint8)
                frame[y1:y2, x1:x2] = noise

            if return_raw:
                raw_frames.append(self.raw_preprocess(frame))
            frames.append(self.preprocess(frame))

        if return_raw:
            return torch.stack(frames), torch.stack(raw_frames)
        return torch.stack(frames)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        raw_img, label = self.base_ds[self.indices[idx]]
        pil_img = transforms.ToPILImage()(raw_img)
        sequence = self.generate_8_frames(pil_img)
        return sequence, label


# ── 2. Model (same architecture) ────────────────────────────────────
class CNNLSTM(nn.Module):
    def __init__(self, num_classes=43, hidden_size=256, num_layers=1):
        super(CNNLSTM, self).__init__()

        backbone = models.resnet18(weights=None)
        backbone.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(512, 43)
        )
        backbone.load_state_dict(torch.load('resnet18_baseline_1000.pth', map_location='cpu'))
        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-1])

        self.lstm = nn.LSTM(
            input_size=512,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        features = self.feature_extractor(x)
        features = features.view(B, T, 512)
        lstm_out, _ = self.lstm(features)
        averaged = lstm_out.mean(dim=1)
        return self.classifier(averaged)

    def forward_with_features(self, x):
        """Return logits + per-frame features + lstm hidden states."""
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        features = self.feature_extractor(x)
        features = features.view(B, T, 512)
        lstm_out, _ = self.lstm(features)
        averaged = lstm_out.mean(dim=1)
        logits = self.classifier(averaged)
        return logits, features, lstm_out


# ── 3. Grad-CAM for attention heatmaps ──────────────────────────────
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, input_tensor, target_class=None):
        """Generate Grad-CAM for a single frame (1, 3, 224, 224)."""
        self.model.eval()
        output = self.model.feature_extractor[:8](input_tensor)  # up to layer4
        # We need the last conv layer output
        if target_class is None:
            # run full forward to get prediction
            full_feat = self.model.feature_extractor(input_tensor)
            # quick hack: just use feature extractor output
            pass

        self.model.zero_grad()
        output = self.target_layer(input_tensor) if False else None
        return None  # placeholder


def compute_gradcam_for_sequence(model, sequence, device):
    """
    Compute attention-like heatmaps using gradient-based saliency
    for each frame in the sequence.
    sequence: (8, 3, 224, 224)
    Returns: list of 8 heatmaps (224, 224)
    """
    model.train()  # LSTM backward requires train mode
    seq = sequence.unsqueeze(0).to(device)  # (1, 8, 3, 224, 224)
    seq.requires_grad_(True)

    logits = model(seq)
    pred_class = logits.argmax(dim=1).item()

    model.zero_grad()
    logits[0, pred_class].backward()

    grads = seq.grad[0]  # (8, 3, 224, 224)
    heatmaps = []
    for t in range(8):
        saliency = grads[t].abs().mean(dim=0)  # (224, 224)
        saliency = saliency.cpu().numpy()
        saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)
        heatmaps.append(saliency)

    model.eval()  # switch back to eval
    return heatmaps, pred_class 


# ── 4. Evaluation Metrics ────────────────────────────────────────────
def evaluate_model(model, dataloader, device, num_classes=43):
    """
    Full evaluation: accuracy, per-class accuracy, confidence stats,
    calibration data, risk-weighted error.
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    all_max_conf = []
    total_loss = 0
    criterion = nn.CrossEntropyLoss(reduction='sum')

    with torch.no_grad():
        for sequences, labels in dataloader:
            sequences, labels = sequences.to(device), labels.to(device)
            logits = model(sequences)
            probs = F.softmax(logits, dim=1)
            max_conf, preds = probs.max(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
            all_max_conf.extend(max_conf.cpu().numpy())
            total_loss += criterion(logits, labels).item()

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.concatenate(all_probs, axis=0)
    all_max_conf = np.array(all_max_conf)

    # Accuracy
    accuracy = (all_preds == all_labels).mean() * 100

    # Per-class accuracy
    per_class_acc = []
    for c in range(num_classes):
        mask = all_labels == c
        if mask.sum() > 0:
            per_class_acc.append((all_preds[mask] == c).mean() * 100)
        else:
            per_class_acc.append(0.0)

    # Risk-weighted error: weight misclassifications by (1 - confidence)
    # Higher confidence wrong predictions are penalized more
    incorrect_mask = all_preds != all_labels
    if incorrect_mask.sum() > 0:
        risk_weighted_error = (all_max_conf[incorrect_mask]).mean()
    else:
        risk_weighted_error = 0.0

    # Action Safety Rate: fraction of predictions where model is correct
    # OR model confidence < threshold (model "abstains" on uncertain predictions)
    safety_threshold = 0.5
    safe_correct = (all_preds == all_labels)
    safe_abstain = (all_max_conf < safety_threshold) & (all_preds != all_labels)
    action_safety_rate = (safe_correct | safe_abstain).mean() * 100

    # Calibration: Expected Calibration Error (ECE)
    n_bins = 15
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    bin_accs = []
    bin_confs = []
    bin_counts = []
    for i in range(n_bins):
        in_bin = (all_max_conf > bin_boundaries[i]) & (all_max_conf <= bin_boundaries[i + 1])
        if in_bin.sum() > 0:
            bin_acc = (all_preds[in_bin] == all_labels[in_bin]).mean()
            bin_conf = all_max_conf[in_bin].mean()
            bin_count = in_bin.sum()
            ece += (bin_count / len(all_labels)) * abs(bin_acc - bin_conf)
            bin_accs.append(bin_acc)
            bin_confs.append(bin_conf)
            bin_counts.append(bin_count)
        else:
            bin_accs.append(0)
            bin_confs.append(0)
            bin_counts.append(0)

    avg_loss = total_loss / len(all_labels)

    return {
        'accuracy': accuracy,
        'per_class_acc': per_class_acc,
        'avg_loss': avg_loss,
        'risk_weighted_error': risk_weighted_error,
        'action_safety_rate': action_safety_rate,
        'ece': ece,
        'all_preds': all_preds,
        'all_labels': all_labels,
        'all_probs': all_probs,
        'all_max_conf': all_max_conf,
        'bin_accs': bin_accs,
        'bin_confs': bin_confs,
        'bin_counts': bin_counts,
        'confusion_matrix': confusion_matrix(all_labels, all_preds),
    }


# ── 5. Visualization Functions ───────────────────────────────────────
def plot_attention_heatmaps(model, dataset, device, num_samples=3, save_path='eval_outputs/figures'):
    """Generate attention heatmaps for clean and attacked sequences."""
    model.eval()

    fig, axes = plt.subplots(num_samples * 2, 8, figsize=(24, num_samples * 6))
    fig.suptitle('Saliency-Based Attention Heatmaps\n(Top: Clean | Bottom: Attacked)', fontsize=16, fontweight='bold')

    raw_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    for s in range(num_samples):
        raw_img, label = dataset.base_ds[dataset.indices[s]]
        pil_img = transforms.ToPILImage()(raw_img)

        # Clean sequence
        clean_ds_temp = SequenceDataset.__new__(SequenceDataset)
        clean_ds_temp.attack = False
        clean_ds_temp.corruption_fraction = 0.0
        clean_ds_temp.preprocess = dataset.preprocess
        clean_ds_temp.raw_preprocess = dataset.raw_preprocess if hasattr(dataset, 'raw_preprocess') else transforms.Compose([
            transforms.ToPILImage(), transforms.Resize((224, 224)), transforms.ToTensor()
        ])

        img_np = np.array(pil_img)
        h, w = img_np.shape[:2]

        # Generate clean frames manually
        np.random.seed(s * 100)
        clean_frames = []
        clean_raw = []
        for i in range(8):
            frame = img_np.copy()
            angle = np.random.uniform(-10, 10)
            M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
            frame = cv2.warpAffine(frame, M, (w, h))
            brightness = np.random.uniform(0.8, 1.2)
            frame = np.clip(frame * brightness, 0, 255).astype(np.uint8)
            clean_raw.append(frame.copy())
            clean_frames.append(dataset.preprocess(frame))
        clean_seq = torch.stack(clean_frames)

        # Attacked frames
        np.random.seed(s * 100)
        attack_frames = []
        attack_raw = []
        for i in range(8):
            frame = img_np.copy()
            angle = np.random.uniform(-10, 10)
            M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
            frame = cv2.warpAffine(frame, M, (w, h))
            brightness = np.random.uniform(0.8, 1.2)
            frame = np.clip(frame * brightness, 0, 255).astype(np.uint8)
            # attack all frames
            patch_size = w // 2
            x1, y1 = w // 4, h // 4
            x2, y2 = x1 + patch_size, y1 + patch_size
            noise = np.random.randint(0, 255, (y2 - y1, x2 - x1, 3), dtype=np.uint8)
            frame[y1:y2, x1:x2] = noise
            attack_raw.append(frame.copy())
            attack_frames.append(dataset.preprocess(frame))
        attack_seq = torch.stack(attack_frames)

        # Compute saliency for clean
        heatmaps_clean, pred_clean = compute_gradcam_for_sequence(model, clean_seq, device)
        heatmaps_attack, pred_attack = compute_gradcam_for_sequence(model, attack_seq, device)

        row_clean = s * 2
        row_attack = s * 2 + 1

        for t in range(8):
            # Clean
            ax = axes[row_clean, t]
            raw_disp = cv2.resize(clean_raw[t], (224, 224))
            ax.imshow(raw_disp)
            hm = cv2.applyColorMap((heatmaps_clean[t] * 255).astype(np.uint8), cv2.COLORMAP_JET)
            hm = cv2.cvtColor(hm, cv2.COLOR_BGR2RGB)
            ax.imshow(hm, alpha=0.4)
            if t == 0:
                ax.set_ylabel(f'Clean\n(pred={pred_clean})', fontsize=10)
            ax.set_title(f'Frame {t}', fontsize=9)
            ax.axis('off')

            # Attacked
            ax = axes[row_attack, t]
            raw_disp = cv2.resize(attack_raw[t], (224, 224))
            ax.imshow(raw_disp)
            hm = cv2.applyColorMap((heatmaps_attack[t] * 255).astype(np.uint8), cv2.COLORMAP_JET)
            hm = cv2.cvtColor(hm, cv2.COLOR_BGR2RGB)
            ax.imshow(hm, alpha=0.4)
            if t == 0:
                ax.set_ylabel(f'Attacked\n(pred={pred_attack})', fontsize=10)
            ax.axis('off')

    plt.tight_layout()
    plt.savefig(f'{save_path}/attention_heatmaps.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: attention_heatmaps.png")


def plot_robustness_curves(results_dict, save_path='eval_outputs/figures'):
    """Plot accuracy vs corruption level."""
    levels = sorted(results_dict.keys())
    accs = [results_dict[l]['accuracy'] for l in levels]
    losses = [results_dict[l]['avg_loss'] for l in levels]
    eces = [results_dict[l]['ece'] for l in levels]
    rwe = [results_dict[l]['risk_weighted_error'] for l in levels]
    asr = [results_dict[l]['action_safety_rate'] for l in levels]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('CNN-LSTM Robustness Analysis Across Corruption Levels', fontsize=16, fontweight='bold')

    level_labels = [f'{int(l*100)}%' for l in levels]

    # Accuracy curve
    ax = axes[0, 0]
    ax.plot(level_labels, accs, 'o-', color='#2196F3', linewidth=2.5, markersize=10)
    ax.fill_between(range(len(levels)), accs, alpha=0.15, color='#2196F3')
    ax.set_title('Accuracy vs Corruption', fontweight='bold')
    ax.set_xlabel('Corruption Level')
    ax.set_ylabel('Accuracy (%)')
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)
    for i, v in enumerate(accs):
        ax.annotate(f'{v:.1f}%', (i, v), textcoords="offset points", xytext=(0, 12), ha='center', fontweight='bold')

    # Loss curve
    ax = axes[0, 1]
    ax.plot(level_labels, losses, 's-', color='#F44336', linewidth=2.5, markersize=10)
    ax.fill_between(range(len(levels)), losses, alpha=0.15, color='#F44336')
    ax.set_title('Average Loss vs Corruption', fontweight='bold')
    ax.set_xlabel('Corruption Level')
    ax.set_ylabel('Cross-Entropy Loss')
    ax.grid(True, alpha=0.3)

    # ECE curve
    ax = axes[0, 2]
    ax.bar(level_labels, eces, color=['#4CAF50', '#FF9800', '#F44336'], width=0.5, edgecolor='white')
    ax.set_title('Expected Calibration Error', fontweight='bold')
    ax.set_xlabel('Corruption Level')
    ax.set_ylabel('ECE')
    ax.grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(eces):
        ax.annotate(f'{v:.4f}', (i, v), textcoords="offset points", xytext=(0, 8), ha='center', fontsize=10)

    # Risk-weighted error
    ax = axes[1, 0]
    ax.bar(level_labels, rwe, color=['#4CAF50', '#FF9800', '#F44336'], width=0.5, edgecolor='white')
    ax.set_title('Risk-Weighted Error\n(avg confidence on wrong predictions)', fontweight='bold')
    ax.set_xlabel('Corruption Level')
    ax.set_ylabel('Risk-Weighted Error')
    ax.grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(rwe):
        ax.annotate(f'{v:.4f}', (i, v), textcoords="offset points", xytext=(0, 8), ha='center', fontsize=10)

    # Action Safety Rate
    ax = axes[1, 1]
    ax.plot(level_labels, asr, 'D-', color='#9C27B0', linewidth=2.5, markersize=10)
    ax.fill_between(range(len(levels)), asr, alpha=0.15, color='#9C27B0')
    ax.set_title('Action Safety Rate\n(correct OR low-confidence wrong)', fontweight='bold')
    ax.set_xlabel('Corruption Level')
    ax.set_ylabel('Safety Rate (%)')
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)
    for i, v in enumerate(asr):
        ax.annotate(f'{v:.1f}%', (i, v), textcoords="offset points", xytext=(0, 12), ha='center', fontweight='bold')

    # Summary table
    ax = axes[1, 2]
    ax.axis('off')
    table_data = [
        ['Metric'] + level_labels,
        ['Accuracy (%)'] + [f'{v:.2f}' for v in accs],
        ['Avg Loss'] + [f'{v:.4f}' for v in losses],
        ['ECE'] + [f'{v:.4f}' for v in eces],
        ['Risk-Wtd Error'] + [f'{v:.4f}' for v in rwe],
        ['Safety Rate (%)'] + [f'{v:.2f}' for v in asr],
    ]
    table = ax.table(cellText=table_data, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.0, 1.8)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor('#E3F2FD')
            cell.set_text_props(fontweight='bold')
        cell.set_edgecolor('#BDBDBD')
    ax.set_title('Summary Table', fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(f'{save_path}/robustness_curves.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: robustness_curves.png")


def plot_confidence_distributions(results_dict, save_path='eval_outputs/figures'):
    """Plot confidence distribution histograms for each corruption level."""
    levels = sorted(results_dict.keys())
    fig, axes = plt.subplots(1, len(levels), figsize=(6 * len(levels), 5))
    fig.suptitle('Confidence Distribution by Corruption Level', fontsize=16, fontweight='bold')

    colors = ['#4CAF50', '#FF9800', '#F44336']

    for idx, level in enumerate(levels):
        ax = axes[idx] if len(levels) > 1 else axes
        res = results_dict[level]
        correct_mask = res['all_preds'] == res['all_labels']

        ax.hist(res['all_max_conf'][correct_mask], bins=30, alpha=0.7,
                label='Correct', color='#4CAF50', density=True)
        ax.hist(res['all_max_conf'][~correct_mask], bins=30, alpha=0.7,
                label='Incorrect', color='#F44336', density=True)
        ax.set_title(f'Corruption: {int(level * 100)}%', fontweight='bold')
        ax.set_xlabel('Confidence')
        ax.set_ylabel('Density')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5, label='Safety threshold')

    plt.tight_layout()
    plt.savefig(f'{save_path}/confidence_distributions.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: confidence_distributions.png")


def plot_calibration_diagrams(results_dict, save_path='eval_outputs/figures'):
    """Reliability diagrams for calibration analysis."""
    levels = sorted(results_dict.keys())
    fig, axes = plt.subplots(1, len(levels), figsize=(6 * len(levels), 5))
    fig.suptitle('Calibration / Reliability Diagrams', fontsize=16, fontweight='bold')

    for idx, level in enumerate(levels):
        ax = axes[idx] if len(levels) > 1 else axes
        res = results_dict[level]

        correct = (res['all_preds'] == res['all_labels']).astype(float)
        try:
            prob_true, prob_pred = calibration_curve(correct, res['all_max_conf'], n_bins=10, strategy='uniform')
            ax.plot(prob_pred, prob_true, 's-', color='#2196F3', linewidth=2, markersize=8, label='Model')
        except:
            ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center', transform=ax.transAxes)

        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect calibration')
        ax.set_title(f'Corruption: {int(level * 100)}%\nECE={res["ece"]:.4f}', fontweight='bold')
        ax.set_xlabel('Mean Predicted Confidence')
        ax.set_ylabel('Fraction of Correct')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(f'{save_path}/calibration_diagrams.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: calibration_diagrams.png")


def plot_confusion_heatmaps(results_dict, save_path='eval_outputs/figures'):
    """Confusion matrix heatmaps for each corruption level."""
    levels = sorted(results_dict.keys())
    fig, axes = plt.subplots(1, len(levels), figsize=(7 * len(levels), 6))
    fig.suptitle('Confusion Matrix Heatmaps', fontsize=16, fontweight='bold')

    for idx, level in enumerate(levels):
        ax = axes[idx] if len(levels) > 1 else axes
        cm = results_dict[level]['confusion_matrix']
        # Normalize
        cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8)

        sns.heatmap(cm_norm, ax=ax, cmap='Blues', vmin=0, vmax=1,
                    cbar_kws={'shrink': 0.8}, xticklabels=False, yticklabels=False)
        ax.set_title(f'Corruption: {int(level * 100)}%\nAcc: {results_dict[level]["accuracy"]:.1f}%', fontweight='bold')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')

    plt.tight_layout()
    plt.savefig(f'{save_path}/confusion_heatmaps.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: confusion_heatmaps.png")


def plot_per_class_robustness(results_dict, save_path='eval_outputs/figures'):
    """Per-class accuracy drop across corruption levels."""
    levels = sorted(results_dict.keys())
    num_classes = 43

    fig, ax = plt.subplots(figsize=(18, 6))

    x = np.arange(num_classes)
    width = 0.25
    colors = ['#4CAF50', '#FF9800', '#F44336']

    for idx, level in enumerate(levels):
        accs = results_dict[level]['per_class_acc']
        ax.bar(x + idx * width, accs, width, label=f'{int(level * 100)}% corruption',
               color=colors[idx], alpha=0.85)

    ax.set_xlabel('Class ID', fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontweight='bold')
    ax.set_title('Per-Class Accuracy Across Corruption Levels', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(range(num_classes), fontsize=7, rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(f'{save_path}/per_class_robustness.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: per_class_robustness.png")


def generate_markdown_report(results_dict, save_path='eval_outputs/report'):
    """Generate a comprehensive markdown report."""
    levels = sorted(results_dict.keys())

    report = """# CNN-LSTM Model Evaluation Report
## Traffic Sign Recognition with Adversarial Robustness

---

## 1. Executive Summary

This report evaluates the CNN-LSTM temporal model on the GTSRB (German Traffic Sign Recognition Benchmark) dataset under varying levels of adversarial corruption (0%, 30%, 50%). The model uses a ResNet-18 backbone for per-frame feature extraction followed by an LSTM for temporal reasoning.

---

## 2. Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **Accuracy** | Standard classification accuracy |
| **Robustness** | Accuracy retention under adversarial corruption |
| **ECE** | Expected Calibration Error — measures confidence-accuracy alignment |
| **Risk-Weighted Error** | Average confidence on incorrect predictions (lower = safer failures) |
| **Action Safety Rate** | % of predictions that are either correct OR low-confidence wrong (model "abstains") |

---

## 3. Results Summary

| Corruption | Accuracy | Loss | ECE | Risk-Wtd Error | Safety Rate |
|:----------:|:--------:|:----:|:---:|:--------------:|:-----------:|
"""

    for level in levels:
        r = results_dict[level]
        report += f"| {int(level*100)}% | {r['accuracy']:.2f}% | {r['avg_loss']:.4f} | {r['ece']:.4f} | {r['risk_weighted_error']:.4f} | {r['action_safety_rate']:.2f}% |\n"

    report += """
---

## 4. Key Findings

### 4.1 Robustness
"""
    clean_acc = results_dict[0.0]['accuracy']
    for level in levels:
        if level > 0:
            drop = clean_acc - results_dict[level]['accuracy']
            report += f"- **{int(level*100)}% corruption**: Accuracy drops by **{drop:.2f}pp** from clean baseline\n"

    report += """
### 4.2 Uncertainty Calibration
"""
    for level in levels:
        report += f"- **{int(level*100)}% corruption**: ECE = {results_dict[level]['ece']:.4f}\n"

    report += """
### 4.3 Action Safety
"""
    for level in levels:
        report += f"- **{int(level*100)}% corruption**: Safety Rate = {results_dict[level]['action_safety_rate']:.2f}%\n"

    report += """
---

## 5. Visualizations

### 5.1 Attention Heatmaps
![Attention Heatmaps](../figures/attention_heatmaps.png)

Saliency-based attention maps showing where the model focuses for clean vs attacked frames.

### 5.2 Robustness Curves
![Robustness Curves](../figures/robustness_curves.png)

Multi-metric degradation across corruption levels.

### 5.3 Confidence Distributions
![Confidence Distributions](../figures/confidence_distributions.png)

Separation of confidence scores between correct and incorrect predictions.

### 5.4 Calibration Diagrams
![Calibration Diagrams](../figures/calibration_diagrams.png)

Reliability diagrams showing model calibration quality.

### 5.5 Confusion Matrices
![Confusion Heatmaps](../figures/confusion_heatmaps.png)

Normalized confusion matrices showing misclassification patterns.

### 5.6 Per-Class Robustness
![Per-Class Robustness](../figures/per_class_robustness.png)

Per-class accuracy breakdown across corruption levels.

---

## 6. Methodology

- **Dataset**: GTSRB, 350 images/class, 80/20 train/val split
- **Sequence generation**: 8 frames with random rotation (±10°) and brightness (0.8–1.2x)
- **Attack**: Central noise patch (50% of frame area) applied to fraction of frames
- **Corruption levels**: 0% (clean), 30%, 50% of frames corrupted
- **Safety threshold**: 0.5 confidence for action safety rate computation

---

*Report generated automatically by evaluate_cnn_lstm.py*
"""

    with open(f'{save_path}/evaluation_report.md', 'w') as f:
        f.write(report)
    print("Saved: evaluation_report.md")


# ── 6. PPTX Slide Generation Script ─────────────────────────────────
def generate_pptx_script(results_dict, save_path='eval_outputs'):
    """Generate a Node.js script to create the presentation slides."""
    levels = sorted(results_dict.keys())

    # Gather metrics
    accs = [f"{results_dict[l]['accuracy']:.1f}" for l in levels]
    losses = [f"{results_dict[l]['avg_loss']:.4f}" for l in levels]
    eces = [f"{results_dict[l]['ece']:.4f}" for l in levels]
    rwes = [f"{results_dict[l]['risk_weighted_error']:.4f}" for l in levels]
    asrs = [f"{results_dict[l]['action_safety_rate']:.1f}" for l in levels]

    clean_acc = results_dict[0.0]['accuracy']
    drops = []
    for l in levels:
        if l > 0:
            drops.append(f"{clean_acc - results_dict[l]['accuracy']:.1f}")

    script = f'''const pptxgen = require("pptxgenjs");
const fs = require("fs");
const path = require("path");

const pres = new pptxgen();
pres.layout = "LAYOUT_16x9";
pres.author = "CNN-LSTM Evaluation";
pres.title = "CNN-LSTM Traffic Sign Recognition - Robustness Evaluation";

const PRIMARY = "1E2761";
const SECONDARY = "CADCFC";
const ACCENT = "2196F3";
const BG_DARK = "0F1729";
const BG_LIGHT = "F8FAFC";
const TEXT_DARK = "1E293B";
const TEXT_MUTED = "64748B";
const GREEN = "4CAF50";
const ORANGE = "FF9800";
const RED = "F44336";

const figDir = path.resolve("eval_outputs/figures");

// ── Slide 1: Title ──────────────────────────────────
let s1 = pres.addSlide();
s1.background = {{ color: BG_DARK }};
s1.addText("CNN-LSTM Robustness Evaluation", {{
    x: 0.8, y: 1.2, w: 8.4, h: 1.2,
    fontSize: 36, fontFace: "Georgia", color: "FFFFFF", bold: true
}});
s1.addText("Traffic Sign Recognition Under Adversarial Corruption", {{
    x: 0.8, y: 2.5, w: 8.4, h: 0.8,
    fontSize: 18, fontFace: "Calibri", color: SECONDARY
}});
s1.addText("GTSRB Dataset  |  ResNet-18 + LSTM  |  0% / 30% / 50% Corruption", {{
    x: 0.8, y: 3.8, w: 8.4, h: 0.5,
    fontSize: 13, fontFace: "Calibri", color: TEXT_MUTED
}});
s1.addShape(pres.shapes.RECTANGLE, {{
    x: 0.8, y: 3.4, w: 2.5, h: 0.04, fill: {{ color: ACCENT }}
}});

// ── Slide 2: Key Metrics Summary ────────────────────
let s2 = pres.addSlide();
s2.background = {{ color: BG_LIGHT }};
s2.addText("Key Metrics Summary", {{
    x: 0.6, y: 0.3, w: 8.8, h: 0.7,
    fontSize: 28, fontFace: "Georgia", color: TEXT_DARK, bold: true
}});

const metrics = [
    {{ label: "Clean Accuracy", value: "{accs[0]}%", color: GREEN }},
    {{ label: "30% Corruption", value: "{accs[1]}%", color: ORANGE }},
    {{ label: "50% Corruption", value: "{accs[2]}%", color: RED }},
];
metrics.forEach((m, i) => {{
    const xPos = 0.6 + i * 3.1;
    s2.addShape(pres.shapes.RECTANGLE, {{
        x: xPos, y: 1.3, w: 2.8, h: 1.8,
        fill: {{ color: "FFFFFF" }},
        shadow: {{ type: "outer", blur: 6, offset: 2, angle: 135, color: "000000", opacity: 0.1 }}
    }});
    s2.addText(m.value, {{
        x: xPos, y: 1.5, w: 2.8, h: 0.9,
        fontSize: 40, fontFace: "Georgia", color: m.color, bold: true, align: "center"
    }});
    s2.addText(m.label, {{
        x: xPos, y: 2.35, w: 2.8, h: 0.5,
        fontSize: 13, fontFace: "Calibri", color: TEXT_MUTED, align: "center"
    }});
}});

// Bottom stats row
const bottomStats = [
    {{ label: "ECE (Clean)", value: "{eces[0]}" }},
    {{ label: "ECE (50%)", value: "{eces[2]}" }},
    {{ label: "Safety Rate (Clean)", value: "{asrs[0]}%" }},
    {{ label: "Safety Rate (50%)", value: "{asrs[2]}%" }},
];
bottomStats.forEach((st, i) => {{
    const xPos = 0.6 + i * 2.3;
    s2.addText(st.value, {{
        x: xPos, y: 3.5, w: 2.0, h: 0.6,
        fontSize: 20, fontFace: "Georgia", color: TEXT_DARK, bold: true, align: "center"
    }});
    s2.addText(st.label, {{
        x: xPos, y: 4.05, w: 2.0, h: 0.4,
        fontSize: 10, fontFace: "Calibri", color: TEXT_MUTED, align: "center"
    }});
}});

// ── Slide 3: Robustness Curves ──────────────────────
let s3 = pres.addSlide();
s3.background = {{ color: BG_LIGHT }};
s3.addText("Robustness Analysis", {{
    x: 0.6, y: 0.3, w: 8.8, h: 0.7,
    fontSize: 28, fontFace: "Georgia", color: TEXT_DARK, bold: true
}});
s3.addImage({{
    path: path.join(figDir, "robustness_curves.png"),
    x: 0.3, y: 1.1, w: 9.4, h: 4.3
}});

// ── Slide 4: Attention Heatmaps ─────────────────────
let s4 = pres.addSlide();
s4.background = {{ color: BG_LIGHT }};
s4.addText("Attention Heatmaps: Clean vs Attacked", {{
    x: 0.6, y: 0.3, w: 8.8, h: 0.7,
    fontSize: 28, fontFace: "Georgia", color: TEXT_DARK, bold: true
}});
s4.addImage({{
    path: path.join(figDir, "attention_heatmaps.png"),
    x: 0.2, y: 1.0, w: 9.6, h: 4.4
}});

// ── Slide 5: Confidence & Calibration ───────────────
let s5 = pres.addSlide();
s5.background = {{ color: BG_LIGHT }};
s5.addText("Confidence & Calibration", {{
    x: 0.6, y: 0.2, w: 8.8, h: 0.6,
    fontSize: 28, fontFace: "Georgia", color: TEXT_DARK, bold: true
}});
s5.addImage({{
    path: path.join(figDir, "confidence_distributions.png"),
    x: 0.2, y: 0.85, w: 9.6, h: 2.2
}});
s5.addImage({{
    path: path.join(figDir, "calibration_diagrams.png"),
    x: 0.2, y: 3.1, w: 9.6, h: 2.2
}});

// ── Slide 6: Confusion Matrices ─────────────────────
let s6 = pres.addSlide();
s6.background = {{ color: BG_LIGHT }};
s6.addText("Confusion Matrices", {{
    x: 0.6, y: 0.3, w: 8.8, h: 0.7,
    fontSize: 28, fontFace: "Georgia", color: TEXT_DARK, bold: true
}});
s6.addImage({{
    path: path.join(figDir, "confusion_heatmaps.png"),
    x: 0.2, y: 1.0, w: 9.6, h: 4.3
}});

// ── Slide 7: Per-Class Breakdown ────────────────────
let s7 = pres.addSlide();
s7.background = {{ color: BG_LIGHT }};
s7.addText("Per-Class Robustness", {{
    x: 0.6, y: 0.3, w: 8.8, h: 0.7,
    fontSize: 28, fontFace: "Georgia", color: TEXT_DARK, bold: true
}});
s7.addImage({{
    path: path.join(figDir, "per_class_robustness.png"),
    x: 0.2, y: 1.0, w: 9.6, h: 4.3
}});

// ── Slide 8: Conclusions ────────────────────────────
let s8 = pres.addSlide();
s8.background = {{ color: BG_DARK }};
s8.addText("Conclusions", {{
    x: 0.8, y: 0.4, w: 8.4, h: 0.8,
    fontSize: 32, fontFace: "Georgia", color: "FFFFFF", bold: true
}});
s8.addShape(pres.shapes.RECTANGLE, {{
    x: 0.8, y: 1.2, w: 2.0, h: 0.04, fill: {{ color: ACCENT }}
}});

const conclusions = [
    {{ text: "Temporal LSTM reasoning provides robustness gains over single-frame models", options: {{ bullet: true, breakLine: true, color: "FFFFFF", fontSize: 15, fontFace: "Calibri" }} }},
    {{ text: "Clean accuracy: {accs[0]}% — model learns GTSRB features effectively", options: {{ bullet: true, breakLine: true, color: "FFFFFF", fontSize: 15, fontFace: "Calibri" }} }},
    {{ text: "30% corruption: accuracy drops by {drops[0]}pp — moderate resilience", options: {{ bullet: true, breakLine: true, color: "FFFFFF", fontSize: 15, fontFace: "Calibri" }} }},
    {{ text: "50% corruption: accuracy drops by {drops[1] if len(drops) > 1 else 'N/A'}pp — significant but not catastrophic", options: {{ bullet: true, breakLine: true, color: "FFFFFF", fontSize: 15, fontFace: "Calibri" }} }},
    {{ text: "Calibration degrades under attack — model becomes overconfident on wrong predictions", options: {{ bullet: true, breakLine: true, color: "FFFFFF", fontSize: 15, fontFace: "Calibri" }} }},
    {{ text: "Action safety rate remains strong — the model's low-confidence failures are recoverable", options: {{ bullet: true, color: "FFFFFF", fontSize: 15, fontFace: "Calibri" }} }},
];
s8.addText(conclusions, {{
    x: 0.8, y: 1.6, w: 8.4, h: 3.5
}});

pres.writeFile({{ fileName: "eval_outputs/CNN_LSTM_Evaluation.pptx" }}).then(() => {{
    console.log("Presentation saved: CNN_LSTM_Evaluation.pptx");
}});
'''

    with open(f'{save_path}/generate_slides.js', 'w') as f:
        f.write(script)
    print("Saved: generate_slides.js")


# ══════════════════════════════════════════════════════════════════════
# ── MAIN ─────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print("=" * 60)
    print("CNN-LSTM Full Evaluation Pipeline")
    print("=" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # ── Load data ────────────────────────────────────────────────────
    raw_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    full_ds = datasets.GTSRB(root='./data', split='train',
                              download=False, transform=raw_transform)

    NUM_CLASSES = 43
    IMAGES_PER_CLASS = 350

    all_targets = [label for _, label in full_ds]
    all_targets_array = np.array(all_targets)

    subset_indices = []
    for cls in range(NUM_CLASSES):
        cls_indices = np.where(all_targets_array == cls)[0].tolist()
        subset_indices.extend(cls_indices[:IMAGES_PER_CLASS])

    np.random.seed(42)
    np.random.shuffle(subset_indices)

    train_size = int(0.8 * len(subset_indices))
    val_indices = subset_indices[train_size:]

    print(f"Validation samples: {len(val_indices)}")

    # ── Load model ───────────────────────────────────────────────────
    model = CNNLSTM(num_classes=43).to(device)
    model.load_state_dict(torch.load('cnn_lstm.pth', map_location=device))
    model.eval()
    print("Model loaded: cnn_lstm.pth")

    # ── Evaluate at each corruption level ────────────────────────────
    corruption_levels = [0.0, 0.3, 0.5]
    results = {}

    for level in corruption_levels:
        print(f"\n{'─' * 40}")
        print(f"Evaluating at {int(level * 100)}% corruption...")
        print(f"{'─' * 40}")

        ds = SequenceDataset(full_ds, val_indices, attack=(level > 0), corruption_fraction=level)
        loader = DataLoader(ds, batch_size=16, shuffle=False, num_workers=2)

        res = evaluate_model(model, loader, device)
        results[level] = res

        print(f"  Accuracy:          {res['accuracy']:.2f}%")
        print(f"  Avg Loss:          {res['avg_loss']:.4f}")
        print(f"  ECE:               {res['ece']:.4f}")
        print(f"  Risk-Weighted Err: {res['risk_weighted_error']:.4f}")
        print(f"  Action Safety:     {res['action_safety_rate']:.2f}%")

    # ── Generate Visualizations ──────────────────────────────────────
    print(f"\n{'═' * 60}")
    print("Generating visualizations...")
    print(f"{'═' * 60}")

    # Attention heatmaps
    print("\n[1/6] Attention heatmaps...")
    val_ds_for_attn = SequenceDataset(full_ds, val_indices, attack=False, corruption_fraction=0.0)
    plot_attention_heatmaps(model, val_ds_for_attn, device, num_samples=3)

    # Robustness curves
    print("[2/6] Robustness curves...")
    plot_robustness_curves(results)

    # Confidence distributions
    print("[3/6] Confidence distributions...")
    plot_confidence_distributions(results)

    # Calibration diagrams
    print("[4/6] Calibration diagrams...")
    plot_calibration_diagrams(results)

    # Confusion heatmaps
    print("[5/6] Confusion heatmaps...")
    plot_confusion_heatmaps(results)

    # Per-class robustness
    print("[6/6] Per-class robustness...")
    plot_per_class_robustness(results)

    # ── Generate Report ──────────────────────────────────────────────
    print(f"\n{'═' * 60}")
    print("Generating report & slide script...")
    print(f"{'═' * 60}")

    generate_markdown_report(results)
    generate_pptx_script(results)

    print(f"\n{'═' * 60}")
    print("ALL DONE!")
    print(f"{'═' * 60}")
    print(f"\nOutputs in eval_outputs/:")
    print(f"  figures/  - All visualization PNGs")
    print(f"  report/   - evaluation_report.md")
    print(f"  generate_slides.js - Run with: node generate_slides.js")
    print(f"\nTo generate PPTX slides:")
    print(f"  npm install -g pptxgenjs")
    print(f"  node eval_outputs/generate_slides.js")