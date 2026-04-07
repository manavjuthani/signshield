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


os.makedirs('eval_outputs/figures', exist_ok=True)
os.makedirs('eval_outputs/report', exist_ok=True)

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
    model.train()  
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


    incorrect_mask = all_preds != all_labels
    if incorrect_mask.sum() > 0:
        risk_weighted_error = (all_max_conf[incorrect_mask]).mean()
    else:
        risk_weighted_error = 0.0

    safety_threshold = 0.5
    safe_correct = (all_preds == all_labels)
    safe_abstain = (all_max_conf < safety_threshold) & (all_preds != all_labels)
    action_safety_rate = (safe_correct | safe_abstain).mean() * 100

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