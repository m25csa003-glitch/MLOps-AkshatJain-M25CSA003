import torch
import numpy as np
import matplotlib.pyplot as plt
import wandb
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_cifar100_loaders(batch_size=64, num_workers=4):
    """CIFAR-100 train/val/test dataloaders"""

    mean = (0.5071, 0.4867, 0.4408)
    std  = (0.2675, 0.2565, 0.2761)

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    test_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_full = datasets.CIFAR100(root='./data', train=True,
                                   download=True, transform=train_transform)
    test_set   = datasets.CIFAR100(root='./data', train=False,
                                   download=True, transform=test_transform)

    # 80/20 train-val split
    val_size   = int(0.2 * len(train_full))
    train_size = len(train_full) - val_size
    train_set, val_set = torch.utils.data.random_split(
        train_full, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_set, batch_size=batch_size,
                              shuffle=True,  num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_set,   batch_size=batch_size,
                              shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_set,  batch_size=batch_size,
                              shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader


def train_one_epoch(model, loader, optimizer, criterion, device, scaler=None):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        if scaler:
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += preds.eq(labels).sum().item()
        total   += images.size(0)

    return total_loss / total, 100.0 * correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss    = criterion(outputs, labels)

        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += preds.eq(labels).sum().item()
        total   += images.size(0)

    return total_loss / total, 100.0 * correct / total


@torch.no_grad()
def get_classwise_accuracy(model, loader, device, num_classes=100):
    model.eval()
    class_correct = torch.zeros(num_classes)
    class_total   = torch.zeros(num_classes)

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        preds   = outputs.argmax(dim=1)

        for c in range(num_classes):
            mask = labels == c
            class_correct[c] += preds[mask].eq(labels[mask]).sum().item()
            class_total[c]   += mask.sum().item()

    return (class_correct / class_total.clamp(min=1) * 100).numpy()


def plot_classwise_histogram(classwise_acc, title, save_path):
    fig, ax = plt.subplots(figsize=(20, 5))
    ax.bar(range(100), classwise_acc, color='steelblue', alpha=0.8)
    ax.set_xlabel('Class ID')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title(title)
    ax.set_xticks(range(0, 100, 5))
    ax.axhline(y=classwise_acc.mean(), color='red',
               linestyle='--', label=f'Mean: {classwise_acc.mean():.1f}%')
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    return fig


def log_gradient_norms(model, step):
    """Log LoRA weight gradient norms to WandB"""
    grad_dict = {}
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            if 'lora' in name.lower():
                grad_dict[f'grad_norm/{name}'] = param.grad.norm().item()
    if grad_dict:
        wandb.log(grad_dict, step=step)
