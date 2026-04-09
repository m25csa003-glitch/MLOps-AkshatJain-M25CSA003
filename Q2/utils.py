"""
Q2 Utilities - CIFAR-10 dataloaders, training, evaluation
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import wandb


CIFAR10_CLASSES = ['airplane','automobile','bird','cat','deer',
                   'dog','frog','horse','ship','truck']


def get_cifar10_loaders(batch_size=128, num_workers=4):
    mean = (0.4914, 0.4822, 0.4465)
    std  = (0.2023, 0.1994, 0.2010)

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_set = datasets.CIFAR10('./data', train=True,
                                  download=True, transform=train_transform)
    test_set  = datasets.CIFAR10('./data', train=False,
                                  download=True, transform=test_transform)

    # 80/20 split
    val_size   = int(0.2 * len(train_set))
    train_size = len(train_set) - val_size
    train_set, val_set = torch.utils.data.random_split(
        train_set, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_set, batch_size=batch_size,
                              shuffle=True,  num_workers=num_workers,
                              pin_memory=True)
    val_loader   = DataLoader(val_set,   batch_size=batch_size,
                              shuffle=False, num_workers=num_workers,
                              pin_memory=True)
    test_loader  = DataLoader(test_set,  batch_size=batch_size,
                              shuffle=False, num_workers=num_workers,
                              pin_memory=True)
    return train_loader, val_loader, test_loader


def get_cifar10_test_loader_unnormalized(batch_size=128, num_workers=4):
    """Returns test loader WITHOUT normalization — needed for FGSM visualization"""
    transform = transforms.Compose([transforms.ToTensor()])
    test_set  = datasets.CIFAR10('./data', train=False,
                                  download=True, transform=transform)
    return DataLoader(test_set, batch_size=batch_size,
                      shuffle=False, num_workers=num_workers)


def normalize_batch(x, device):
    mean = torch.tensor([0.4914,0.4822,0.4465]).view(1,3,1,1).to(device)
    std  = torch.tensor([0.2023,0.1994,0.2010]).view(1,3,1,1).to(device)
    return (x - mean) / std


def denormalize_batch(x, device):
    mean = torch.tensor([0.4914,0.4822,0.4465]).view(1,3,1,1).to(device)
    std  = torch.tensor([0.2023,0.1994,0.2010]).view(1,3,1,1).to(device)
    return x * std + mean


def train_one_epoch(model, loader, optimizer, criterion, device, scaler=None):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        if scaler:
            with torch.cuda.amp.autocast():
                out  = model(images)
                loss = criterion(out, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            out  = model(images)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * images.size(0)
        correct    += out.argmax(1).eq(labels).sum().item()
        total      += images.size(0)
    return total_loss / total, 100.0 * correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        out  = model(images)
        loss = criterion(out, labels)
        total_loss += loss.item() * images.size(0)
        correct    += out.argmax(1).eq(labels).sum().item()
        total      += images.size(0)
    return total_loss / total, 100.0 * correct / total


def plot_adversarial_comparison(clean_imgs, adv_imgs_scratch, adv_imgs_art,
                                 true_labels, pred_clean, pred_scratch, pred_art,
                                 n=10, save_path='comparison.png'):
    """
    Plot: Original | FGSM Scratch | FGSM ART
    for n samples side by side
    """
    fig, axes = plt.subplots(n, 3, figsize=(9, 3 * n))
    cols = ['Original', 'FGSM (Scratch)', 'FGSM (ART)']

    for col_idx, (imgs, preds) in enumerate([
        (clean_imgs,       pred_clean),
        (adv_imgs_scratch, pred_scratch),
        (adv_imgs_art,     pred_art)
    ]):
        for row_idx in range(n):
            ax  = axes[row_idx, col_idx]
            img = imgs[row_idx].permute(1,2,0).cpu().numpy()
            img = np.clip(img, 0, 1)
            ax.imshow(img)
            ax.axis('off')
            true_cls = CIFAR10_CLASSES[true_labels[row_idx]]
            pred_cls = CIFAR10_CLASSES[preds[row_idx]]
            color    = 'green' if true_labels[row_idx] == preds[row_idx] \
                       else 'red'
            ax.set_title(f'T:{true_cls}\nP:{pred_cls}',
                         fontsize=7, color=color)
            if row_idx == 0:
                ax.set_title(f'{cols[col_idx]}\nT:{true_cls}\nP:{pred_cls}',
                             fontsize=7, color=color)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    return fig


def log_wandb_samples(clean, adv_fgsm_scratch, adv_fgsm_art,
                       adv_pgd, adv_bim, labels, n=10):
    """Log 10 samples of each attack type to WandB"""
    def make_grid_image(imgs, title):
        fig, axes = plt.subplots(1, n, figsize=(2*n, 2))
        fig.suptitle(title, fontsize=10)
        for i in range(n):
            img = imgs[i].permute(1,2,0).cpu().numpy()
            img = np.clip(img, 0, 1)
            axes[i].imshow(img)
            axes[i].axis('off')
            axes[i].set_title(CIFAR10_CLASSES[labels[i]], fontsize=7)
        plt.tight_layout()
        return fig

    wandb.log({
        'samples/clean':            wandb.Image(make_grid_image(clean,            'Clean Images')),
        'samples/fgsm_scratch':     wandb.Image(make_grid_image(adv_fgsm_scratch, 'FGSM (Scratch)')),
        'samples/fgsm_art':         wandb.Image(make_grid_image(adv_fgsm_art,     'FGSM (IBM ART)')),
        'samples/pgd':              wandb.Image(make_grid_image(adv_pgd,          'PGD Attack')),
        'samples/bim':              wandb.Image(make_grid_image(adv_bim,          'BIM Attack')),
    })
