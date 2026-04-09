"""
Q2(ii)(a): Adversarial Detector using ResNet34
Input: Clean + PGD adversarial images
Output: Binary classification (0=clean, 1=adversarial)
Target: >= 70% detection accuracy
"""
import os, argparse, torch, wandb
import torch.nn as nn
import torchvision.models as models
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from art.attacks.evasion import ProjectedGradientDescent
from art.estimators.classification import PyTorchClassifier
from utils import get_cifar10_loaders, normalize_batch, CIFAR10_CLASSES
import matplotlib.pyplot as plt


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--victim_ckpt', type=str,
                   default='../weights/Q2/resnet18_best.pth')
    p.add_argument('--epochs',      type=int,   default=20)
    p.add_argument('--batch_size',  type=int,   default=64)
    p.add_argument('--lr',          type=float, default=1e-4)
    p.add_argument('--num_workers', type=int,   default=4)
    p.add_argument('--eps',         type=float, default=0.05)
    p.add_argument('--eps_step',    type=float, default=0.01)
    p.add_argument('--max_iter',    type=int,   default=40)
    p.add_argument('--save_dir',    type=str,   default='../weights/Q2')
    p.add_argument('--wandb_project', type=str, default='DLOps-Ass5-Q2')
    return p.parse_args()


def build_victim_resnet18():
    model = models.resnet18(pretrained=False)
    model.conv1   = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc      = nn.Linear(512, 10)
    return model


def build_detector_resnet34():
    """ResNet34 for binary detection"""
    model = models.resnet34(pretrained=False)
    model.conv1   = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc      = nn.Linear(512, 2)   # binary: clean vs adversarial
    return model


def generate_pgd_samples(victim_model, clean_images, clean_labels, device,
                          eps, eps_step, max_iter):
    """Generate PGD adversarial examples using IBM ART"""
    class NormModel(nn.Module):
        def __init__(self, base):
            super().__init__()
            self.base = base
            self.mean = torch.tensor([0.4914,0.4822,0.4465]).view(1,3,1,1)
            self.std  = torch.tensor([0.2023,0.1994,0.2010]).view(1,3,1,1)
        def forward(self, x):
            x = (x - self.mean.to(x.device)) / self.std.to(x.device)
            return self.base(x)

    norm_model    = NormModel(victim_model).to(device)
    art_classifier = PyTorchClassifier(
        model       = norm_model,
        loss        = nn.CrossEntropyLoss(),
        optimizer   = torch.optim.SGD(norm_model.parameters(), lr=0.01),
        input_shape = (3, 32, 32),
        nb_classes  = 10,
        clip_values = (0.0, 1.0),
        device_type = 'gpu' if device.type == 'cuda' else 'cpu'
    )

    pgd = ProjectedGradientDescent(
        estimator   = art_classifier,
        eps         = eps,
        eps_step    = eps_step,
        max_iter    = max_iter,
        targeted    = False,
        verbose     = False
    )

    x_np  = clean_images.numpy()
    x_adv = pgd.generate(x=x_np)
    return torch.tensor(x_adv)


def build_detection_dataset(clean_imgs, adv_imgs):
    """
    Mix clean (label=0) + adversarial (label=1)
    Returns TensorDataset
    """
    # Normalize both
    mean = torch.tensor([0.4914,0.4822,0.4465]).view(1,3,1,1)
    std  = torch.tensor([0.2023,0.1994,0.2010]).view(1,3,1,1)

    clean_norm = (clean_imgs - mean) / std
    adv_norm   = (adv_imgs   - mean) / std

    X = torch.cat([clean_norm, adv_norm], dim=0)
    y = torch.cat([torch.zeros(len(clean_norm), dtype=torch.long),
                   torch.ones( len(adv_norm),   dtype=torch.long)], dim=0)

    # Shuffle
    idx = torch.randperm(len(X))
    return TensorDataset(X[idx], y[idx])


def main():
    args   = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    wandb.init(project=args.wandb_project,
               name='detector_pgd', config=vars(args))

    # Load victim model
    victim = build_victim_resnet18().to(device)
    victim.load_state_dict(torch.load(args.victim_ckpt, map_location=device))
    victim.eval()

    # Get CIFAR-10 data (unnormalized)
    from utils import get_cifar10_test_loader_unnormalized
    test_loader_raw = get_cifar10_test_loader_unnormalized(
        args.batch_size, args.num_workers
    )
    all_imgs, all_lbs = [], []
    for imgs, lbs in test_loader_raw:
        all_imgs.append(imgs)
        all_lbs.append(lbs)
    all_imgs = torch.cat(all_imgs)
    all_lbs  = torch.cat(all_lbs)

    # Use 8000 for train, 2000 for test
    train_clean = all_imgs[:8000]
    test_clean  = all_imgs[8000:]
    train_lbs   = all_lbs[:8000]
    test_lbs    = all_lbs[8000:]

    print("Generating PGD adversarial examples for training set...")
    train_adv = generate_pgd_samples(victim, train_clean, train_lbs,
                                      device, args.eps, args.eps_step,
                                      args.max_iter)
    print("Generating PGD adversarial examples for test set...")
    test_adv  = generate_pgd_samples(victim, test_clean,  test_lbs,
                                      device, args.eps, args.eps_step,
                                      args.max_iter)

    # Save samples for WandB logging
    torch.save({'clean': test_clean[:10],
                'adv':   test_adv[:10],
                'labels': test_lbs[:10]},
               os.path.join(args.save_dir, 'pgd_samples.pth'))

    # Build detection datasets
    train_ds = build_detection_dataset(train_clean, train_adv)
    test_ds  = build_detection_dataset(test_clean,  test_adv)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                               shuffle=True,  num_workers=args.num_workers)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size,
                               shuffle=False, num_workers=args.num_workers)

    # Build detector
    detector  = build_detector_resnet34().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(detector.parameters(),
                                   lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, args.epochs
    )
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None

    os.makedirs(args.save_dir, exist_ok=True)
    best_acc = 0.0

    print(f"\n{'Epoch':>5} {'Tr Loss':>8} {'Val Loss':>8} "
          f"{'Tr Acc':>7} {'Val Acc':>7}")
    print("-" * 45)

    for epoch in range(1, args.epochs + 1):
        # Train
        detector.train()
        tr_loss, correct, total = 0.0, 0, 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            if scaler:
                with torch.cuda.amp.autocast():
                    out  = detector(X)
                    loss = criterion(out, y)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                out  = detector(X)
                loss = criterion(out, y)
                loss.backward()
                optimizer.step()
            tr_loss += loss.item() * X.size(0)
            correct += out.argmax(1).eq(y).sum().item()
            total   += X.size(0)

        tr_loss /= total
        tr_acc   = 100.0 * correct / total

        # Eval
        detector.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for X, y in test_loader:
                X, y = X.to(device), y.to(device)
                out  = detector(X)
                loss = criterion(out, y)
                val_loss    += loss.item() * X.size(0)
                val_correct += out.argmax(1).eq(y).sum().item()
                val_total   += X.size(0)
        val_loss /= val_total
        val_acc   = 100.0 * val_correct / val_total

        scheduler.step()
        print(f"{epoch:>5} {tr_loss:>8.4f} {val_loss:>8.4f} "
              f"{tr_acc:>6.2f}% {val_acc:>6.2f}%")

        wandb.log({'epoch': epoch,
                   'train/loss': tr_loss, 'train/accuracy': tr_acc,
                   'val/loss':   val_loss, 'val/accuracy':   val_acc})

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(detector.state_dict(),
                       os.path.join(args.save_dir, 'detector_pgd_best.pth'))
            print(f"  ✓ Best saved (val_acc={val_acc:.2f}%)")

    print(f"\nBest Detection Accuracy (PGD): {best_acc:.2f}%")
    print(f"Target (>=70%): {'✓ PASSED' if best_acc >= 70 else '✗ FAILED'}")
    wandb.log({'best_detection_accuracy': best_acc})
    wandb.finish()


if __name__ == '__main__':
    main()
