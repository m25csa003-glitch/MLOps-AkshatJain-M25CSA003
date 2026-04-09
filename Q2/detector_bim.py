"""
Q2(ii)(b): Adversarial Detector using ResNet34
Input: Clean + BIM adversarial images
Output: Binary classification (0=clean, 1=adversarial)
Target: >= 70% detection accuracy
"""
import os, argparse, torch, wandb
import torch.nn as nn
import torchvision.models as models
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from art.attacks.evasion import BasicIterativeMethod
from art.estimators.classification import PyTorchClassifier
from utils import get_cifar10_test_loader_unnormalized


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
    p.add_argument('--max_iter',    type=int,   default=10)
    p.add_argument('--save_dir',    type=str,   default='../weights/Q2')
    p.add_argument('--wandb_project', type=str, default='DLOps-Ass5-Q2')
    return p.parse_args()


def build_victim():
    model = models.resnet18(pretrained=False)
    model.conv1   = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc      = nn.Linear(512, 10)
    return model


def build_detector():
    model = models.resnet34(pretrained=False)
    model.conv1   = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc      = nn.Linear(512, 2)
    return model


def generate_bim_samples(victim, clean_imgs, device,
                          eps, eps_step, max_iter):
    class NormModel(nn.Module):
        def __init__(self, base):
            super().__init__()
            self.base = base
            self.mean = torch.tensor([0.4914,0.4822,0.4465]).view(1,3,1,1)
            self.std  = torch.tensor([0.2023,0.1994,0.2010]).view(1,3,1,1)
        def forward(self, x):
            x = (x - self.mean.to(x.device)) / self.std.to(x.device)
            return self.base(x)

    nm  = NormModel(victim).to(device)
    clf = PyTorchClassifier(
        model       = nm,
        loss        = nn.CrossEntropyLoss(),
        optimizer   = torch.optim.SGD(nm.parameters(), lr=0.01),
        input_shape = (3, 32, 32),
        nb_classes  = 10,
        clip_values = (0.0, 1.0),
        device_type = 'gpu' if device.type == 'cuda' else 'cpu'
    )

    bim   = BasicIterativeMethod(estimator=clf, eps=eps,
                                  eps_step=eps_step,
                                  max_iter=max_iter, verbose=False)
    x_adv = bim.generate(x=clean_imgs.numpy())
    return torch.tensor(x_adv)


def build_detection_dataset(clean, adv):
    mean = torch.tensor([0.4914,0.4822,0.4465]).view(1,3,1,1)
    std  = torch.tensor([0.2023,0.1994,0.2010]).view(1,3,1,1)
    cn   = (clean - mean) / std
    an   = (adv   - mean) / std
    X    = torch.cat([cn, an])
    y    = torch.cat([torch.zeros(len(cn), dtype=torch.long),
                      torch.ones( len(an), dtype=torch.long)])
    idx  = torch.randperm(len(X))
    return TensorDataset(X[idx], y[idx])


def main():
    args   = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    wandb.init(project=args.wandb_project,
               name='detector_bim', config=vars(args))

    victim = build_victim().to(device)
    victim.load_state_dict(torch.load(args.victim_ckpt, map_location=device))
    victim.eval()

    loader = get_cifar10_test_loader_unnormalized(
        args.batch_size, args.num_workers
    )
    imgs_list, lbs_list = [], []
    for imgs, lbs in loader:
        imgs_list.append(imgs); lbs_list.append(lbs)
    all_imgs = torch.cat(imgs_list)
    all_lbs  = torch.cat(lbs_list)

    train_clean, test_clean = all_imgs[:8000], all_imgs[8000:]

    print("Generating BIM adversarial examples...")
    train_adv = generate_bim_samples(victim, train_clean, device,
                                      args.eps, args.eps_step, args.max_iter)
    test_adv  = generate_bim_samples(victim, test_clean,  device,
                                      args.eps, args.eps_step, args.max_iter)

    torch.save({'clean': test_clean[:10], 'adv': test_adv[:10],
                'labels': all_lbs[8000:8010]},
               os.path.join(args.save_dir, 'bim_samples.pth'))

    train_ds = build_detection_dataset(train_clean, train_adv)
    test_ds  = build_detection_dataset(test_clean,  test_adv)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                               shuffle=True,  num_workers=args.num_workers)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size,
                               shuffle=False, num_workers=args.num_workers)

    detector  = build_detector().to(device)
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
        detector.train()
        tr_loss, correct, total = 0.0, 0, 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            if scaler:
                with torch.cuda.amp.autocast():
                    out  = detector(X); loss = criterion(out, y)
                scaler.scale(loss).backward()
                scaler.step(optimizer); scaler.update()
            else:
                out  = detector(X); loss = criterion(out, y)
                loss.backward(); optimizer.step()
            tr_loss += loss.item() * X.size(0)
            correct += out.argmax(1).eq(y).sum().item()
            total   += X.size(0)
        tr_loss /= total; tr_acc = 100.0 * correct / total

        detector.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for X, y in test_loader:
                X, y = X.to(device), y.to(device)
                out  = detector(X); loss = criterion(out, y)
                val_loss    += loss.item() * X.size(0)
                val_correct += out.argmax(1).eq(y).sum().item()
                val_total   += X.size(0)
        val_loss /= val_total; val_acc = 100.0 * val_correct / val_total
        scheduler.step()

        print(f"{epoch:>5} {tr_loss:>8.4f} {val_loss:>8.4f} "
              f"{tr_acc:>6.2f}% {val_acc:>6.2f}%")
        wandb.log({'epoch': epoch,
                   'train/loss': tr_loss, 'train/accuracy': tr_acc,
                   'val/loss': val_loss, 'val/accuracy': val_acc})

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(detector.state_dict(),
                       os.path.join(args.save_dir, 'detector_bim_best.pth'))
            print(f"  ✓ Best saved ({val_acc:.2f}%)")

    print(f"\nBest Detection Accuracy (BIM): {best_acc:.2f}%")
    print(f"Target (>=70%): {'✓ PASSED' if best_acc >= 70 else '✗ FAILED'}")
    wandb.log({'best_detection_accuracy': best_acc})
    wandb.finish()


if __name__ == '__main__':
    main()
