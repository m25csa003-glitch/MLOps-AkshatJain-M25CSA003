"""
Q2(i) Step 1: Train ResNet18 on CIFAR-10 from scratch
Target: >= 72% test accuracy
"""
import os, argparse, torch, wandb
import torch.nn as nn
import torchvision.models as models
from utils import get_cifar10_loaders, train_one_epoch, evaluate


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--epochs',      type=int,   default=50)
    p.add_argument('--batch_size',  type=int,   default=128)
    p.add_argument('--lr',          type=float, default=0.1)
    p.add_argument('--num_workers', type=int,   default=4)
    p.add_argument('--save_dir',    type=str,   default='../weights/Q2')
    p.add_argument('--wandb_project', type=str, default='DLOps-Ass5-Q2')
    return p.parse_args()


def build_resnet18():
    # NOT pretrained — from scratch
    model = models.resnet18(pretrained=False)
    # CIFAR-10: smaller input, modify first conv
    model.conv1   = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc      = nn.Linear(512, 10)
    return model


def main():
    args   = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    wandb.init(project=args.wandb_project,
               name='resnet18_cifar10_scratch',
               config=vars(args))

    train_loader, val_loader, test_loader = get_cifar10_loaders(
        args.batch_size, args.num_workers
    )

    model     = build_resnet18().to(device)
    total_p   = sum(p.numel() for p in model.parameters())
    print(f"Total params: {total_p:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                 momentum=0.9, weight_decay=5e-4,
                                 nesterov=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, args.epochs
    )
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None

    os.makedirs(args.save_dir, exist_ok=True)
    best_val_acc = 0.0

    print(f"\n{'Epoch':>5} {'Tr Loss':>8} {'Val Loss':>8} "
          f"{'Tr Acc':>7} {'Val Acc':>7}")
    print("-" * 45)

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc   = train_one_epoch(model, train_loader, optimizer,
                                            criterion, device, scaler)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        print(f"{epoch:>5} {tr_loss:>8.4f} {val_loss:>8.4f} "
              f"{tr_acc:>6.2f}% {val_acc:>6.2f}%")

        wandb.log({
            'epoch': epoch,
            'train/loss': tr_loss, 'train/accuracy': tr_acc,
            'val/loss':   val_loss, 'val/accuracy':   val_acc,
            'lr':         scheduler.get_last_lr()[0]
        })

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(),
                       os.path.join(args.save_dir, 'resnet18_best.pth'))
            print(f"  ✓ Best saved (val_acc={val_acc:.2f}%)")

    # Final test
    model.load_state_dict(torch.load(
        os.path.join(args.save_dir, 'resnet18_best.pth'),
        map_location=device
    ))
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"\n{'='*40}")
    print(f"Test Accuracy: {test_acc:.2f}%")
    print(f"Target (>=72%): {'✓ PASSED' if test_acc >= 72 else '✗ FAILED'}")
    print(f"{'='*40}")

    wandb.log({'test/accuracy': test_acc})
    wandb.finish()


if __name__ == '__main__':
    main()
