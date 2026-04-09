"""
Q1 - Step 1: Finetune ViT-S classification head ONLY (no LoRA)
"""
import os, sys, argparse, torch, wandb
import torch.nn as nn
import timm
from utils import (get_cifar100_loaders, train_one_epoch,
                   evaluate, get_classwise_accuracy,
                   plot_classwise_histogram)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--epochs',     type=int,   default=10)
    p.add_argument('--batch_size', type=int,   default=64)
    p.add_argument('--lr',         type=float, default=1e-3)
    p.add_argument('--num_workers',type=int,   default=4)
    p.add_argument('--save_dir',   type=str,   default='../weights/Q1')
    p.add_argument('--wandb_project', type=str, default='DLOps-Ass5-Q1')
    return p.parse_args()


def build_model():
    model = timm.create_model('vit_small_patch16_224', pretrained=True)

    # Freeze everything except classification head
    for name, param in model.named_parameters():
        param.requires_grad = False

    # Replace + unfreeze head for 100 classes
    in_features = model.head.in_features
    model.head  = nn.Linear(in_features, 100)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,} / Total: {total:,}")
    return model


def main():
    args   = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    wandb.init(
        project=args.wandb_project,
        name='baseline_no_lora',
        config=vars(args)
    )

    train_loader, val_loader, test_loader = get_cifar100_loaders(
        args.batch_size, args.num_workers
    )

    model     = build_model().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=0.01
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    scaler    = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None

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
            'lr': scheduler.get_last_lr()[0]
        })

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(),
                       os.path.join(args.save_dir, 'baseline_best.pth'))
            print(f"  ✓ Best model saved (val_acc={val_acc:.2f}%)")

    # Test evaluation
    model.load_state_dict(torch.load(
        os.path.join(args.save_dir, 'baseline_best.pth'),
        map_location=device
    ))
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"\nTest Accuracy (Baseline): {test_acc:.2f}%")

    # Classwise histogram
    classwise = get_classwise_accuracy(model, test_loader, device)
    fig = plot_classwise_histogram(
        classwise,
        f'Baseline (No LoRA) — Test Acc: {test_acc:.2f}%',
        os.path.join(args.save_dir, 'baseline_classwise.png')
    )
    wandb.log({'test/accuracy': test_acc,
               'classwise_histogram': wandb.Image(fig)})

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    wandb.log({'trainable_params': trainable_params})
    wandb.finish()
    print("Done!")


if __name__ == '__main__':
    main()
