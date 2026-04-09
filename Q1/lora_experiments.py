"""
Q1 - Step 2: ViT-S + LoRA experiments
Combinations: Rank(2,4,8) x Alpha(2,4,8) = 9 experiments
LoRA injected in Q, K, V attention weights
Dropout: 0.1 fixed
"""
import os, argparse, torch, wandb
import torch.nn as nn
import timm
from peft import LoraConfig, get_peft_model, TaskType
from utils import (get_cifar100_loaders, train_one_epoch,
                   evaluate, get_classwise_accuracy,
                   plot_classwise_histogram, log_gradient_norms)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--rank',       type=int,   default=4)
    p.add_argument('--alpha',      type=int,   default=4)
    p.add_argument('--dropout',    type=float, default=0.1)
    p.add_argument('--epochs',     type=int,   default=10)
    p.add_argument('--batch_size', type=int,   default=64)
    p.add_argument('--lr',         type=float, default=1e-4)
    p.add_argument('--num_workers',type=int,   default=4)
    p.add_argument('--save_dir',   type=str,   default='../weights/Q1')
    p.add_argument('--wandb_project', type=str, default='DLOps-Ass5-Q1')
    return p.parse_args()


def build_lora_model(rank, alpha, dropout):
    # Load pretrained ViT-S
    model = timm.create_model('vit_small_patch16_224', pretrained=True)

    # Replace head for 100 classes
    in_features = model.head.in_features
    model.head  = nn.Linear(in_features, 100)

    # LoRA config — inject into Q, K, V projections
    lora_config = LoraConfig(
        r                = rank,
        lora_alpha       = alpha,
        lora_dropout     = dropout,
        target_modules   = ["qkv"],   # ViT-S uses fused qkv
        bias             = "none",
        modules_to_save  = ["head"],  # Keep head trainable
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,} / Total: {total:,} "
          f"({100*trainable/total:.2f}%)")
    return model, trainable


def main():
    args   = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device} | Rank: {args.rank} | "
          f"Alpha: {args.alpha} | Dropout: {args.dropout}")

    run_name = f"lora_r{args.rank}_a{args.alpha}_do{args.dropout}"

    wandb.init(
        project = args.wandb_project,
        name    = run_name,
        config  = {
            'rank': args.rank, 'alpha': args.alpha,
            'dropout': args.dropout, 'epochs': args.epochs,
            'lr': args.lr, 'batch_size': args.batch_size,
            'lora_layers': 'Q,K,V (fused qkv)'
        }
    )

    train_loader, val_loader, test_loader = get_cifar100_loaders(
        args.batch_size, args.num_workers
    )

    model, trainable_params = build_lora_model(
        args.rank, args.alpha, args.dropout
    )
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=0.01
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, args.epochs
    )
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None

    os.makedirs(args.save_dir, exist_ok=True)
    best_val_acc = 0.0
    global_step  = 0

    print(f"\n{'Epoch':>5} {'Tr Loss':>8} {'Val Loss':>8} "
          f"{'Tr Acc':>7} {'Val Acc':>7}")
    print("-" * 45)

    for epoch in range(1, args.epochs + 1):
        # Train
        model.train()
        total_loss, correct, total = 0.0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            if scaler:
                with torch.cuda.amp.autocast():
                    outputs = model(images)
                    loss    = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(images)
                loss    = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            correct    += outputs.argmax(1).eq(labels).sum().item()
            total      += images.size(0)
            global_step += 1

            # Log gradient norms every 100 steps
            if global_step % 100 == 0:
                log_gradient_norms(model, global_step)

        tr_loss = total_loss / total
        tr_acc  = 100.0 * correct / total

        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        print(f"{epoch:>5} {tr_loss:>8.4f} {val_loss:>8.4f} "
              f"{tr_acc:>6.2f}% {val_acc:>6.2f}%")

        wandb.log({
            'epoch':           epoch,
            'train/loss':      tr_loss,
            'train/accuracy':  tr_acc,
            'val/loss':        val_loss,
            'val/accuracy':    val_acc,
            'lr':              scheduler.get_last_lr()[0]
        }, step=global_step)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = os.path.join(
                args.save_dir, f'lora_r{args.rank}_a{args.alpha}_best.pth'
            )
            # Save only LoRA + head weights
            model.save_pretrained(args.save_dir +
                                  f'/lora_r{args.rank}_a{args.alpha}')
            print(f"  ✓ Best saved (val_acc={val_acc:.2f}%)")

    # Test evaluation
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"\nTest Accuracy: {test_acc:.2f}%")

    # Classwise histogram
    classwise = get_classwise_accuracy(model, test_loader, device)
    fig = plot_classwise_histogram(
        classwise,
        f'LoRA r={args.rank} α={args.alpha} — Test: {test_acc:.2f}%',
        os.path.join(args.save_dir,
                     f'classwise_r{args.rank}_a{args.alpha}.png')
    )

    wandb.log({
        'test/accuracy':      test_acc,
        'trainable_params':   trainable_params,
        'classwise_histogram': wandb.Image(fig)
    })
    wandb.finish()
    print("Done!")


if __name__ == '__main__':
    main()
