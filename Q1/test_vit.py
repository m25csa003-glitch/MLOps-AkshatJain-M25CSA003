"""
Q1 - Testing script: loads any saved model and evaluates on CIFAR-100 test set
"""
import os, argparse, json, torch
import torch.nn as nn
import timm
from peft import PeftModel
from utils import get_cifar100_loaders, evaluate, get_classwise_accuracy, \
                  plot_classwise_histogram


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--mode',      type=str, default='lora',
                   choices=['baseline', 'lora'],
                   help='Which model to test')
    p.add_argument('--lora_dir',  type=str,
                   default='../weights/Q1/lora_r4_a4',
                   help='Path to saved PEFT model dir (for lora mode)')
    p.add_argument('--baseline_ckpt', type=str,
                   default='../weights/Q1/baseline_best.pth',
                   help='Path to baseline checkpoint')
    p.add_argument('--batch_size',type=int,   default=64)
    p.add_argument('--num_workers',type=int,  default=4)
    p.add_argument('--save_dir',  type=str,   default='../weights/Q1')
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    _, _, test_loader = get_cifar100_loaders(args.batch_size, args.num_workers)
    criterion         = nn.CrossEntropyLoss()

    if args.mode == 'baseline':
        model = timm.create_model('vit_small_patch16_224', pretrained=True)
        model.head = nn.Linear(model.head.in_features, 100)
        model.load_state_dict(torch.load(args.baseline_ckpt, map_location=device))
        label = 'Baseline (No LoRA)'
    else:
        base  = timm.create_model('vit_small_patch16_224', pretrained=True)
        base.head = nn.Linear(base.head.in_features, 100)
        model = PeftModel.from_pretrained(base, args.lora_dir)
        model = model.merge_and_unload()
        label = os.path.basename(args.lora_dir)

    model = model.to(device)
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"\n{'='*40}")
    print(f"Model       : {label}")
    print(f"Test Loss   : {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.2f}%")
    print(f"{'='*40}")

    classwise = get_classwise_accuracy(model, test_loader, device)
    fig = plot_classwise_histogram(
        classwise, f'{label} — Test Acc: {test_acc:.2f}%',
        os.path.join(args.save_dir, f'test_classwise_{label}.png')
    )
    print(f"Classwise histogram saved.")


if __name__ == '__main__':
    main()
