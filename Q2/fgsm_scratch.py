"""
Q2(i) Step 2: FGSM Attack FROM SCRATCH (no ART)
"""
import os, argparse, torch, wandb
import torch.nn as nn
import torchvision.models as models
import numpy as np
from utils import (get_cifar10_loaders, normalize_batch,
                   denormalize_batch, CIFAR10_CLASSES)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt',        type=str,
                   default='../weights/Q2/resnet18_best.pth')
    p.add_argument('--epsilons',    type=float, nargs='+',
                   default=[0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3])
    p.add_argument('--batch_size',  type=int,   default=128)
    p.add_argument('--num_workers', type=int,   default=4)
    p.add_argument('--save_dir',    type=str,   default='../weights/Q2')
    p.add_argument('--wandb_project', type=str, default='DLOps-Ass5-Q2')
    return p.parse_args()


def build_resnet18():
    model = models.resnet18(pretrained=False)
    model.conv1   = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc      = nn.Linear(512, 10)
    return model


def fgsm_attack(model, images, labels, epsilon, device):
    """
    FGSM from scratch:
    x_adv = x + epsilon * sign(grad_x(Loss(f(x), y)))
    """
    images = images.clone().detach().to(device)
    labels = labels.to(device)

    # Normalize before passing to model
    images_norm = normalize_batch(images, device)
    images_norm.requires_grad_(True)

    outputs = model(images_norm)
    loss    = nn.CrossEntropyLoss()(outputs, labels)
    model.zero_grad()
    loss.backward()

    # Get sign of gradient w.r.t input
    grad_sign      = images_norm.grad.sign()
    # Apply perturbation in PIXEL space (unnormalized)
    adv_images     = images + epsilon * grad_sign
    adv_images     = torch.clamp(adv_images, 0, 1)
    return adv_images


@torch.no_grad()
def evaluate_on_adversarial(model, adv_images, labels, device):
    model.eval()
    correct, total = 0, 0
    for i in range(0, len(adv_images), 128):
        imgs = adv_images[i:i+128].to(device)
        lbs  = labels[i:i+128].to(device)
        imgs_norm = normalize_batch(imgs, device)
        out  = model(imgs_norm)
        correct += out.argmax(1).eq(lbs).sum().item()
        total   += lbs.size(0)
    return 100.0 * correct / total


def main():
    args   = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    wandb.init(project=args.wandb_project,
               name='fgsm_scratch', config=vars(args))

    # Load model
    model = build_resnet18().to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()

    # Get test data (unnormalized)
    from utils import get_cifar10_test_loader_unnormalized
    test_loader = get_cifar10_test_loader_unnormalized(
        args.batch_size, args.num_workers
    )

    # Collect all test images + labels
    all_images, all_labels = [], []
    for imgs, lbs in test_loader:
        all_images.append(imgs)
        all_labels.append(lbs)
    all_images = torch.cat(all_images)
    all_labels = torch.cat(all_labels)

    # Clean accuracy
    clean_acc = evaluate_on_adversarial(model, all_images, all_labels, device)
    print(f"Clean Accuracy: {clean_acc:.2f}%")
    wandb.log({'clean_accuracy': clean_acc})

    os.makedirs(args.save_dir, exist_ok=True)

    print(f"\n{'Epsilon':>10} {'Adv Accuracy':>14} {'Acc Drop':>10}")
    print("-" * 38)

    results = {}
    for eps in args.epsilons:
        # Generate adversarial examples in batches
        adv_list = []
        for i in range(0, len(all_images), 128):
            imgs_b = all_images[i:i+128]
            lbs_b  = all_labels[i:i+128]
            adv_b  = fgsm_attack(model, imgs_b, lbs_b, eps, device)
            adv_list.append(adv_b.cpu())
        adv_images = torch.cat(adv_list)

        adv_acc  = evaluate_on_adversarial(model, adv_images, all_labels, device)
        acc_drop = clean_acc - adv_acc
        print(f"{eps:>10.3f} {adv_acc:>13.2f}% {acc_drop:>9.2f}%")

        wandb.log({f'fgsm_scratch/eps_{eps}/accuracy': adv_acc,
                   f'fgsm_scratch/eps_{eps}/acc_drop': acc_drop,
                   'epsilon': eps})
        results[eps] = {'adv_acc': adv_acc, 'acc_drop': acc_drop,
                        'adv_images': adv_images}

    # Save sample adversarial images for eps=0.1
    eps_sample = 0.1
    torch.save({
        'clean_images':  all_images[:100],
        'adv_images':    results[eps_sample]['adv_images'][:100],
        'labels':        all_labels[:100],
        'epsilon':       eps_sample
    }, os.path.join(args.save_dir, 'fgsm_scratch_samples.pth'))

    print(f"\nSamples saved to {args.save_dir}/fgsm_scratch_samples.pth")
    wandb.finish()


if __name__ == '__main__':
    main()
