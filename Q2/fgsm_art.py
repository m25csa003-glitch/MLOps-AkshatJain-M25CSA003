"""
Q2(i) Step 3: FGSM using IBM ART + visual comparison
"""
import os, argparse, torch, wandb
import torch.nn as nn
import torchvision.models as models
import numpy as np
import matplotlib.pyplot as plt
from art.attacks.evasion import FastGradientMethod
from art.estimators.classification import PyTorchClassifier
from utils import (get_cifar10_test_loader_unnormalized,
                   normalize_batch, CIFAR10_CLASSES,
                   plot_adversarial_comparison)


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


def main():
    args   = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    wandb.init(project=args.wandb_project,
               name='fgsm_art_comparison', config=vars(args))

    # Load model
    model = build_resnet18().to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()

    # ART classifier wrapper
    # ART expects normalized input — wrap model with normalization inside
    class NormalizedResNet(nn.Module):
        def __init__(self, base):
            super().__init__()
            self.base = base
            self.mean = torch.tensor([0.4914,0.4822,0.4465]).view(1,3,1,1)
            self.std  = torch.tensor([0.2023,0.1994,0.2010]).view(1,3,1,1)
        def forward(self, x):
            x = (x - self.mean.to(x.device)) / self.std.to(x.device)
            return self.base(x)

    norm_model = NormalizedResNet(model).to(device)
    criterion  = nn.CrossEntropyLoss()

    art_classifier = PyTorchClassifier(
        model       = norm_model,
        loss        = criterion,
        optimizer   = torch.optim.SGD(norm_model.parameters(), lr=0.01),
        input_shape = (3, 32, 32),
        nb_classes  = 10,
        clip_values = (0.0, 1.0),
        device_type = 'gpu' if device.type == 'cuda' else 'cpu'
    )

    # Load test data
    test_loader = get_cifar10_test_loader_unnormalized(
        args.batch_size, args.num_workers
    )
    all_images, all_labels = [], []
    for imgs, lbs in test_loader:
        all_images.append(imgs)
        all_labels.append(lbs)
    all_images = torch.cat(all_images)
    all_labels = torch.cat(all_labels)

    x_np = all_images.numpy()
    y_np = all_labels.numpy()

    # Clean accuracy via ART
    preds_clean = art_classifier.predict(x_np, batch_size=128)
    clean_acc   = np.mean(preds_clean.argmax(1) == y_np) * 100
    print(f"Clean Accuracy (ART): {clean_acc:.2f}%")

    os.makedirs(args.save_dir, exist_ok=True)

    print(f"\n{'Epsilon':>10} {'ART Adv Acc':>12} {'Acc Drop':>10}")
    print("-" * 36)

    art_results = {}
    for eps in args.epsilons:
        fgsm    = FastGradientMethod(estimator=art_classifier, eps=eps)
        x_adv   = fgsm.generate(x=x_np[:1000])   # sample 1000 for speed
        y_sub   = y_np[:1000]

        preds_adv = art_classifier.predict(x_adv, batch_size=128)
        adv_acc   = np.mean(preds_adv.argmax(1) == y_sub) * 100
        acc_drop  = clean_acc - adv_acc
        print(f"{eps:>10.3f} {adv_acc:>11.2f}% {acc_drop:>9.2f}%")

        wandb.log({f'fgsm_art/eps_{eps}/accuracy': adv_acc,
                   f'fgsm_art/eps_{eps}/acc_drop': acc_drop,
                   'epsilon': eps})
        art_results[eps] = x_adv

    # Visual comparison for eps=0.1
    eps_vis     = 0.1
    n_show      = 10
    x_adv_vis   = art_results[eps_vis][:n_show]

    # Load scratch results for comparison
    scratch_data = torch.load(
        os.path.join(args.save_dir, 'fgsm_scratch_samples.pth'),
        map_location='cpu'
    )

    clean_t   = scratch_data['clean_images'][:n_show]
    adv_sc_t  = scratch_data['adv_images'][:n_show]
    adv_art_t = torch.tensor(x_adv_vis)
    labels_t  = scratch_data['labels'][:n_show].tolist()

    # Get predictions
    with torch.no_grad():
        def get_preds(imgs_t):
            imgs_n = normalize_batch(imgs_t, 'cpu')
            out    = model.cpu()(imgs_n)
            return out.argmax(1).tolist()

    pred_clean  = get_preds(clean_t)
    pred_scratch= get_preds(adv_sc_t)
    pred_art    = get_preds(adv_art_t)

    fig = plot_adversarial_comparison(
        clean_t, adv_sc_t, adv_art_t,
        labels_t, pred_clean, pred_scratch, pred_art,
        n=n_show,
        save_path=os.path.join(args.save_dir, 'fgsm_comparison.png')
    )

    wandb.log({'fgsm_visual_comparison': wandb.Image(fig)})

    # Save ART samples
    torch.save({
        'clean_images': all_images[:100],
        'adv_images':   torch.tensor(art_results[0.1]),
        'labels':       all_labels[:100],
        'epsilon':      0.1
    }, os.path.join(args.save_dir, 'fgsm_art_samples.pth'))

    print("\nDone! Comparison saved.")
    wandb.finish()


if __name__ == '__main__':
    main()
