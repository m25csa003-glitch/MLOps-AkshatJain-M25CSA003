"""
Q1 - Step 5: Optuna hyperparameter search for LoRA
Searches over: rank, alpha (dropout fixed at 0.1)
"""
import os, torch, wandb, optuna
import torch.nn as nn
import timm
from peft import LoraConfig, get_peft_model
from utils import get_cifar100_loaders, train_one_epoch, evaluate


DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPOCHS      = 5          # Fewer epochs for search speed
BATCH_SIZE  = 64
NUM_WORKERS = 4
SAVE_DIR    = '../weights/Q1'
PROJECT     = 'DLOps-Ass5-Q1-Optuna'


def build_model(rank, alpha, dropout):
    model = timm.create_model('vit_small_patch16_224', pretrained=True)
    model.head = nn.Linear(model.head.in_features, 100)

    cfg   = LoraConfig(
        r               = rank,
        lora_alpha      = alpha,
        lora_dropout    = dropout,
        target_modules  = ["qkv"],
        bias            = "none",
        modules_to_save = ["head"],
    )
    return get_peft_model(model, cfg)


def objective(trial):
    rank    = trial.suggest_categorical('rank',    [2, 4, 8, 16])
    alpha   = trial.suggest_categorical('alpha',   [2, 4, 8, 16])
    dropout = trial.suggest_float('dropout', 0.0, 0.3, step=0.1)
    lr      = trial.suggest_float('lr', 1e-5, 1e-3, log=True)

    run = wandb.init(
        project  = PROJECT,
        name     = f'trial_{trial.number}_r{rank}_a{alpha}',
        config   = {'rank': rank, 'alpha': alpha,
                    'dropout': dropout, 'lr': lr},
        reinit   = True
    )

    train_loader, val_loader, _ = get_cifar100_loaders(BATCH_SIZE, NUM_WORKERS)

    model     = build_model(rank, alpha, dropout).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=0.01
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, EPOCHS)
    scaler    = torch.cuda.amp.GradScaler() if DEVICE.type == 'cuda' else None

    best_val_acc = 0.0
    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc   = train_one_epoch(model, train_loader, optimizer,
                                            criterion, DEVICE, scaler)
        val_loss, val_acc = evaluate(model, val_loader, criterion, DEVICE)
        scheduler.step()

        wandb.log({'epoch': epoch, 'val/accuracy': val_acc,
                   'train/accuracy': tr_acc})

        if val_acc > best_val_acc:
            best_val_acc = val_acc

        # Optuna pruning
        trial.report(val_acc, epoch)
        if trial.should_prune():
            run.finish()
            raise optuna.exceptions.TrialPruned()

    run.finish()
    return best_val_acc


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    pruner = optuna.pruners.MedianPruner(n_startup_trials=3,
                                          n_warmup_steps=2)
    study  = optuna.create_study(direction='maximize', pruner=pruner,
                                  study_name='lora_vit_cifar100')

    study.optimize(objective, n_trials=20, timeout=7200)

    print("\n=== Best Trial ===")
    t = study.best_trial
    print(f"  Val Accuracy : {t.value:.2f}%")
    print(f"  Rank         : {t.params['rank']}")
    print(f"  Alpha        : {t.params['alpha']}")
    print(f"  Dropout      : {t.params['dropout']}")
    print(f"  LR           : {t.params['lr']:.6f}")

    # Save best params
    import json
    with open(os.path.join(SAVE_DIR, 'best_params.json'), 'w') as f:
        json.dump(t.params, f, indent=2)
    print(f"\nBest params saved to {SAVE_DIR}/best_params.json")


if __name__ == '__main__':
    main()
