# -*- coding: utf-8 -*-
"""
Assignment 4 - Ray Tune + Optuna Hyperparameter Tuning
English to Hindi Transformer Translation
"""

# ─────────────────────────────────────────────
# 1. IMPORTS
# ─────────────────────────────────────────────
import os
import math
import pickle
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from collections import Counter
from torch.utils.data import Dataset, DataLoader

import ray
from ray import tune
from ray.tune import RunConfig
from ray.tune.search.optuna import OptunaSearch
from ray.tune.schedulers import ASHAScheduler
import nltk
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

nltk.download('punkt', quiet=True)

# ─────────────────────────────────────────────
# 2. DATA LOADING & PREPROCESSING
# ─────────────────────────────────────────────
df = pd.read_csv('English-Hindi.tsv', sep='\t', header=None, names=["id1", "en", "id2", "hi"])
df = df[["en", "hi"]].dropna().reset_index(drop=True)
print(f"Total sentence pairs: {len(df)}")

# ─────────────────────────────────────────────
# 3. VOCABULARY
# ─────────────────────────────────────────────
class Vocabulary:
    def __init__(self, freq_threshold=2):
        self.freq_threshold = freq_threshold
        self.itos = {0: "<pad>", 1: "<sos>", 2: "<eos>", 3: "<unk>"}
        self.stoi = {"<pad>": 0, "<sos>": 1, "<eos>": 2, "<unk>": 3}
        self.idx = 4

    def build_vocab(self, sentence_list):
        frequencies = Counter()
        for sentence in sentence_list:
            for word in self.tokenize(sentence):
                frequencies[word] += 1
        for word, freq in frequencies.items():
            if freq >= self.freq_threshold:
                self.stoi[word] = self.idx
                self.itos[self.idx] = word
                self.idx += 1

    def tokenize(self, sentence):
        return sentence.lower().strip().split()

    def numericalize(self, sentence):
        tokens = self.tokenize(sentence)
        return [self.stoi.get(token, self.stoi["<unk>"]) for token in tokens]

    def __len__(self):
        return len(self.stoi)

    def __getitem__(self, token):
        return self.stoi.get(token, self.stoi["<unk>"])


# Build vocabs globally (shared across all trials)
en_vocab = Vocabulary(freq_threshold=2)
hi_vocab = Vocabulary(freq_threshold=2)
en_vocab.build_vocab(df["en"].tolist())
hi_vocab.build_vocab(df["hi"].tolist())

print(f"English vocab size: {len(en_vocab)}")
print(f"Hindi vocab size:   {len(hi_vocab)}")

# ─────────────────────────────────────────────
# 4. ENCODING UTILITY
# ─────────────────────────────────────────────
MAX_LEN = 50

def encode_sentence(sentence, vocab, max_len=MAX_LEN):
    tokens = [vocab.stoi["<sos>"]] + vocab.numericalize(sentence)[:max_len - 2] + [vocab.stoi["<eos>"]]
    return tokens + [vocab.stoi["<pad>"]] * (max_len - len(tokens))

# ─────────────────────────────────────────────
# 5. DATASET & DATALOADER
# ─────────────────────────────────────────────
class TranslationDataset(Dataset):
    def __init__(self, df, en_vocab, hi_vocab, max_len=MAX_LEN):
        self.en_sentences = df["en"].tolist()
        self.hi_sentences = df["hi"].tolist()
        self.en_vocab = en_vocab
        self.hi_vocab = hi_vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.en_sentences)

    def __getitem__(self, idx):
        src = encode_sentence(self.en_sentences[idx], self.en_vocab, self.max_len)
        tgt = encode_sentence(self.hi_sentences[idx], self.hi_vocab, self.max_len)
        return torch.tensor(src), torch.tensor(tgt)


def collate_fn(batch):
    src_batch, tgt_batch = zip(*batch)
    src_batch = torch.stack(src_batch)
    tgt_batch = torch.stack(tgt_batch)
    tgt_input  = tgt_batch[:, :-1]
    tgt_output = tgt_batch[:, 1:]
    return src_batch, tgt_input, tgt_output

# ─────────────────────────────────────────────
# 6. TRANSFORMER MODEL ARCHITECTURE
# ─────────────────────────────────────────────
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.query_linear = nn.Linear(d_model, d_model)
        self.key_linear   = nn.Linear(d_model, d_model)
        self.value_linear = nn.Linear(d_model, d_model)
        self.out_linear   = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        B = q.size(0)
        Q = self.query_linear(q).view(B, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.key_linear(k).view(B, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.value_linear(v).view(B, -1, self.num_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn = self.dropout(torch.softmax(scores, dim=-1))
        out  = torch.matmul(attn, V).transpose(1, 2).contiguous().view(B, -1, self.d_model)
        return self.out_linear(out)


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )

    def forward(self, x):
        return self.net(x)


class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta  = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std  = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn   = FeedForward(d_model, d_ff, dropout)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.drop  = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        x = self.norm1(x + self.drop(self.self_attn(x, x, x, mask)))
        x = self.norm2(x + self.drop(self.ffn(x)))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super().__init__()
        self.self_attn  = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn   = FeedForward(d_model, d_ff, dropout)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)
        self.drop  = nn.Dropout(dropout)

    def forward(self, x, enc_out, src_mask=None, tgt_mask=None):
        x = self.norm1(x + self.drop(self.self_attn(x, x, x, tgt_mask)))
        x = self.norm2(x + self.drop(self.cross_attn(x, enc_out, enc_out, src_mask)))
        x = self.norm3(x + self.drop(self.ffn(x)))
        return x


class Encoder(nn.Module):
    def __init__(self, src_vocab_size, d_model, num_layers, num_heads, d_ff, max_len, dropout):
        super().__init__()
        self.embed   = nn.Embedding(src_vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len)
        self.layers  = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.drop    = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        x = self.drop(self.pos_enc(self.embed(x)))
        for layer in self.layers:
            x = layer(x, mask)
        return x


class Decoder(nn.Module):
    def __init__(self, tgt_vocab_size, d_model, num_layers, num_heads, d_ff, max_len, dropout):
        super().__init__()
        self.embed   = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len)
        self.layers  = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.drop    = nn.Dropout(dropout)

    def forward(self, x, enc_out, src_mask=None, tgt_mask=None):
        x = self.drop(self.pos_enc(self.embed(x)))
        for layer in self.layers:
            x = layer(x, enc_out, src_mask, tgt_mask)
        return x


class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size,
                 d_model=512, num_layers=6, num_heads=8,
                 d_ff=2048, max_len=100, dropout=0.1):
        super().__init__()
        self.encoder = Encoder(src_vocab_size, d_model, num_layers, num_heads, d_ff, max_len, dropout)
        self.decoder = Decoder(tgt_vocab_size, d_model, num_layers, num_heads, d_ff, max_len, dropout)
        self.fc_out  = nn.Linear(d_model, tgt_vocab_size)

    def make_pad_mask(self, seq, pad_idx):
        return (seq != pad_idx).unsqueeze(1).unsqueeze(2)

    def make_subsequent_mask(self, size):
        return torch.tril(torch.ones((size, size))).bool().to(next(self.parameters()).device)

    def forward(self, src, tgt, src_pad_idx, tgt_pad_idx):
        src_mask     = self.make_pad_mask(src, src_pad_idx)
        tgt_pad_mask = self.make_pad_mask(tgt, tgt_pad_idx)
        tgt_sub_mask = self.make_subsequent_mask(tgt.size(1))
        tgt_mask     = tgt_pad_mask & tgt_sub_mask

        enc_out = self.encoder(src, src_mask)
        dec_out = self.decoder(tgt, enc_out, src_mask, tgt_mask)
        return self.fc_out(dec_out)

# ─────────────────────────────────────────────
# 7. TRANSLATION & BLEU EVALUATION
# ─────────────────────────────────────────────
def translate_sentence(model, sentence, en_vocab, hi_vocab, device, max_len=MAX_LEN):
    model.eval()
    tokens     = encode_sentence(sentence, en_vocab, max_len)
    src_tensor = torch.tensor(tokens).unsqueeze(0).to(device)
    tgt_tokens = [hi_vocab["<sos>"]]

    for _ in range(max_len):
        tgt_tensor = torch.tensor(tgt_tokens).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(src_tensor, tgt_tensor, en_vocab["<pad>"], hi_vocab["<pad>"])
        next_token = output[0, -1].argmax().item()
        tgt_tokens.append(next_token)
        if next_token == hi_vocab["<eos>"]:
            break

    return ' '.join([hi_vocab.itos[idx] for idx in tgt_tokens[1:-1]])


VAL_DATASET = [
    ("I love you.",                  "मैं तुमसे प्यार करता हूँ।"),
    ("How are you?",                  "आप कैसे हैं?"),
    ("You should sleep.",             "आपको सोना चाहिए।"),
    ("Maybe Tom doesn't love you.",   "टॉम शायद तुमसे प्यार नहीं करता है।"),
    ("Let me tell Tom.",              "मुझे टॉम को बताने दीजिए।"),
]

def evaluate_bleu(model, device):
    smoothie   = SmoothingFunction().method4
    references = []
    hypotheses = []
    for en_sent, hi_sent in VAL_DATASET:
        pred        = translate_sentence(model, en_sent, en_vocab, hi_vocab, device)
        hypotheses.append(pred.split())
        references.append([hi_sent.split()])
    return corpus_bleu(references, hypotheses, smoothing_function=smoothie)

# ─────────────────────────────────────────────
# 8. RAY TUNE TRAINING FUNCTION
# ─────────────────────────────────────────────
def train_tune(config):
    """
    Ray Tune compatible training function.
    Accepts a config dict with all hyperparameters.
    Reports loss (and optionally BLEU) after each epoch.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------- Hyperparameters from config ----------
    lr         = config["lr"]
    batch_size = config["batch_size"]
    num_heads  = config["num_heads"]
    d_ff       = config["d_ff"]
    dropout    = config["dropout"]
    num_layers = config["num_layers"]
    num_epochs = config["num_epochs"]

    # d_model must be divisible by num_heads — we fix d_model=512
    D_MODEL = 512
    assert D_MODEL % num_heads == 0, f"d_model ({D_MODEL}) must be divisible by num_heads ({num_heads})"

    SRC_PAD_IDX = en_vocab["<pad>"]
    TGT_PAD_IDX = hi_vocab["<pad>"]

    # ---------- DataLoader ----------
    dataset     = TranslationDataset(df, en_vocab, hi_vocab, max_len=MAX_LEN)
    train_loader = DataLoader(dataset, batch_size=batch_size,
                              shuffle=True, collate_fn=collate_fn,
                              num_workers=0, pin_memory=False)

    # ---------- Model ----------
    model = Transformer(
        src_vocab_size=len(en_vocab),
        tgt_vocab_size=len(hi_vocab),
        d_model=D_MODEL,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        max_len=MAX_LEN,
        dropout=dropout
    ).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=TGT_PAD_IDX)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Learning rate scheduler: warmup + cosine decay
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr,
        steps_per_epoch=len(train_loader),
        epochs=num_epochs
    )

    # ---------- Training Loop ----------
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0

        for src, tgt_input, tgt_output in train_loader:
            src        = src.to(device)
            tgt_input  = tgt_input.to(device)
            tgt_output = tgt_output.to(device)

            output = model(src, tgt_input, SRC_PAD_IDX, TGT_PAD_IDX)
            output = output.reshape(-1, output.shape[-1])
            tgt_output = tgt_output.reshape(-1)

            loss = criterion(output, tgt_output)

            optimizer.zero_grad()
            loss.backward()
            # Gradient clipping for stability
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)

        # Report to Ray Tune every epoch
        # ASHA will use 'loss' to decide whether to prune
        tune.report({"loss": avg_loss, "epoch": epoch + 1})

# ─────────────────────────────────────────────
# 9. SEARCH SPACE DEFINITION
# ─────────────────────────────────────────────
search_space = {
    # Hyperparameter 1: Learning rate (log scale — most impactful)
    "lr": tune.loguniform(1e-5, 1e-3),

    # Hyperparameter 2: Batch size
    "batch_size": tune.choice([32, 64]),

    # Hyperparameter 3: Number of attention heads
    # NOTE: d_model=512, so valid num_heads must divide 512 evenly
    "num_heads": tune.choice([4, 8]),

    # Hyperparameter 4: FeedForward dimension
    "d_ff": tune.choice([1024, 2048, 4096]),

    # Hyperparameter 5: Dropout rate
    "dropout": tune.uniform(0.1, 0.4),

    # Hyperparameter 6: Number of encoder/decoder layers
    "num_layers": tune.choice([3, 4, 6]),

    # Fixed: epochs per trial (efficiency challenge — keep << 100)
    "num_epochs": 40,
}

# ─────────────────────────────────────────────
# 10. RAY TUNE SWEEP
# ─────────────────────────────────────────────
if __name__ == "__main__":

    # Initialize Ray with explicit settings to avoid warnings
    import os as _os
    _os.environ["RAY_TRAIN_ENABLE_V2_MIGRATION_WARNINGS"] = "0"
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, num_cpus=2, num_gpus=1)

    # Optuna search algorithm — minimizes 'loss'
    optuna_search = OptunaSearch(metric="loss", mode="min")

    # ASHA Scheduler — kills underperforming trials early
    # grace_period: minimum epochs before a trial can be stopped
    asha_scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=40,           # Max epochs per trial
        grace_period=5,     # Don't stop before epoch 5
        reduction_factor=2  # Halve number of surviving trials each bracket
    )

    # Configure and run the Tuner
    tuner = tune.Tuner(
        tune.with_resources(train_tune, resources={"cpu": 1, "gpu": 0.5}),
        tune_config=tune.TuneConfig(
            search_alg=optuna_search,
            scheduler=asha_scheduler,
            num_samples=20,
            max_concurrent_trials=2,  # Limit parallel trials to avoid OOM
        ),
        param_space=search_space,
        run_config=RunConfig(
            name="en_to_hi_optuna_sweep",
            storage_path=os.path.abspath("./ray_results"),
        )
    )

    print("\n" + "="*60)
    print("   Starting Ray Tune + Optuna Sweep")
    print("   - 20 trials, max 40 epochs each")
    print("   - ASHA will prune bad trials early")
    print("="*60 + "\n")

    results = tuner.fit()

    # ─────────────────────────────────────────────
    # 11. BEST CONFIG & FINAL TRAINING
    # ─────────────────────────────────────────────
    best_result = results.get_best_result(metric="loss", mode="min")
    best_config  = best_result.config

    print("\n" + "="*60)
    print("   BEST HYPERPARAMETER CONFIGURATION")
    print("="*60)
    for k, v in best_config.items():
        print(f"   {k:15s}: {v}")
    print(f"\n   Best Loss : {best_result.metrics['loss']:.4f}")
    print("="*60 + "\n")

    # ─── Retrain best model fully for BLEU evaluation ───
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    D_MODEL = 512

    best_model = Transformer(
        src_vocab_size=len(en_vocab),
        tgt_vocab_size=len(hi_vocab),
        d_model=D_MODEL,
        num_layers=best_config["num_layers"],
        num_heads=best_config["num_heads"],
        d_ff=best_config["d_ff"],
        max_len=MAX_LEN,
        dropout=best_config["dropout"]
    ).to(device)

    SRC_PAD_IDX = en_vocab["<pad>"]
    TGT_PAD_IDX = hi_vocab["<pad>"]

    dataset      = TranslationDataset(df, en_vocab, hi_vocab, max_len=MAX_LEN)
    train_loader = DataLoader(dataset, batch_size=best_config["batch_size"],
                              shuffle=True, collate_fn=collate_fn, num_workers=0)

    criterion = nn.CrossEntropyLoss(ignore_index=TGT_PAD_IDX)
    optimizer = optim.Adam(best_model.parameters(), lr=best_config["lr"])
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=best_config["lr"],
        steps_per_epoch=len(train_loader),
        epochs=best_config["num_epochs"]
    )

    import time
    print("Retraining best model for final BLEU evaluation...\n")
    t0 = time.time()

    for epoch in range(best_config["num_epochs"]):
        best_model.train()
        epoch_loss = 0.0
        for src, tgt_input, tgt_output in train_loader:
            src, tgt_input, tgt_output = src.to(device), tgt_input.to(device), tgt_output.to(device)
            output = best_model(src, tgt_input, SRC_PAD_IDX, TGT_PAD_IDX)
            output = output.reshape(-1, output.shape[-1])
            loss   = criterion(output, tgt_output.reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(best_model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item()

        avg = epoch_loss / len(train_loader)
        if (epoch + 1) % 5 == 0:
            print(f"  Epoch [{epoch+1}/{best_config['num_epochs']}]  Loss: {avg:.4f}")

    elapsed = (time.time() - t0) / 60
    print(f"\nTraining completed in {elapsed:.2f} minutes")

    # ─── BLEU Score ───
    bleu = evaluate_bleu(best_model, device)
    print(f"\n{'='*60}")
    print(f"   FINAL RESULTS (Best Tuned Model)")
    print(f"{'='*60}")
    print(f"   Training Time : {elapsed:.2f} minutes")
    print(f"   Final Loss    : {avg:.4f}")
    print(f"   BLEU Score    : {bleu:.4f}")
    print(f"\n   Baseline Comparison:")
    print(f"   Baseline BLEU : 0.7234  (100 epochs, 56.55 min)")
    print(f"   Tuned BLEU    : {bleu:.4f}  ({best_config['num_epochs']} epochs, {elapsed:.2f} min)")
    status = "✅ BEAT" if bleu >= 0.7234 else ("✅ MATCHED" if bleu >= 0.70 else "⚠️ Below")
    print(f"   Status        : {status} baseline")
    print(f"{'='*60}\n")

    # ─── Sample translations ───
    print("Sample Translations:")
    for en_sent, _ in VAL_DATASET:
        pred = translate_sentence(best_model, en_sent, en_vocab, hi_vocab, device)
        print(f"  EN: {en_sent}")
        print(f"  HI: {pred}\n")

    # ─── Save best model weights ───
    torch.save(best_model.state_dict(), "M25CSA003_ass_4_best_model.pth")

    # ─── Save vocabs ───
    with open("en_vocab.pkl", "wb") as f:
        pickle.dump(en_vocab, f)
    with open("hi_vocab.pkl", "wb") as f:
        pickle.dump(hi_vocab, f)

    print("✅ Best model saved as: M25CSA003_ass_4_best_model.pth")
    print("✅ Vocabs saved: en_vocab.pkl, hi_vocab.pkl")

    ray.shutdown()