import math, pickle, time
import pandas as pd
import torch, torch.nn as nn, torch.optim as optim
from collections import Counter
from torch.utils.data import Dataset, DataLoader
import nltk
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
nltk.download('punkt', quiet=True)

# Exact baseline hyperparams — only batch_size changed 60->64
CONFIG = {"lr":1e-4, "batch_size":64, "dropout":0.1, "num_heads":8, "d_ff":2048, "num_layers":6}
NUM_EPOCHS = 80
MAX_LEN = 50
D_MODEL = 512

df = pd.read_csv('English-Hindi.tsv', sep='\t', header=None, names=["id1","en","id2","hi"])
df = df[["en","hi"]].dropna().reset_index(drop=True)
print(f"Total pairs: {len(df)}")

class Vocabulary:
    def __init__(self, freq_threshold=2):
        self.itos={0:"<pad>",1:"<sos>",2:"<eos>",3:"<unk>"}
        self.stoi={"<pad>":0,"<sos>":1,"<eos>":2,"<unk>":3}
        self.idx=4; self.freq_threshold=freq_threshold
    def build_vocab(self, sentences):
        freq=Counter()
        for s in sentences:
            for w in s.lower().strip().split(): freq[w]+=1
        for w,f in freq.items():
            if f>=self.freq_threshold:
                self.stoi[w]=self.idx; self.itos[self.idx]=w; self.idx+=1
    def numericalize(self, s):
        return [self.stoi.get(w,self.stoi["<unk>"]) for w in s.lower().strip().split()]
    def __len__(self): return len(self.stoi)
    def __getitem__(self,t): return self.stoi.get(t,self.stoi["<unk>"])

en_vocab=Vocabulary(); hi_vocab=Vocabulary()
en_vocab.build_vocab(df["en"].tolist())
hi_vocab.build_vocab(df["hi"].tolist())
print(f"EN:{len(en_vocab)} HI:{len(hi_vocab)}")

def encode(sentence, vocab, max_len=MAX_LEN):
    t=[vocab.stoi["<sos>"]]+vocab.numericalize(sentence)[:max_len-2]+[vocab.stoi["<eos>"]]
    return t+[vocab.stoi["<pad>"]]*(max_len-len(t))

class DS(Dataset):
    def __init__(self,df,ev,hv):
        self.en=df["en"].tolist(); self.hi=df["hi"].tolist(); self.ev=ev; self.hv=hv
    def __len__(self): return len(self.en)
    def __getitem__(self,i):
        return torch.tensor(encode(self.en[i],self.ev)), torch.tensor(encode(self.hi[i],self.hv))

def collate(batch):
    s,t=zip(*batch); s=torch.stack(s); t=torch.stack(t)
    return s,t[:,:-1],t[:,1:]

class PE(nn.Module):
    def __init__(self,d,ml=5000):
        super().__init__()
        pe=torch.zeros(ml,d); pos=torch.arange(0,ml).unsqueeze(1).float()
        div=torch.exp(torch.arange(0,d,2).float()*(-math.log(10000.0)/d))
        pe[:,0::2]=torch.sin(pos*div); pe[:,1::2]=torch.cos(pos*div)
        self.register_buffer('pe',pe.unsqueeze(0))
    def forward(self,x): return x+self.pe[:,:x.size(1)]

class MHA(nn.Module):
    def __init__(self,d,nh,drop=0.1):
        super().__init__()
        self.dk=d//nh; self.nh=nh; self.d=d
        self.wq=nn.Linear(d,d); self.wk=nn.Linear(d,d); self.wv=nn.Linear(d,d); self.wo=nn.Linear(d,d)
        self.drop=nn.Dropout(drop)
    def forward(self,q,k,v,mask=None):
        B=q.size(0)
        Q=self.wq(q).view(B,-1,self.nh,self.dk).transpose(1,2)
        K=self.wk(k).view(B,-1,self.nh,self.dk).transpose(1,2)
        V=self.wv(v).view(B,-1,self.nh,self.dk).transpose(1,2)
        sc=torch.matmul(Q,K.transpose(-2,-1))/(self.dk**0.5)
        if mask is not None: sc=sc.masked_fill(mask==0,-1e9)
        out=torch.matmul(self.drop(torch.softmax(sc,dim=-1)),V)
        return self.wo(out.transpose(1,2).contiguous().view(B,-1,self.d))

class FF(nn.Module):
    def __init__(self,d,dff,drop=0.1):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(d,dff),nn.ReLU(),nn.Dropout(drop),nn.Linear(dff,d))
    def forward(self,x): return self.net(x)

class LN(nn.Module):
    def __init__(self,d,eps=1e-6):
        super().__init__()
        self.g=nn.Parameter(torch.ones(d)); self.b=nn.Parameter(torch.zeros(d)); self.eps=eps
    def forward(self,x):
        m=x.mean(-1,keepdim=True); s=x.std(-1,keepdim=True)
        return self.g*(x-m)/(s+self.eps)+self.b

class EL(nn.Module):
    def __init__(self,d,nh,dff,drop):
        super().__init__()
        self.a=MHA(d,nh,drop); self.f=FF(d,dff,drop)
        self.n1=LN(d); self.n2=LN(d); self.drop=nn.Dropout(drop)
    def forward(self,x,mask=None):
        x=self.n1(x+self.drop(self.a(x,x,x,mask))); return self.n2(x+self.drop(self.f(x)))

class DL(nn.Module):
    def __init__(self,d,nh,dff,drop):
        super().__init__()
        self.sa=MHA(d,nh,drop); self.ca=MHA(d,nh,drop); self.f=FF(d,dff,drop)
        self.n1=LN(d); self.n2=LN(d); self.n3=LN(d); self.drop=nn.Dropout(drop)
    def forward(self,x,enc,sm=None,tm=None):
        x=self.n1(x+self.drop(self.sa(x,x,x,tm))); x=self.n2(x+self.drop(self.ca(x,enc,enc,sm)))
        return self.n3(x+self.drop(self.f(x)))

class TF(nn.Module):
    def __init__(self,sv,tv,d=512,nl=6,nh=8,dff=2048,ml=50,drop=0.1):
        super().__init__()
        self.ee=nn.Embedding(sv,d); self.ep=PE(d,ml)
        self.el=nn.ModuleList([EL(d,nh,dff,drop) for _ in range(nl)])
        self.de=nn.Embedding(tv,d); self.dp=PE(d,ml)
        self.dl=nn.ModuleList([DL(d,nh,dff,drop) for _ in range(nl)])
        self.fc=nn.Linear(d,tv); self.drop=nn.Dropout(drop)
    def pm(self,s,p): return (s!=p).unsqueeze(1).unsqueeze(2)
    def sm(self,sz): return torch.tril(torch.ones(sz,sz)).bool().to(next(self.parameters()).device)
    def forward(self,src,tgt,sp,tp):
        sm=self.pm(src,sp); tm=self.pm(tgt,tp)&self.sm(tgt.size(1))
        x=self.drop(self.ep(self.ee(src)))
        for l in self.el: x=l(x,sm)
        y=self.drop(self.dp(self.de(tgt)))
        for l in self.dl: y=l(y,x,sm,tm)
        return self.fc(y)

def translate(model,sentence,device):
    model.eval()
    src=torch.tensor(encode(sentence,en_vocab)).unsqueeze(0).to(device)
    out=[hi_vocab["<sos>"]]
    for _ in range(MAX_LEN):
        t=torch.tensor(out).unsqueeze(0).to(device)
        with torch.no_grad(): lg=model(src,t,en_vocab["<pad>"],hi_vocab["<pad>"])
        nxt=lg[0,-1].argmax().item(); out.append(nxt)
        if nxt==hi_vocab["<eos>"]: break
    return ' '.join([hi_vocab.itos[i] for i in out[1:-1]])

VAL=[("I love you.","मैं तुमसे प्यार करता हूँ।"),("How are you?","आप कैसे हैं?"),
     ("You should sleep.","आपको सोना चाहिए।"),("Maybe Tom doesn't love you.","टॉम शायद तुमसे प्यार नहीं करता है।"),
     ("Let me tell Tom.","मुझे टॉम को बताने दीजिए।")]

def get_bleu(model,device):
    sm=SmoothingFunction().method4; refs,hyps=[],[]
    for en,hi in VAL:
        hyps.append(translate(model,en,device).split()); refs.append([hi.split()])
    return corpus_bleu(refs,hyps,smoothing_function=sm)

if __name__=="__main__":
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device:{device}")
    SP=en_vocab["<pad>"]; TP=hi_vocab["<pad>"]
    ds=DS(df,en_vocab,hi_vocab)
    dl=DataLoader(ds,batch_size=CONFIG["batch_size"],shuffle=True,collate_fn=collate,num_workers=0)
    model=TF(len(en_vocab),len(hi_vocab),D_MODEL,CONFIG["num_layers"],CONFIG["num_heads"],
             CONFIG["d_ff"],MAX_LEN,CONFIG["dropout"]).to(device)
    crit=nn.CrossEntropyLoss(ignore_index=TP)
    opt=optim.Adam(model.parameters(),lr=CONFIG["lr"])
    # No scheduler — exact baseline Adam setup
    print(f"\nTraining {NUM_EPOCHS} epochs on GPU...\n")
    t0=time.time()
    best_bleu=0.0; best_epoch=0
    for ep in range(NUM_EPOCHS):
        model.train(); loss_sum=0.0
        for src,ti,to in dl:
            src,ti,to=src.to(device),ti.to(device),to.to(device)
            out=model(src,ti,SP,TP)
            loss=crit(out.reshape(-1,out.shape[-1]),to.reshape(-1))
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),1.0)
            opt.step(); loss_sum+=loss.item()
        avg=loss_sum/len(dl)
        if (ep+1)%5==0:
            b=get_bleu(model,device)
            elapsed=(time.time()-t0)/60
            print(f"  Epoch [{ep+1}/{NUM_EPOCHS}]  Loss:{avg:.4f}  BLEU:{b:.4f}  Time:{elapsed:.1f}min")
            if b>best_bleu:
                best_bleu=b; best_epoch=ep+1
                torch.save(model.state_dict(),"M25CSA003_ass_4_best_model.pth")
                print(f"  *** New best BLEU:{best_bleu:.4f} saved! ***")
            if best_bleu>=0.7234:
                print(f"\n  BASELINE BEAT at epoch {ep+1}! BLEU={best_bleu:.4f}")
                break
    elapsed=(time.time()-t0)/60
    print(f"\n{'='*55}")
    print(f"  Best BLEU : {best_bleu:.4f} at epoch {best_epoch}")
    print(f"  Time      : {elapsed:.2f} min")
    print(f"  Baseline  : BLEU=0.7234 (100ep, 56.55min)")
    print(f"  Status    : {'BEAT!' if best_bleu>=0.7234 else 'Below'}")
    print(f"{'='*55}")
    with open("en_vocab.pkl","wb") as f: pickle.dump(en_vocab,f)
    with open("hi_vocab.pkl","wb") as f: pickle.dump(hi_vocab,f)
    print("Done!")
