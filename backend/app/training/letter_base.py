"""Train the VisionBridge letter base model on prepared hand landmarks."""
from __future__ import annotations
import argparse,json,random
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader,TensorDataset
from app.models.letter_model import LETTER_LABELS,VisionBridgeLetterBaseModel,save_checkpoint

def seed_everything(seed:int)->None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

def load_split(root:Path,name:str):
    path=root/(name+".npz")
    if not path.is_file(): raise FileNotFoundError(f"Missing {name} split: {path}")
    data=np.load(path,allow_pickle=False); x=torch.from_numpy(data["x"]).float(); y=torch.from_numpy(data["y"]).long()
    if x.ndim!=2 or x.shape[1]!=126 or y.ndim!=1 or len(x)!=len(y) or len(x)==0: raise ValueError(f"Invalid {name} split")
    if int(y.min())<0 or int(y.max())>=26: raise ValueError(f"Invalid {name} labels")
    return x,y

def accuracy(model,loader,device):
    model.eval(); correct=total=0
    with torch.inference_mode():
        for x,y in loader:
            pred=model(x.to(device)).argmax(1); correct+=int((pred==y.to(device)).sum()); total+=len(y)
    return correct/max(total,1)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--data-dir",required=True); p.add_argument("--output",required=True)
    p.add_argument("--epochs",type=int,default=30); p.add_argument("--batch-size",type=int,default=128)
    p.add_argument("--lr",type=float,default=1e-3); p.add_argument("--weight-decay",type=float,default=1e-4)
    p.add_argument("--patience",type=int,default=6); p.add_argument("--seed",type=int,default=42)
    a=p.parse_args(); seed_everything(a.seed)
    root=Path(a.data_dir); labels=json.loads((root/"labels.json").read_text())["labels"]
    if tuple(labels)!=LETTER_LABELS: raise ValueError("Expected A-Z label contract")
    train=DataLoader(TensorDataset(*load_split(root,"train")),batch_size=a.batch_size,shuffle=True)
    val=DataLoader(TensorDataset(*load_split(root,"val")),batch_size=a.batch_size)
    test=DataLoader(TensorDataset(*load_split(root,"test")),batch_size=a.batch_size)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"); model=VisionBridgeLetterBaseModel().to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=a.weight_decay); loss_fn=nn.CrossEntropyLoss()
    best=-1.0; best_state=None; stale=0
    for epoch in range(1,a.epochs+1):
        model.train(); total_loss=0.0; total=0
        for x,y in train:
            opt.zero_grad(set_to_none=True); loss=loss_fn(model(x.to(device)),y.to(device))
            if not torch.isfinite(loss): raise RuntimeError("Non-finite base-model loss")
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
            total_loss+=float(loss.item())*len(y); total+=len(y)
        val_acc=accuracy(model,val,device); print(f"epoch={epoch:02d} train_loss={total_loss/max(total,1):.4f} val_accuracy={val_acc:.4f}")
        if val_acc>best:
            best=val_acc; best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}; stale=0
        else:
            stale+=1
            if stale>=a.patience: break
    if best_state is None: raise RuntimeError("Base-model training produced no checkpoint")
    model.load_state_dict(best_state); test_acc=accuracy(model,test,device); save_checkpoint(model,a.output)
    print(f"best_val_accuracy={best:.4f}"); print(f"test_accuracy={test_acc:.4f}"); print(f"checkpoint={a.output}")
if __name__=="__main__": main()
