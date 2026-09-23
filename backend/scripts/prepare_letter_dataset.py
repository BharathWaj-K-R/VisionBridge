"""Convert RealSign-style ISL alphabet images into 126D two-hand landmarks."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import cv2,mediapipe as mp,numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app.services.letter_fewshot import normalize_hand_pair
LABELS=tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

def find_split(root,names):
    for name in names:
        p=root/name
        if p.is_dir(): return p
    for p in root.rglob("*"):
        if p.is_dir() and p.name in names: return p
    raise FileNotFoundError(f"Could not find dataset split: {names}")

def extract_pair(image,hands):
    result=hands.process(cv2.cvtColor(image,cv2.COLOR_BGR2RGB)); left=np.zeros(63,dtype=np.float32); right=np.zeros(63,dtype=np.float32)
    for landmarks,handedness in zip(result.multi_hand_landmarks or [],result.multi_handedness or []):
        side=handedness.classification[0].label.lower()
        vec=np.asarray([[p.x,p.y,p.z] for p in landmarks.landmark],dtype=np.float32).reshape(-1)
        if vec.size!=63: continue
        if side=="left": left[:]=vec
        elif side=="right": right[:]=vec
    try: return normalize_hand_pair(np.concatenate([left,right]).tolist())
    except ValueError: return None

def process_split(root,hands):
    xs=[]; ys=[]; skipped=0; counts={l:0 for l in LABELS}
    for folder in sorted(p for p in root.iterdir() if p.is_dir()):
        letter=folder.name.strip().upper()
        if letter not in LABELS: continue
        for path in sorted(folder.iterdir()):
            if path.suffix.lower() not in {".jpg",".jpeg",".png",".bmp"}: continue
            image=cv2.imread(str(path))
            if image is None: skipped+=1; continue
            vec=extract_pair(image,hands)
            if vec is None: skipped+=1; continue
            xs.append(vec); ys.append(LABELS.index(letter)); counts[letter]+=1
    if not xs: raise RuntimeError(f"No usable samples in {root}")
    return np.stack(xs),np.asarray(ys,dtype=np.int64),{"counts":counts,"skipped":skipped}

def main():
    p=argparse.ArgumentParser(); p.add_argument("--input-root",required=True); p.add_argument("--output-dir",required=True); a=p.parse_args()
    root=Path(a.input_root); out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True)
    splits={"train":("Training (A-Z)","Training"),"val":("Validation (A-Z)","Validation","Val"),"test":("Testing (A-Z)","Testing","Test")}
    stats={}
    with mp.solutions.hands.Hands(static_image_mode=True,max_num_hands=2,min_detection_confidence=0.5) as hands:
        for name,names in splits.items():
            x,y,report=process_split(find_split(root,names),hands); np.savez_compressed(out/(name+".npz"),x=x,y=y); stats[name]=report
            print(name,"samples=",len(x),"skipped=",report["skipped"])
    (out/"labels.json").write_text(json.dumps({"labels":list(LABELS),"stats":stats},indent=2))
if __name__=="__main__": main()
