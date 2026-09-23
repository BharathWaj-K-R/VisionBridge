"""Frozen base model for isolated ISL letter recognition."""
from __future__ import annotations
from pathlib import Path
import torch
from torch import nn

INPUT_DIM=126
EMBEDDING_DIM=64
NUM_CLASSES=26
LETTER_LABELS=tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
MODEL_VERSION="visionbridge-letter-base-v1"

class VisionBridgeLetterBaseModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_dim=INPUT_DIM
        self.embedding_dim=EMBEDDING_DIM
        self.encoder=nn.Sequential(
            nn.LayerNorm(INPUT_DIM),
            nn.Linear(INPUT_DIM,128),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(128,EMBEDDING_DIM),
            nn.LayerNorm(EMBEDDING_DIM),
            nn.GELU(),
        )
        self.output_head=nn.Linear(EMBEDDING_DIM,NUM_CLASSES)
    def embed(self,inputs:torch.Tensor)->torch.Tensor:
        if inputs.ndim!=2 or inputs.shape[-1]!=INPUT_DIM:
            raise ValueError(f"Expected [batch, {INPUT_DIM}] hand features, got {tuple(inputs.shape)}")
        return self.encoder(inputs)
    def forward(self,inputs:torch.Tensor)->torch.Tensor:
        return self.output_head(self.embed(inputs))

def build_checkpoint(model:VisionBridgeLetterBaseModel)->dict:
    return {"model_version":MODEL_VERSION,"input_dim":INPUT_DIM,"embedding_dim":EMBEDDING_DIM,"num_classes":NUM_CLASSES,"labels":list(LETTER_LABELS),"state_dict":model.state_dict()}

def save_checkpoint(model:VisionBridgeLetterBaseModel,path:str|Path)->None:
    target=Path(path); target.parent.mkdir(parents=True,exist_ok=True); torch.save(build_checkpoint(model),target)

def load_checkpoint(path:str|Path)->VisionBridgeLetterBaseModel:
    target=Path(path)
    if not target.is_file(): raise FileNotFoundError(f"Letter base-model checkpoint is missing: {target}")
    payload=torch.load(target,map_location="cpu",weights_only=True)
    if not isinstance(payload,dict): raise ValueError("Letter base-model checkpoint must be a dictionary")
    if payload.get("model_version")!=MODEL_VERSION: raise ValueError("Unsupported letter base-model version")
    if payload.get("input_dim")!=INPUT_DIM or payload.get("embedding_dim")!=EMBEDDING_DIM or payload.get("num_classes")!=NUM_CLASSES: raise ValueError("Letter base-model contract is incompatible")
    if tuple(payload.get("labels",()))!=LETTER_LABELS: raise ValueError("Letter base-model labels are incompatible")
    state=payload.get("state_dict")
    if not isinstance(state,dict): raise ValueError("Letter base-model state_dict is missing")
    model=VisionBridgeLetterBaseModel(); model.load_state_dict(state,strict=True); model.eval(); return model

def checkpoint_status(path:str|Path)->dict[str,str|bool]:
    try:
        load_checkpoint(path)
        return {"available":True,"status":"ready","modality":"hand-only letter base + few-shot adapter","model_version":MODEL_VERSION}
    except FileNotFoundError:
        return {"available":False,"status":"letter_base_model_missing","modality":"hand-only letter base + few-shot adapter"}
    except (OSError,RuntimeError,ValueError,EOFError,pickle.UnpicklingError,UnicodeError):
        return {"available":False,"status":"letter_base_model_invalid","modality":"hand-only letter base + few-shot adapter"}
