import numpy as np
import pytest
import torch

from app.models.letter_model import VisionBridgeLetterBaseModel, save_checkpoint, load_checkpoint
from app.services import letter_fewshot

def _pair(seed):
    rng=np.random.default_rng(seed); points=rng.normal(0,0.02,(21,3)).astype(np.float32); points[0]=0
    return points.reshape(-1).tolist()+points.reshape(-1).tolist()

def test_base_model_contract():
    model=VisionBridgeLetterBaseModel(); x=torch.randn(4,126); logits=model(x); emb=model.embed(x)
    assert logits.shape==(4,26); assert emb.shape==(4,64); assert torch.isfinite(logits).all(); assert torch.isfinite(emb).all()

def test_checkpoint_round_trip(tmp_path):
    path=tmp_path/"base.pt"; model=VisionBridgeLetterBaseModel(); save_checkpoint(model,path); loaded=load_checkpoint(path)
    assert loaded.input_dim==126 and loaded.embedding_dim==64 and loaded.output_head.out_features==26

def test_few_shot_adapter_binds_to_base_checkpoint(tmp_path,monkeypatch):
    model=VisionBridgeLetterBaseModel(); base=tmp_path/"base.pt"; save_checkpoint(model,base)
    monkeypatch.setattr(letter_fewshot.settings,"LETTER_BASE_MODEL_PATH",str(base))
    monkeypatch.setattr(letter_fewshot.settings,"ADAPTER_WEIGHTS_DIR",str(tmp_path/"adapters"))
    fitted=letter_fewshot.fit_prototype_adapter(model,[("A",_pair(1)),("A",_pair(1)),("B",_pair(2)),("B",_pair(2))])
    adapter_path=letter_fewshot.save_prototype_adapter(fitted["payload"])
    loaded=letter_fewshot.load_prototype_adapter(adapter_path,base)
    pred,conf,scores=letter_fewshot.predict_letter(model,loaded,_pair(1))
    assert pred=="A"; assert 0<conf<=1; assert scores[0][0]=="A"

def test_adapter_rejects_changed_base(tmp_path,monkeypatch):
    model=VisionBridgeLetterBaseModel(); a=tmp_path/"a.pt"; b=tmp_path/"b.pt"; save_checkpoint(model,a); save_checkpoint(VisionBridgeLetterBaseModel(),b)
    monkeypatch.setattr(letter_fewshot.settings,"LETTER_BASE_MODEL_PATH",str(a)); monkeypatch.setattr(letter_fewshot.settings,"ADAPTER_WEIGHTS_DIR",str(tmp_path/"adapters"))
    fitted=letter_fewshot.fit_prototype_adapter(model,[("A",_pair(1)),("B",_pair(2))]); p=letter_fewshot.save_prototype_adapter(fitted["payload"])
    with pytest.raises(ValueError,match="different base-model"): letter_fewshot.load_prototype_adapter(p,b)

def test_degenerate_input_rejected():
    with pytest.raises(ValueError,match="No visible"): letter_fewshot.normalize_hand_pair([0.0]*126)
