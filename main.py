from fastapi import FastAPI
from pydantic import BaseModel
import torch
import torch.nn as nn
import numpy as np

from utils.preprocessing import clean_sequence
from utils.features import compute_gc, detect_replicon, kmer_features

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

INPUT_DIM = 516

class HGTModel(nn.Module):
    def __init__(self, input_dim: int = INPUT_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)

def load_model(path: str) -> nn.Module:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, nn.Module):
        checkpoint.eval()
        return checkpoint
    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        model = HGTModel(input_dim=INPUT_DIM)
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        return model
    raise ValueError()

model = load_model("model/best_model.pt")

class Patient(BaseModel):
    name: str = ""
    organism: str = ""
    resistance: str = ""
    sample: str = ""
    date: str = ""
    sequence: str

class RequestData(BaseModel):
    patient_A: Patient
    patient_B: Patient

REP_MAP = {"IncF": 1, "IncI": 2, "IncN": 3, "Unknown": 0}

def build_features(p: Patient):
    seq = clean_sequence(p.sequence)
    gc = compute_gc(seq)
    rep = detect_replicon(seq)
    rep_val = REP_MAP.get(rep, 0)
    kmer = kmer_features(seq)
    features = np.array(kmer + [gc, rep_val], dtype=np.float32)
    return features, gc, rep

@app.post("/predict")
def predict(data: RequestData):
    fA, gcA, repA = build_features(data.patient_A)
    fB, gcB, repB = build_features(data.patient_B)

    x = np.concatenate([fA, fB])
    x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        score = torch.sigmoid(model(x_tensor)).item()

    if score > 0.8:
        level = "High"
    elif score > 0.5:
        level = "Medium"
    else:
        level = "Low"

    return {
        "risk_score": round(score * 100),
        "risk_level": level,
        "patient_info": {
            "A": data.patient_A.name,
            "B": data.patient_B.name,
        },
        "features": {
            "gc_A": round(gcA * 100, 1),
            "rep_A": repA,
            "gc_B": round(gcB * 100, 1),
            "rep_B": repB,
        },
    }