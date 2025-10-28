# bge_utils.py
from torch import nn
import torch


BGE_QUERY_PROMPT = "Represent this sentence for searching relevant passages: "

def is_bge_model(name: str) -> bool:
    res = "bge-" in (name or "").lower()
    return res

class BGELoss(nn.Module):
    """InfoNCE-style loss with temperature scaling, like BGE pretraining."""
    def __init__(self, model, temperature: float = 0.02):
        super().__init__()
        self.model = model
        self.temperature = temperature
        self.ce = nn.CrossEntropyLoss()

    def forward(self, sentence_features, labels=None):
        # Expect [query_features, passage_features]
        q = self.model(sentence_features[0])["sentence_embedding"]
        p = self.model(sentence_features[1])["sentence_embedding"]

        # Normalize embeddings for cosine similarity
        q = nn.functional.normalize(q, dim=-1)
        p = nn.functional.normalize(p, dim=-1)

        # Compute logits and apply temperature scaling
        logits = torch.matmul(q, p.T) / self.temperature
        targets = torch.arange(logits.size(0), device=logits.device)
        return self.ce(logits, targets)