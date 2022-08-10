"""
NLP Text Classifier — multi-label intent and topic classification
for conversational AI and document understanding pipelines.
"""

from dataclasses import dataclass
from typing import Optional
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


@dataclass
class ClassificationResult:
    text: str
    labels: list[str]
    scores: list[float]
    top_label: str
    top_score: float


class NLPTextClassifier:
    """
    Fine-tunable multi-label text classifier based on transformer encoders.
    Supports intent detection, topic labeling, and sentiment classification.
    """

    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        labels: Optional[list[str]] = None,
        threshold: float = 0.5,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()
        self.labels = labels or []
        self.threshold = threshold

    def predict(self, text: str) -> ClassificationResult:
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=512, padding=True
        )
        with torch.no_grad():
            logits = self.model(**inputs).logits
        probs = torch.sigmoid(logits)[0].tolist()

        if self.labels:
            label_score_pairs = list(zip(self.labels, probs))
        else:
            label_score_pairs = [(str(i), p) for i, p in enumerate(probs)]

        active = [
            (lbl, round(score, 3))
            for lbl, score in label_score_pairs
            if score >= self.threshold
        ]
        active.sort(key=lambda x: x[1], reverse=True)

        top_label, top_score = active[0] if active else ("unknown", 0.0)

        return ClassificationResult(
            text=text,
            labels=[lbl for lbl, _ in active],
            scores=[score for _, score in active],
            top_label=top_label,
            top_score=top_score,
        )

    def batch_predict(self, texts: list[str]) -> list[ClassificationResult]:
        return [self.predict(t) for t in texts]

    def set_threshold(self, threshold: float):
        self.threshold = max(0.0, min(1.0, threshold))
