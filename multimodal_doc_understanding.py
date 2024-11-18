"""
Multimodal Document Understanding — combines vision and language models
for structured information extraction from scanned documents and PDFs.
"""

import torch
from dataclasses import dataclass, field
from PIL import Image
from transformers import (
    LayoutLMv3Processor,
    LayoutLMv3ForTokenClassification,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
)


@dataclass
class ExtractedField:
    label: str
    value: str
    confidence: float
    bbox: list[int] = field(default_factory=list)


@dataclass
class DocumentUnderstandingResult:
    doc_id: str
    fields: list[ExtractedField]
    summary: str
    language: str


class MultimodalDocUnderstanding:
    """
    Pipeline combining LayoutLMv3 (vision+layout+text) for field extraction
    and a seq2seq model for document summarization.
    """

    def __init__(
        self,
        extraction_model: str = "microsoft/layoutlmv3-base",
        summarization_model: str = "facebook/bart-large-cnn",
    ):
        self.processor = LayoutLMv3Processor.from_pretrained(extraction_model)
        self.extractor = LayoutLMv3ForTokenClassification.from_pretrained(extraction_model)
        self.extractor.eval()

        self.sum_tokenizer = AutoTokenizer.from_pretrained(summarization_model)
        self.summarizer = AutoModelForSeq2SeqLM.from_pretrained(summarization_model)
        self.summarizer.eval()

    def extract_fields(self, image: Image.Image, words: list[str], boxes: list[list[int]]) -> list[ExtractedField]:
        encoding = self.processor(
            image,
            words,
            boxes=boxes,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        with torch.no_grad():
            outputs = self.extractor(**encoding)
        probs = torch.softmax(outputs.logits, dim=-1)
        pred_ids = probs.argmax(dim=-1)[0].tolist()
        scores = probs.max(dim=-1).values[0].tolist()

        id2label = self.extractor.config.id2label
        fields = []
        for word, pred_id, score, box in zip(words, pred_ids[1:-1], scores[1:-1], boxes):
            label = id2label.get(pred_id, "O")
            if label != "O" and score > 0.8:
                fields.append(ExtractedField(
                    label=label,
                    value=word,
                    confidence=round(score, 3),
                    bbox=box,
                ))
        return fields

    def summarize(self, text: str, max_length: int = 128) -> str:
        inputs = self.sum_tokenizer(
            text, return_tensors="pt", truncation=True, max_length=1024
        )
        with torch.no_grad():
            summary_ids = self.summarizer.generate(
                inputs["input_ids"],
                max_length=max_length,
                num_beams=4,
                early_stopping=True,
            )
        return self.sum_tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    def process(self, doc_id: str, image: Image.Image, words: list[str], boxes: list[list[int]], raw_text: str) -> DocumentUnderstandingResult:
        fields = self.extract_fields(image, words, boxes)
        summary = self.summarize(raw_text)
        return DocumentUnderstandingResult(
            doc_id=doc_id,
            fields=fields,
            summary=summary,
            language="en",
        )
