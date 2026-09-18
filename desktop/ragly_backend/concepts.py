"""Local image understanding: what is in a picture, in words, with no cloud call.

The vision model already in the project (CLIP ViT-B/32, ONNX, CPU) shares one embedding
space between its image encoder and its text encoder. That is what makes this possible
without a second model: each concept below is encoded once as a sentence ("a photo of a
car"), and an image's own vector is compared against all of them. The concepts that score
well are what the picture contains, and the description is written from them.

This runs ONCE per image, at indexing time, and the result is stored. Searching never runs
a vision model over the corpus; it compares a query against what was already understood.

Nothing here maps a query word to a picture. The vocabulary describes the world, not the
test questions, and an image earns a concept only from the model's own similarity.
"""
from __future__ import annotations

import logging
import re

import numpy as np

log = logging.getLogger("ragly.concepts")

#: Everyday things that turn up in documents people index. Grouped only for readability -
#: the model decides which apply. Add to this list to widen coverage; nothing is hard-coded
#: to a query, and removing a word only makes the system blinder, never wrong.
VOCABULARY: dict[str, tuple[str, ...]] = {
    "vehicles": ("car", "truck", "bus", "motorcycle", "bicycle", "van", "tractor", "aeroplane",
                 "boat", "train", "wheel", "tyre", "number plate", "windscreen", "car bumper",
                 "damaged car", "road", "traffic", "auto rickshaw", "scooter", "lorry", "sedan", "suv"),
    "workplace": ("laptop", "computer monitor", "keyboard", "mouse", "mobile phone", "desk",
                  "office chair", "printer", "server rack", "cable", "workstation", "whiteboard",
                  "notebook", "pen", "coffee mug", "water bottle", "headphones", "camera",
                  "desktop computer", "tablet", "projector", "filing cabinet", "bookshelf", "meeting room"),
    "documents": ("invoice", "receipt", "bank statement", "contract", "form", "certificate",
                  "identity card", "passport", "cheque", "spreadsheet", "printed table",
                  "handwritten note", "signature", "stamp", "letterhead", "barcode", "qr code",
                  "tax invoice", "purchase order", "shipping label", "medical report", "lab report",
                  "agreement", "affidavit", "bill of lading", "driving licence", "pan card", "aadhaar card"),
    "people": ("person", "group of people", "portrait photograph", "hand", "face", "crowd",
               "worker wearing a helmet", "engineer", "doctor", "student", "technician", "patient", "nurse"),
    "places": ("building", "office interior", "factory", "warehouse", "construction site",
               "shop front", "kitchen", "classroom", "hospital", "street", "parking area",
               "laboratory", "clinic", "server room", "reception area"),
    "equipment": ("electrical panel", "switchgear", "transformer", "circuit breaker", "meter",
                  "pipe", "valve", "motor", "generator", "solar panel", "machine", "tool",
                  "screw", "bolt", "wire", "pump", "engine", "circuit board", "battery", "gauge", "pressure meter"),
    "charts": ("bar chart", "line graph", "pie chart", "flow diagram", "map", "floor plan",
               "circuit diagram", "screenshot", "logo", "signature scan", "flowchart", "schematic",
               "scatter plot", "histogram", "infographic", "table of figures"),
    "nature": ("plant", "tree", "flower", "leaf", "garden", "soil", "animal", "dog", "cat",
               "bird", "food", "fruit", "vegetable", "sky", "water", "landscape", "sunlight"),
}

#: Words that mean the same thing to a person searching. Used only to widen a QUERY, never
#: to award a concept to an image.
SYNONYMS: dict[str, tuple[str, ...]] = {
    "vehicle": ("car", "truck", "bus", "van", "motorcycle", "tractor", "automobile", "scooter"),
    "automobile": ("car", "truck", "van", "sedan", "suv"),
    "bike": ("motorcycle", "bicycle", "scooter"),
    "cellphone": ("mobile phone",), "smartphone": ("mobile phone",), "phone": ("mobile phone",),
    "computer": ("laptop", "computer monitor", "desktop computer", "workstation"),
    "pc": ("laptop", "computer monitor", "desktop computer"),
    "screen": ("computer monitor", "display screen"), "display": ("computer monitor",),
    "mug": ("coffee mug",), "cup": ("coffee mug",),
    "bill": ("invoice", "receipt", "tax invoice"), "id": ("identity card",), "id card": ("identity card",),
    "badge": ("identity card",), "employee id": ("identity card",), "pass": ("identity card",),
    "statement": ("bank statement",), "chart": ("bar chart", "line graph", "pie chart", "flow diagram"),
    "graph": ("line graph", "bar chart", "scatter plot"), "diagram": ("flow diagram", "circuit diagram", "schematic", "flowchart"),
    "workstation": ("laptop", "desk", "computer monitor", "office chair"),
    "office": ("office interior", "desk", "workstation"), "machinery": ("machine", "motor", "generator", "engine"),
    "report": ("lab report", "medical report", "spreadsheet"), "doctor": ("doctor", "hospital", "clinic"),
    "worker": ("worker wearing a helmet", "engineer", "technician"), "man": ("person", "portrait photograph"),
    "woman": ("person", "portrait photograph"), "sign": ("signature", "signature scan"),
}

PROMPT = "a photo of {}"
#: A concept is only claimed when it clearly beats the field, not merely when it ranks first.
MARGIN = 0.035
MAX_CONCEPTS = 6


def all_concepts() -> list[str]:
    return [c for group in VOCABULARY.values() for c in group]


def _article(word: str) -> str:
    return f"an {word}" if word[:1] in "aeiou" else f"a {word}"


def clean_image_query(text: str) -> str:
    """Strip conversational filler from search query to focus on the semantic visual subject."""
    q = (text or "").strip()
    patterns = [
        r"^(?:please\s+)?(?:show|find|search|get|list|display|give)(?:\s+me)?\s+(?:all\s+)?(?:the\s+)?(?:images?|photos?|pictures?)\s+(?:of|with|containing|showing|having|about)?\s*",
        r"^(?:find|show|search)\s+(?:an?\s+)?(?:image|photo|picture)\s+(?:of|with|containing|showing|having)?\s*",
        r"^(?:images?|photos?|pictures?)\s+(?:of|with|containing|showing|having|about)\s*",
        r"^(?:where\s+is\s+the\s+)?(?:image|photo|picture)\s+(?:of|with)?\s*",
    ]
    for p in patterns:
        q = re.sub(p, "", q, flags=re.IGNORECASE).strip()
    return q or text.strip()


class ConceptTagger:
    """Scores an image vector against the vocabulary. Text prompts are embedded once."""

    def __init__(self, embedder):
        self.embedder = embedder
        self._matrix: np.ndarray | None = None
        self._labels: list[str] = []

    def ready(self) -> bool:
        return bool(self.embedder) and self.embedder.available()

    def _prompts(self) -> np.ndarray:
        if self._matrix is None:
            labels = all_concepts()
            vectors = []
            for label in labels:
                vectors.append(self.embedder.embed_text(PROMPT.format(_article(label))))
            self._labels = labels
            self._matrix = np.vstack(vectors).astype(np.float32)
            log.info("concept vocabulary embedded: %d concepts", len(labels))
        return self._matrix

    def tag(self, image_vector: np.ndarray) -> tuple[list[str], str, float]:
        """Returns (objects, description, best_score) for one image vector.

        ``best_score`` is how strongly the winning concept describes this picture. It is the
        yardstick a later text query is measured against: a query that describes the image less
        well than its own best word is not describing it at all.
        """
        if not self.ready():
            return [], "", 0.0
        try:
            matrix = self._prompts()
        except Exception as exc:
            log.warning("concept vocabulary unavailable: %s", exc)
            return [], "", 0.0
        vec = np.asarray(image_vector, dtype=np.float32).reshape(-1)
        norm = np.linalg.norm(vec) or 1.0
        scores = matrix @ (vec / norm)
        order = np.argsort(-scores)
        best = float(scores[order[0]])
        floor = max(best - MARGIN, float(np.mean(scores)) + 2 * float(np.std(scores)))
        objects = [self._labels[i] for i in order[:MAX_CONCEPTS] if float(scores[i]) >= floor]
        return objects, describe(objects), best


def describe(objects: list[str]) -> str:
    """A sentence a person can read, written from the concepts the model found."""
    if not objects:
        return ""
    if len(objects) == 1:
        return f"An image showing {_article(objects[0])}."
    return f"An image showing {', '.join(objects[:-1])} and {objects[-1]}."


def expand_query(question: str) -> list[str]:
    """Concept words a query is asking about, including the obvious synonyms."""
    clean = clean_image_query(question)
    words = re.findall(r"[a-z0-9][a-z0-9 \-]{1,}", (clean or question or "").lower())
    text = " " + " ".join(words) + " "
    wanted: list[str] = []
    for concept in all_concepts():
        if re.search(rf"\b{re.escape(concept)}\b", text):
            wanted.append(concept)
    for term, alts in SYNONYMS.items():
        if re.search(rf"\b{re.escape(term)}\b", text):
            wanted.extend(alts)
    seen: list[str] = []
    for c in wanted:
        if c not in seen:
            seen.append(c)
    return seen
