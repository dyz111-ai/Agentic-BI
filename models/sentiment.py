from __future__ import annotations

import re
from collections import Counter
import pandas as pd

POSITIVE_WORDS = {
    "bom", "boa", "excelente", "rapida", "rapido", "qualidade", "gostei", "otimo", "ótimo", "perfeito",
    "good", "great", "excellent", "fast", "quality", "perfect"
}
NEGATIVE_WORDS = {
    "ruim", "atrasada", "atrasado", "demorou", "caro", "danificada", "problema", "pessimo", "péssimo",
    "bad", "delay", "late", "expensive", "broken", "problem", "poor"
}
STOPWORDS = {
    "de", "do", "da", "e", "a", "o", "um", "uma", "para", "por", "com", "que", "no", "na", "em",
    "the", "and", "or", "to", "of", "is", "was", "very", "it"
}


def simple_sentiment(text: str, review_score: float | None = None) -> float:
    text = (text or "").lower()
    tokens = re.findall(r"[a-zA-ZÀ-ÿ]+", text)
    pos = sum(t in POSITIVE_WORDS for t in tokens)
    neg = sum(t in NEGATIVE_WORDS for t in tokens)
    lex = (pos - neg) / max(len(tokens), 1)
    if review_score is not None and pd.notna(review_score):
        score_part = (float(review_score) - 3.0) / 2.0
        return round(0.7 * score_part + 0.3 * lex, 4)
    return round(lex, 4)


def extract_keywords(texts, top_n: int = 20):
    counter = Counter()
    for text in texts:
        tokens = re.findall(r"[a-zA-ZÀ-ÿ]+", str(text).lower())
        tokens = [t for t in tokens if len(t) > 2 and t not in STOPWORDS]
        counter.update(tokens)
    return counter.most_common(top_n)


def enrich_reviews_with_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    msg = out.get("review_comment_message", pd.Series([""] * len(out))).fillna("")
    title = out.get("review_comment_title", pd.Series([""] * len(out))).fillna("")
    text = title + " " + msg
    scores = out.get("review_score", pd.Series([None] * len(out)))
    out["sentiment_score"] = [simple_sentiment(t, s) for t, s in zip(text, scores)]
    out["sentiment_label"] = pd.cut(out["sentiment_score"], [-2, -0.25, 0.25, 2], labels=["negative", "neutral", "positive"])
    return out
