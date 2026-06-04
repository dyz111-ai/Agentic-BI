from __future__ import annotations

import re
from collections import Counter
import pandas as pd

# Olist 评论以葡语为主，兼顾少量英语；均为小写以便与 token 匹配
POSITIVE_WORDS = {
    # 葡语 — 总体评价
    "bom", "boa", "bons", "boas", "excelente", "excelentes", "otimo", "ótimo", "otima", "ótima",
    "perfeito", "perfeita", "maravilhoso", "maravilhosa", "maravilha", "incrivel", "incrível",
    "fantastico", "fantástico", "fantastica", "fantástica", "impecavel", "impecável", "legal", "show", "top",
    # 葡语 — 满意 / 推荐
    "gostei", "adorei", "amei", "recomendo", "recomendado", "recomendada", "satisfeito", "satisfeita",
    "satisfacao", "satisfação", "feliz", "contente", "melhor", "superou", "expectativas",
    # 葡语 — 质量 / 外观
    "qualidade", "bonito", "bonita", "lindo", "linda", "lindos", "lindas", "original", "novo", "nova",
    "funciona", "funcionou", "util", "útil", "pratico", "prático", "confiavel", "confiável", "correto", "correta",
    # 葡语 — 物流 / 服务
    "rapido", "rápido", "rapida", "rápida", "pontual", "eficiente", "atencioso", "atenciosa", "educado", "educada",
    "entregue", "chegou", "embalagem", "bem", "embalado", "embalada",
    # 英语
    "good", "great", "excellent", "perfect", "amazing", "awesome", "wonderful", "fantastic", "superb",
    "outstanding", "flawless", "impressed", "love", "loved", "recommend", "recommended", "satisfied",
    "happy", "beautiful", "nice", "pretty", "works", "working", "functional", "genuine", "best", "better",
    "fast", "quick", "speedy", "prompt", "quality", "reliable", "durable", "sturdy", "worth", "helpful",
    "friendly", "smooth", "easy",
}
NEGATIVE_WORDS = {
    # 葡语 — 总体差评
    "ruim", "pessimo", "péssimo", "pessima", "péssima", "horrivel", "horrível", "terrivel", "terrível",
    "decepcionado", "decepcionada", "decepcionante", "decepcao", "decepção", "insatisfeito", "insatisfeita",
    "pior", "horror", "porcaria", "lixo", "mal",
    # 葡语 — 物流延迟
    "atrasado", "atrasada", "atraso", "atrasou", "demorou", "demora", "demorado", "demorada", "lento", "lenta",
    # 葡语 — 价格
    "caro", "cara", "carissimo", "caríssimo", "carissima", "caríssima",
    # 葡语 — 损坏 / 质量
    "quebrado", "quebrada", "quebrou", "danificado", "danificada", "danificou", "defeito", "defeituoso",
    "defeituosa", "estragado", "estragada", "avariado", "avariada", "rachado", "rachada", "amassado", "amassada",
    # 葡语 — 问题 / 错误
    "problema", "problemas", "errado", "errada", "erro", "falha", "falhou", "faltando", "incompleto", "incompleta",
    "diferente", "enganoso", "enganosa", "enganacao", "enganação", "falso", "falsa", "falsificado", "falsificada",
    # 葡语 — 售后 / 未收到
    "reembolso", "devolucao", "devolução", "cancelado", "cancelada", "perdido", "perdida", "roubado", "roubada",
    "nunca", "ausente", "sujo", "suja", "usado", "usada",
    # 英语
    "bad", "worst", "terrible", "awful", "horrible", "dreadful", "pathetic", "disappointing", "disappointed",
    "disappointment", "poor", "hate", "hated", "nightmare", "garbage", "trash", "junk", "useless", "wasted",
    "delay", "delayed", "late", "slow", "sluggish", "expensive", "overpriced", "costly", "cheap", "flimsy",
    "broken", "damaged", "defective", "faulty", "cracked", "shattered", "scratched", "dented", "leaking",
    "problem", "problems", "issue", "issues", "flaw", "flaws", "wrong", "incorrect", "inaccurate",
    "missing", "incomplete", "fake", "counterfeit", "scam", "fraud", "refund", "return", "fail", "failed", "ripped",
}
STOPWORDS = {
    # 葡语虚词
    "de", "do", "da", "dos", "das", "e", "a", "o", "os", "as", "um", "uma", "uns", "umas",
    "para", "por", "com", "que", "no", "na", "nos", "nas", "em", "ao", "aos", "à", "se", "me", "te", "lhe",
    "ele", "ela", "eles", "elas", "eu", "voce", "você", "nós", "nos", "muito", "muita", "mais", "menos",
    "como", "mas", "so", "só", "ja", "já", "nao", "não", "sim", "este", "esta", "esse", "essa", "isso", "aquilo",
    "produto", "produtos", "pedido", "compra", "comprei", "recebi", "loja", "vendedor", "cliente", "item",
    # 英语虚词
    "the", "and", "or", "to", "of", "is", "was", "are", "were", "be", "been", "it", "its", "this", "that",
    "very", "just", "also", "not", "but", "for", "with", "from", "have", "has", "had", "my", "your", "their",
    "product", "order", "item", "seller", "store", "buy", "bought", "received",
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
