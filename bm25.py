from __future__ import annotations

import math
import re
from collections import Counter


TOKEN_RE = re.compile(r"[가-힣]+|[A-Za-z0-9]+")


def tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    for match in TOKEN_RE.finditer(text.lower()):
        token = match.group(0)
        tokens.append(token)
        if re.fullmatch(r"[가-힣]+", token) and len(token) > 1:
            tokens.extend(token)
            tokens.extend(token[i : i + 2] for i in range(len(token) - 1))
    return tokens


class BM25Index:
    def __init__(self, documents: list[dict], k1: float = 1.5, b: float = 0.75):
        self.documents = documents
        self.k1 = k1
        self.b = b
        self.doc_tokens = [tokenize(doc["text"]) for doc in documents]
        self.doc_lengths = [len(tokens) for tokens in self.doc_tokens]
        self.avg_doc_length = (
            sum(self.doc_lengths) / len(self.doc_lengths) if self.doc_lengths else 0.0
        )
        self.term_freqs = [Counter(tokens) for tokens in self.doc_tokens]
        self.doc_freqs: Counter[str] = Counter()
        for tokens in self.doc_tokens:
            self.doc_freqs.update(set(tokens))

    def search(self, query: str, top_k: int = 20) -> list[dict]:
        query_terms = list(dict.fromkeys(tokenize(query)))
        if not query_terms or not self.documents:
            return []

        scores = []
        total_docs = len(self.documents)
        for index, doc in enumerate(self.documents):
            score = 0.0
            doc_length = self.doc_lengths[index]
            term_freq = self.term_freqs[index]

            for term in query_terms:
                freq = term_freq.get(term, 0)
                if freq == 0:
                    continue
                df = self.doc_freqs.get(term, 0)
                idf = math.log(1 + (total_docs - df + 0.5) / (df + 0.5))
                denom = freq + self.k1 * (
                    1 - self.b + self.b * doc_length / max(self.avg_doc_length, 1e-9)
                )
                score += idf * (freq * (self.k1 + 1)) / denom

            if score > 0:
                scores.append(
                    {
                        "id": doc["id"],
                        "score": score,
                        "payload": doc["payload"],
                    }
                )

        scores.sort(key=lambda item: item["score"], reverse=True)
        return scores[:top_k]
