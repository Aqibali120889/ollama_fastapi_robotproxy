import math
from collections import Counter, defaultdict
from typing import Dict, List, Optional


def _tokenize(text: str) -> List[str]:
    cleaned = "".join(ch.lower() if ch.isalnum() else " " for ch in text)
    return [token for token in cleaned.split() if token]


def _metadata_match(metadata: Dict[str, object], filters: Optional[Dict[str, str]]) -> bool:
    if not filters:
        return True
    for key, value in filters.items():
        if str(metadata.get(key, "")).lower() != value.lower():
            return False
    return True


def _keyword_scores(query: str, docs: List[str]) -> Dict[int, float]:
    query_terms = _tokenize(query)
    if not query_terms or not docs:
        return {}

    doc_tokens = [_tokenize(doc) for doc in docs]
    term_doc_freq = Counter()
    for tokens in doc_tokens:
        term_doc_freq.update(set(tokens))

    total_docs = len(docs)
    avg_doc_len = sum(len(tokens) for tokens in doc_tokens) / max(total_docs, 1)

    scores: Dict[int, float] = {}
    k1 = 1.5
    b = 0.75
    for i, tokens in enumerate(doc_tokens):
        if not tokens:
            continue
        token_counts = Counter(tokens)
        doc_len = len(tokens)
        score = 0.0
        for term in query_terms:
            if term not in token_counts:
                continue
            df = term_doc_freq.get(term, 0)
            idf = math.log(1 + (total_docs - df + 0.5) / (df + 0.5))
            tf = token_counts[term]
            denom = tf + k1 * (1 - b + b * (doc_len / max(avg_doc_len, 1)))
            score += idf * ((tf * (k1 + 1)) / max(denom, 1e-9))
        if score > 0:
            scores[i] = score
    return scores


def hybrid_retrieve(
    collection,
    query_embedding: List[float],
    query_text: str,
    top_k: int,
    dense_weight: float,
    keyword_weight: float,
    candidate_pool_size: int,
    min_fusion_score: float,
    metadata_filters: Optional[Dict[str, str]] = None,
) -> Dict[str, List[object]]:
    dense_results = collection.query(
        query_embeddings=[query_embedding],
        n_results=max(top_k, candidate_pool_size),
        where=metadata_filters or None,
    )

    dense_docs = dense_results.get("documents", [[]])[0] if dense_results.get("documents") else []
    dense_metadatas = dense_results.get("metadatas", [[]])[0] if dense_results.get("metadatas") else []
    dense_distances = dense_results.get("distances", [[]])[0] if dense_results.get("distances") else []

    all_docs_payload = collection.get(include=["documents", "metadatas"])
    all_docs = all_docs_payload.get("documents") or []
    all_metas = all_docs_payload.get("metadatas") or []

    keyword_candidates: List[tuple[str, Dict[str, object], float]] = []
    raw_keyword_scores = _keyword_scores(query_text, all_docs)
    if raw_keyword_scores:
        max_kw = max(raw_keyword_scores.values())
        for i, score in raw_keyword_scores.items():
            meta = all_metas[i] if i < len(all_metas) else {}
            if _metadata_match(meta or {}, metadata_filters):
                keyword_candidates.append((all_docs[i], meta or {}, score / max(max_kw, 1e-9)))

    fused = defaultdict(lambda: {"metadata": {}, "distance": float("inf"), "dense": 0.0, "keyword": 0.0})

    for doc, meta, distance in zip(dense_docs, dense_metadatas, dense_distances):
        item = fused[doc]
        item["metadata"] = meta or {}
        item["distance"] = distance
        item["dense"] = max(item["dense"], 1.0 / (1.0 + max(distance, 0.0)))

    for doc, meta, normalized_kw in keyword_candidates:
        item = fused[doc]
        item["metadata"] = item["metadata"] or meta
        item["keyword"] = max(item["keyword"], normalized_kw)

    fused_items = []
    normalized_dense_weight = dense_weight / max(dense_weight + keyword_weight, 1e-9)
    normalized_keyword_weight = keyword_weight / max(dense_weight + keyword_weight, 1e-9)

    for doc, item in fused.items():
        score = (normalized_dense_weight * item["dense"]) + (normalized_keyword_weight * item["keyword"])
        if score >= min_fusion_score:
            fused_items.append(
                {
                    "document": doc,
                    "metadata": item["metadata"],
                    "distance": item["distance"],
                    "fusion_score": score,
                }
            )

    fused_items.sort(key=lambda row: row["fusion_score"], reverse=True)
    selected = fused_items[:top_k]

    return {
        "documents": [item["document"] for item in selected],
        "metadatas": [item["metadata"] for item in selected],
        "distances": [float(item["distance"]) for item in selected],
        "fusion_scores": [round(float(item["fusion_score"]), 6) for item in selected],
    }
