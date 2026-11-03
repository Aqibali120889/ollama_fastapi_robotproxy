import math
from collections import Counter
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
    """BM25-style keyword scores over the supplied corpus.

    This deliberately remains a simple in-memory implementation for this patch.
    A persistent, incrementally refreshed BM25 index is the next retrieval task.
    """
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
    """Retrieve from dense + keyword channels and fuse their rankings with RRF.

    fusion_scores are normalized RRF ranking scores (0..1), not probabilities.
    Distances remain the Chroma dense distances; keyword-only hits use infinity.
    """
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

    raw_keyword_scores = _keyword_scores(query_text, all_docs)
    keyword_rows = []
    for index, score in raw_keyword_scores.items():
        metadata = all_metas[index] if index < len(all_metas) else {}
        metadata = metadata or {}
        if _metadata_match(metadata, metadata_filters):
            keyword_rows.append((all_docs[index], metadata, score))

    keyword_rows.sort(key=lambda row: row[2], reverse=True)
    keyword_rows = keyword_rows[:max(top_k, candidate_pool_size)]
    max_keyword_score = max((row[2] for row in keyword_rows), default=0.0)

    # Keep one record per document text. Chroma's current ingestion pipeline
    # stores document text as the shared key available to both retrieval paths.
    fused: Dict[str, Dict[str, object]] = {}

    for rank, (doc, metadata, distance) in enumerate(
        zip(dense_docs, dense_metadatas, dense_distances), start=1
    ):
        item = fused.setdefault(
            doc,
            {
                "metadata": metadata or {},
                "distance": float("inf"),
                "dense_rank": None,
                "keyword_rank": None,
                "keyword_score": 0.0,
                "rrf_score": 0.0,
                "matched_by": [],
            },
        )
        item["metadata"] = item["metadata"] or metadata or {}
        item["distance"] = float(distance)
        item["dense_rank"] = rank
        item["matched_by"].append("dense")

    for rank, (doc, metadata, raw_score) in enumerate(keyword_rows, start=1):
        item = fused.setdefault(
            doc,
            {
                "metadata": metadata or {},
                "distance": float("inf"),
                "dense_rank": None,
                "keyword_rank": None,
                "keyword_score": 0.0,
                "rrf_score": 0.0,
                "matched_by": [],
            },
        )
        item["metadata"] = item["metadata"] or metadata or {}
        item["keyword_rank"] = rank
        item["keyword_score"] = raw_score / max(max_keyword_score, 1e-9)
        item["matched_by"].append("keyword")

    total_weight = max(dense_weight + keyword_weight, 1e-9)
    dense_weight = dense_weight / total_weight
    keyword_weight = keyword_weight / total_weight
    rrf_constant = 60
    ideal_score = (dense_weight + keyword_weight) / (rrf_constant + 1)

    fused_items = []
    for doc, item in fused.items():
        raw_rrf = 0.0
        if item["dense_rank"] is not None:
            raw_rrf += dense_weight / (rrf_constant + int(item["dense_rank"]))
        if item["keyword_rank"] is not None:
            raw_rrf += keyword_weight / (rrf_constant + int(item["keyword_rank"]))

        normalized_rrf = raw_rrf / max(ideal_score, 1e-12)
        if normalized_rrf < min_fusion_score:
            continue

        fused_items.append(
            {
                "document": doc,
                "metadata": item["metadata"],
                "distance": item["distance"],
                "fusion_score": normalized_rrf,
                "keyword_score": item["keyword_score"],
                "matched_by": item["matched_by"],
                "dense_rank": item["dense_rank"],
                "keyword_rank": item["keyword_rank"],
            }
        )

    fused_items.sort(key=lambda row: row["fusion_score"], reverse=True)
    selected = fused_items[:top_k]

    return {
        "documents": [item["document"] for item in selected],
        "metadatas": [item["metadata"] for item in selected],
        "distances": [float(item["distance"]) for item in selected],
        "fusion_scores": [round(float(item["fusion_score"]), 6) for item in selected],
        "keyword_scores": [round(float(item["keyword_score"]), 6) for item in selected],
        "matched_by": [item["matched_by"] for item in selected],
        "dense_ranks": [item["dense_rank"] for item in selected],
        "keyword_ranks": [item["keyword_rank"] for item in selected],
    }
