from app.hybrid_retriever import hybrid_retrieve


class FakeCollection:
    def __init__(self, dense_doc, dense_meta, dense_distance, docs, metadatas):
        self.dense_doc = dense_doc
        self.dense_meta = dense_meta
        self.dense_distance = dense_distance
        self.docs = docs
        self.metadatas = metadatas

    def query(self, **kwargs):
        return {
            "documents": [[self.dense_doc]],
            "metadatas": [[self.dense_meta]],
            "distances": [[self.dense_distance]],
        }

    def get(self, **kwargs):
        return {"documents": self.docs, "metadatas": self.metadatas}


def test_keyword_only_exact_identifier_is_returned_with_infinite_dense_distance():
    exact_doc = "Product SKU ZX-900 supports 4K video output."
    dense_doc = "Our company provides customer support services."
    collection = FakeCollection(
        dense_doc=dense_doc,
        dense_meta={"source": "overview.md", "type": "text"},
        dense_distance=0.8,
        docs=[dense_doc, exact_doc],
        metadatas=[
            {"source": "overview.md", "type": "text"},
            {"source": "products.md", "type": "text"},
        ],
    )

    result = hybrid_retrieve(
        collection=collection,
        query_embedding=[0.1, 0.2],
        query_text="ZX-900",
        top_k=2,
        dense_weight=0.7,
        keyword_weight=0.3,
        candidate_pool_size=8,
        min_fusion_score=0.05,
    )

    exact_index = result["documents"].index(exact_doc)
    assert result["distances"][exact_index] == float("inf")
    assert "keyword" in result["matched_by"][exact_index]
    assert result["keyword_scores"][exact_index] == 1.0


def test_reciprocal_rank_fusion_prefers_document_ranked_by_both_channels():
    both_doc = "The robot uses ROS 2 for navigation and device communication."
    dense_only_doc = "The robot has a camera and speaker."
    collection = FakeCollection(
        dense_doc=both_doc,
        dense_meta={"source": "robot.md", "type": "text"},
        dense_distance=0.2,
        docs=[both_doc, dense_only_doc],
        metadatas=[
            {"source": "robot.md", "type": "text"},
            {"source": "hardware.md", "type": "text"},
        ],
    )

    result = hybrid_retrieve(
        collection=collection,
        query_embedding=[0.1, 0.2],
        query_text="ROS 2 navigation communication",
        top_k=2,
        dense_weight=0.7,
        keyword_weight=0.3,
        candidate_pool_size=8,
        min_fusion_score=0.05,
    )

    assert result["documents"][0] == both_doc
    assert set(result["matched_by"][0]) == {"dense", "keyword"}


def test_keyword_retrieval_respects_metadata_filters():
    collection = FakeCollection(
        dense_doc="General company overview.",
        dense_meta={"department": "general"},
        dense_distance=0.2,
        docs=["Product SKU ZX-900 is available.", "Product SKU ZX-900 is discontinued."],
        metadatas=[
            {"department": "sales"},
            {"department": "archive"},
        ],
    )

    result = hybrid_retrieve(
        collection=collection,
        query_embedding=[0.1],
        query_text="ZX-900",
        top_k=5,
        dense_weight=0.7,
        keyword_weight=0.3,
        candidate_pool_size=8,
        min_fusion_score=0.05,
        metadata_filters={"department": "sales"},
    )

    assert "Product SKU ZX-900 is available." in result["documents"]
    assert "Product SKU ZX-900 is discontinued." not in result["documents"]
