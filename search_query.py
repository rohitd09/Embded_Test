from qdrant_client import QdrantClient, models
from fastembed import TextEmbedding, SparseTextEmbedding

# 1. Connect to Qdrant using local embedded path
client = QdrantClient(path=r"C:\Users\rohit\OneDrive\Projects\FoxLint\similarity_search")

# 2. Load the same models used during indexing
dense_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
sparse_model = SparseTextEmbedding(model_name="prithivida/Splade_PP_en_v1")

# 3. Define your search query
query_text = "What personal data is collected?"

# 4. Embed the query
query_dense = list(dense_model.embed([query_text]))[0]
query_sparse = list(sparse_model.embed([query_text]))[0]

# 5. Construct Prefetch for both dense and sparse vectors
prefetch = [
    models.Prefetch(
        query=query_dense,
        using="dense"
    ),
    models.Prefetch(
        query=models.SparseVector(
            indices=query_sparse.indices,
            values=query_sparse.values
        ),
        using="sparse"
    )
]

# 6. Perform hybrid search using dbsf fusion
results = client.query_points(
    collection_name="privacy_policy_collection",
    query=models.FusionQuery(
        fusion=models.Fusion.DBSF  # Use Distribution-Based Score Fusion
    ),
    prefetch=prefetch,
    limit=5  # Number of results to return
)

# 7. Print the results
for i, hit in enumerate(results.points, 1):
    print(f"\nResult {i} (score: {hit.score:.4f}):")
    print(hit.payload.get("text", "[No text payload]"))