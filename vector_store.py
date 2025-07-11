from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    VectorParams,
    SparseVectorParams,
    SparseIndexParams,
    SparseVector,
    Distance,
)
from qdrant_client.models import PointStruct
from fastembed import TextEmbedding, SparseTextEmbedding

# 1. Read and split the document
with open("example.txt", "r", encoding="utf-8") as f:
    document = f.read()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
texts = text_splitter.split_text(document)

print(f"Documents split into {len(texts)} chunks")

# 2. Generate dense and sparse embeddings
dense_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
embeddings = list(dense_model.embed(texts))

sparse_model = SparseTextEmbedding(model_name="prithivida/Splade_PP_en_v1")
sparse_embeddings = list(sparse_model.embed(texts))

# 3. Qdrant setup
client = QdrantClient(path=r'C:\Users\rohit\OneDrive\Projects\FoxLint\similarity_search')

# 4. Collection schema
vector_config = {
    "dense": VectorParams(
        size=384,  # Dim of bge-small embedding model
        distance=Distance.COSINE,
    )
}
sparse_vector_config = {
    "sparse": SparseVectorParams(
        index=SparseIndexParams(
            on_disk=False,
        )
    )
}

# 5. Delete and create collection (wait for completion)
client.delete_collection(collection_name="privacy_policy_collection")
client.create_collection(
    collection_name="privacy_policy_collection",
    vectors_config=vector_config,
    sparse_vectors_config=sparse_vector_config,
)

# 6. Prepare and upsert points
points = []
for idx, (text, dense_vec, sparse_vec) in enumerate(zip(texts, embeddings, sparse_embeddings)):
    point = PointStruct(
        id=idx,
        payload={"text": text},
        vector={
            "dense": dense_vec,
            "sparse": SparseVector(
                indices=sparse_vec.indices,
                values=sparse_vec.values
            )
        }
    )
    points.append(point)

for point in points:
    client.upsert(
        collection_name="privacy_policy_collection",
        points=[point],   # Pass as a single-item list
        wait=True
    )


# (Optional) Upsert in batches if you have many points
# batch_size = 1
# for i in range(0, len(points), 1):
#     client.upsert(
#         collection_name="privacy_policy_collection",
#         points=points[i:i+1],
#         wait=True
#     )

# For small datasets,

# client.upsert(
#     collection_name="privacy_policy_collection",
#     points=points,
#     wait=True
# )

# 7. Confirm insertion
print("Total points in collection:", client.count(collection_name="privacy_policy_collection").count)