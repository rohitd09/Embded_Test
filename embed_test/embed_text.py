from langchain_text_splitters import RecursiveCharacterTextSplitter
from fastembed import TextEmbedding, SparseTextEmbedding

with open("example.txt", "r", encoding="utf-8") as f: # Read example privacy policy 
    document = f.read()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100) # Approx 256 token size
texts = text_splitter.split_text(document) # Split the document into recursive chunks

model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5") # Dense embedding for semantic search
embeddings = list(model.embed(texts)) # Pass the list of text to the embedding model

with open("embed_test/embeddings.txt", "w", encoding="utf-8") as f:
    for text, embedding in zip(texts, embeddings):
        f.write(f"Text:\n{text}\n")
        f.write("Embedding:\n")
        f.write(",".join(f"{x:.6f}" for x in embedding) + "\n")
        f.write("\n" + "-" * 80 + "\n\n")

# Creating Sparse Embeddings
sparse_model = SparseTextEmbedding(model_name="prithivida/Splade_PP_en_v1")
sparse_embeddings = list(sparse_model.embed(texts))

with open("embed_test/sparse_embeddings.txt", "w", encoding="utf-8") as f:
    for text, embedding in zip(texts, sparse_embeddings):
        f.write(f"Text:\n{text}\n")
        f.write("Embedding (sparse vector as index:weight):\n")
        f.write(", ".join(f"{idx}:{val:.6f}" for idx, val in zip(embedding.indices, embedding.values)) + "\n")
        f.write("\n" + "-" * 80 + "\n\n")