from langchain_text_splitters import RecursiveCharacterTextSplitter
from fastembed import TextEmbedding

with open("example.txt", "r", encoding="utf-8") as f: # Read example privacy policy 
    document = f.read()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100) # Approx 256 token size
texts = text_splitter.split_text(document) # Split the document into recursive chunks

model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5") # Dense embedding for semantic search
embeddings = list(model.embed(texts)) # Pass the list of text to the embedding model

# with open("embeddings.txt", "w", encoding="utf-8") as f:
#     for text, embedding in zip(texts, embeddings):
#         f.write(f"Text:\n{text}\n")
#         f.write("Embedding:\n")
#         f.write(",".join(f"{x:.6f}" for x in embedding) + "\n")
#         f.write("\n" + "-" * 80 + "\n\n")

