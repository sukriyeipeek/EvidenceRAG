from sentence_transformers import SentenceTransformer


# Embedding modelimizi yüklüyoruz
model = SentenceTransformer("all-MiniLM-L6-v2")


# Örnek metinler
documents = [
    "Transformers use self-attention mechanisms.",
    "Self-attention is an important part of Transformer architecture.",
    "CNNs are commonly used for image classification.",
    "Random Forest is an ensemble learning algorithm.",
    "Python is a popular programming language."
]


# Metinleri embedding vektörlerine dönüştürüyoruz
embeddings = model.encode(documents)


print("Embedding shape:", embeddings.shape)
print("\nİlk dokümanın embedding'i:")
print(embeddings[0])