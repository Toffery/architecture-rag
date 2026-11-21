import argparse
from contextlib import closing
from pathlib import Path
from time import perf_counter

from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer

from chunk_documents import load_chunks


INDEX_PATH = Path(__file__).resolve().parent / "qdrant_data"
COLLECTION = "knowledge_base"
MODEL_NAME = "intfloat/multilingual-e5-base"
VECTOR_SIZE = 768
BATCH_SIZE = 8


def main():
    parser = argparse.ArgumentParser(description="Сборка локального индекса Qdrant.")
    parser.add_argument("--rewrite", action="store_true", help="Заменить существующую коллекцию.")
    args = parser.parse_args()

    chunks = list(load_chunks())
    if not chunks:
        raise SystemExit("No chunks in knowledge base")

    with closing(QdrantClient(path=str(INDEX_PATH))) as client:
        exists = client.collection_exists(COLLECTION)
        if exists and not args.rewrite:
            raise SystemExit(f"Collection already at {INDEX_PATH}. Add --rewrite for replace.")

        model = SentenceTransformer(MODEL_NAME, device="cpu")
        passages = [
            f"passage: {chunk['title']}\n{chunk['section']}\n{chunk['text']}"
            for chunk in chunks
        ]
        started = perf_counter()
        vectors = model.encode(
            passages, batch_size=BATCH_SIZE, normalize_embeddings=True, show_progress_bar=True
        )
        if vectors.shape != (len(chunks), VECTOR_SIZE):
            raise SystemExit(f"Unexcepected embedding shape: {vectors.shape}")
        
        embedding_seconds = perf_counter() - started
        points = [
            models.PointStruct(id=idx, vector=vector.tolist(), payload=chunk)
            for idx, (chunk, vector) in enumerate(zip(chunks, vectors), start=1)
        ]
        if exists:
            client.delete_collection(COLLECTION)
        
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config=models.VectorParams(size=VECTOR_SIZE, distance=models.Distance.COSINE),
        )
        client.upsert(collection_name=COLLECTION, points=points)
        count = client.count(collection_name=COLLECTION, exact=True).count
        if count != len(chunks):
            raise SystemExit(f"Writed {count} of {len(chunks)} chunks.")

    print(f"Индекс: {INDEX_PATH}")
    print(f"Чанков в коллекции: {count}")
    print(f"Генерация эмбеддингов: {embedding_seconds:.1f} с")


if __name__ == "__main__":
    main()
