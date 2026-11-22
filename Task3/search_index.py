import argparse
from contextlib import closing

from huggingface_hub import snapshot_download
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from build_index import COLLECTION, INDEX_PATH, MODEL_NAME


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("questions", nargs="+", help="Поисковые запросы в кавычках")
    args = parser.parse_args()

    if not INDEX_PATH.is_dir():
        raise SystemExit("No index data, build it via build_index.py.")

    with closing(QdrantClient(path=str(INDEX_PATH))) as client:
        if not client.collection_exists(COLLECTION):
            raise SystemExit(f"Collection has no index {COLLECTION}.")
        
        print(f"Чанков в индексе: {client.count(COLLECTION, exact=True).count}")
        model_path = snapshot_download(MODEL_NAME, local_files_only=True)
        model = SentenceTransformer(model_path, device="cpu", local_files_only=True)
        for question in args.questions:
            vector = model.encode("query: " + question, normalize_embeddings=True)
            points = client.query_points(
                collection_name=COLLECTION, query=vector.tolist(), limit=3, with_payload=True
            ).points

            print(f"\nЗапрос: {question}")
            for rank, point in enumerate(points, start=1):
                chunk = point.payload
                section = chunk["section"] or chunk["title"]
                
                print(f"\n{rank}. {point.score:.4f} | {chunk['path']} | {section}")
                print(chunk["text"])


if __name__ == "__main__":
    main()
