import argparse
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss


def main():
    parser = argparse.ArgumentParser(description="Семантический поиск по новостям")
    parser.add_argument("query", type=str, help="Текст поискового запроса")
    parser.add_argument("--model", type=str,
                        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                        help="Имя модели sentence-transformers (должна совпадать с index.py)")
    parser.add_argument("--index", type=str, default="news.index",
                        help="Файл FAISS-индекса")
    parser.add_argument("--texts", type=str, default="news_texts.npy",
                        help="Файл с текстами документов")
    parser.add_argument("--labels", type=str, default="news_labels.npy",
                        help="Файл с метками документов")
    parser.add_argument("--top_k", type=int, default=5,
                        help="Сколько похожих документов выводить")
    args = parser.parse_args()

    print(f"Загружаем индекс из {args.index} ...")
    index = faiss.read_index(args.index)

    print("Загружаем тексты и метки...")
    texts = np.load(args.texts, allow_pickle=True)
    labels = np.load(args.labels, allow_pickle=True)

    print(f"Загружаем модель эмбеддингов: {args.model}")
    model = SentenceTransformer(args.model)

    print(f"Формируем запрос: {args.query}")
    query_emb = model.encode([args.query], convert_to_numpy=True)
    faiss.normalize_L2(query_emb)

    print("Ищем похожие документы...")
    distances, indices = index.search(query_emb, args.top_k)

    print("\nРезультаты:")
    for rank, (idx, score) in enumerate(zip(indices[0], distances[0]), start=1):
        label = labels[idx]
        text = texts[idx]
        snippet = text[:300].replace("\n", " ")
        print(f"\n#{rank} (score={score:.4f}, label={label})")
        print(snippet + ("..." if len(text) > 300 else ""))


if __name__ == "__main__":
    main()