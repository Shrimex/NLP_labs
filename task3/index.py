import argparse
import gzip
import os
import numpy as np

from sentence_transformers import SentenceTransformer
import faiss


def load_news(path: str, limit: int | None = None):
    texts = []
    labels = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            label, title, text = parts[0], parts[1], parts[2]
            full_text = f"{title}. {text}"
            labels.append(label)
            texts.append(full_text)
            if limit is not None and len(texts) >= limit:
                break
    return texts, labels


def main():
    parser = argparse.ArgumentParser(description="Индексация корпуса для семантического поиска")
    parser.add_argument("--data", type=str, default="news.txt.gz",
                        help="Путь к файлу с документами (gzip)")
    parser.add_argument("--limit", type=int, default=5000,
                        help="Максимальное число документов для индексации")
    parser.add_argument("--model", type=str,
                        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                        help="Имя модели sentence-transformers")
    parser.add_argument("--index", type=str, default="news.index",
                        help="Файл для сохранения FAISS-индекса")
    parser.add_argument("--texts", type=str, default="news_texts.npy",
                        help="Файл для сохранения текстов")
    parser.add_argument("--labels", type=str, default="news_labels.npy",
                        help="Файл для сохранения меток")
    args = parser.parse_args()

    print(f"Загружаем документы из {args.data} ...")
    texts, labels = load_news(args.data, limit=args.limit)
    print(f"Загружено документов: {len(texts)}")

    print(f"Загружаем модель эмбеддингов: {args.model}")
    model = SentenceTransformer(args.model)

    print("Вычисляем эмбеддинги документов...")
    embeddings = model.encode(texts, batch_size=64, show_progress_bar=True, convert_to_numpy=True)

    dim = embeddings.shape[1]
    print(f"Размерность эмбеддинга: {dim}")

    index = faiss.IndexFlatIP(dim)
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    print(f"В индекс добавлено векторов: {index.ntotal}")

    print(f"Сохраняем индекс в {args.index}")
    faiss.write_index(index, args.index)

    print("Сохраняем тексты и метки...")
    np.save(args.texts, np.array(texts, dtype=object))
    np.save(args.labels, np.array(labels, dtype=object))

    print("Готово.")


if __name__ == "__main__":
    main()
