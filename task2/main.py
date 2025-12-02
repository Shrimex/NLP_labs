import gzip
import re
import numpy as np

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from pymorphy2 import MorphAnalyzer

from gensim.models import Word2Vec

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score


def load_news(path: str = "news.txt.gz"):

    texts = []
    labels = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            label = parts[0]
            title = parts[1]
            text = parts[2]
            full_text = f"{title} {text}"
            labels.append(label)
            texts.append(full_text)
    return texts, labels


nltk.download("punkt")
nltk.download("stopwords")
morph = MorphAnalyzer()
russian_stopwords = set(stopwords.words("russian"))


def preprocess(text: str):

    text = text.lower()
    text = re.sub(r"[^а-яё\s]", " ", text)
    tokens = word_tokenize(text)
    lemmas = []
    for t in tokens:
        t = t.strip()
        if not t:
            continue
        if t in russian_stopwords:
            continue
        parsed = morph.parse(t)[0]
        lemmas.append(parsed.normal_form)
    return lemmas


def train_word2vec(tokenized_texts):

    model = Word2Vec(
        tokenized_texts,
        vector_size=100,
        window=5,
        min_count=5,
        workers=4,
    )
    return model


def doc_vector_avg(words, model: Word2Vec):

    vectors = [model.wv[w] for w in words if w in model.wv]
    if not vectors:
        return np.zeros(model.vector_size, dtype="float32")
    return np.mean(vectors, axis=0)


def build_tfidf_vectorizer(train_tokens):

    from sklearn.feature_extraction.text import TfidfVectorizer

    def identity(x):
        return x

    vectorizer = TfidfVectorizer(
        tokenizer=identity,
        preprocessor=identity,
        token_pattern=None,
        min_df=2,
    )
    vectorizer.fit(train_tokens)
    feature_names = np.array(vectorizer.get_feature_names_out())
    return vectorizer, feature_names


def doc_vector_tfidf_w2v(words, model: Word2Vec, vectorizer, feature_names):

    tfidf_vec = vectorizer.transform([words])

    if tfidf_vec.nnz == 0:
        return np.zeros(model.vector_size, dtype="float32")

    result_vectors = []
    for idx, value in zip(tfidf_vec.indices, tfidf_vec.data):
        word = feature_names[idx]
        if word in model.wv:
            result_vectors.append(model.wv[word] * value)

    if not result_vectors:
        return np.zeros(model.vector_size, dtype="float32")

    return np.mean(result_vectors, axis=0)

def doc_vector_mean_max(words, model: Word2Vec):

    vectors = [model.wv[w] for w in words if w in model.wv]
    if not vectors:
        return np.zeros(model.vector_size * 2, dtype="float32")
    vectors = np.array(vectors)
    mean_vec = vectors.mean(axis=0)
    max_vec = vectors.max(axis=0)
    return np.concatenate([mean_vec, max_vec])

def main():
    texts, labels = load_news()

    tokenized_texts = [preprocess(t) for t in texts]
    w2v_model = train_word2vec(tokenized_texts)
    X_train_tokens, X_test_tokens, y_train, y_test = train_test_split(
        tokenized_texts,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels,
    )

    X_train_avg = [doc_vector_avg(doc, w2v_model) for doc in X_train_tokens]
    X_test_avg = [doc_vector_avg(doc, w2v_model) for doc in X_test_tokens]

    clf_avg = SVC(kernel="linear", random_state=42)
    clf_avg.fit(X_train_avg, y_train)
    y_pred_avg = clf_avg.predict(X_test_avg)

    acc_avg = accuracy_score(y_test, y_pred_avg)
    print("\n=== РЕЗУЛЬТАТЫ: БАЗОВЫЙ МЕТОД (среднее Word2Vec) ===")
    print(classification_report(y_test, y_pred_avg))

    vectorizer, feature_names = build_tfidf_vectorizer(X_train_tokens)

    X_train_tfidf_w2v = [
        doc_vector_tfidf_w2v(doc, w2v_model, vectorizer, feature_names)
        for doc in X_train_tokens
    ]
    X_test_tfidf_w2v = [
        doc_vector_tfidf_w2v(doc, w2v_model, vectorizer, feature_names)
        for doc in X_test_tokens
    ]
    clf_tfidf_w2v = SVC(kernel="linear", class_weight="balanced", random_state=42)
    clf_tfidf_w2v.fit(X_train_tfidf_w2v, y_train)
    y_pred_tfidf_w2v = clf_tfidf_w2v.predict(X_test_tfidf_w2v)

    acc_tfidf_w2v = accuracy_score(y_test, y_pred_tfidf_w2v)
    print("\n=== РЕЗУЛЬТАТЫ: УЛУЧШЕННЫЙ МЕТОД (TF-IDF + Word2Vec) ===")
    print(classification_report(y_test, y_pred_tfidf_w2v, zero_division=0))

    X_train_mean_max = [doc_vector_mean_max(doc, w2v_model) for doc in X_train_tokens]
    X_test_mean_max = [doc_vector_mean_max(doc, w2v_model) for doc in X_test_tokens]

    clf_mean_max = SVC(kernel="linear", random_state=42)
    clf_mean_max.fit(X_train_mean_max, y_train)
    y_pred_mean_max = clf_mean_max.predict(X_test_mean_max)

    acc_mean_max = accuracy_score(y_test, y_pred_mean_max)
    print("\n=== РЕЗУЛЬТАТЫ: МЕТОД Mean+Max Word2Vec ===")
    print(classification_report(y_test, y_pred_mean_max, zero_division=0))

    print("\nСравнение:")
    print(f"Baseline (mean W2V):        {acc_avg:.4f}")
    print(f"TF-IDF + W2V (balanced):    {acc_tfidf_w2v:.4f}")
    print(f"Mean+Max pooling (W2V):     {acc_mean_max:.4f}")

if __name__ == "__main__":
    main()
