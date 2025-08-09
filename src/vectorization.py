import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
def cargar_y_vectorizar(csv_path, max_features=1000):
    df = pd.read_csv(csv_path)
    df.dropna(inplace=True)
    X_text = df['reviewText'].astype(str)
    y_labels = df['sentimiento']
    vectorizer = TfidfVectorizer(max_features=max_features)
    X = vectorizer.fit_transform(X_text).toarray()
    encoder = LabelEncoder()
    y = encoder.fit_transform(y_labels)
    return train_test_split(X, y, test_size=0.2, random_state=42), encoder.classes_
