import gzip
import json
import pandas as pd
def procesar_reviews(input_path, output_csv, limite=50000):
    data = []
    with gzip.open(input_path, 'rt', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= limite:
                break
            review = json.loads(line)
            data.append({'reviewText': review.get('reviewText'), 'overall': review.get('overall')})
    df = pd.DataFrame(data)
    def sentiment(star):
        if star <= 2: return 'negativo'
        elif star == 3: return 'neutro'
        elif star >= 4: return 'positivo'
    df['sentimiento'] = df['overall'].apply(sentiment)
    keywords_precio = ['precio', 'costoso', 'caro', 'barato']
    keywords_calidad = ['calidad', 'excelente', 'defectuoso', 'mala']
    keywords_envio = ['envío', 'entrega', 'llegó', 'tardó']
    def detect_aspect(text, keywords):
        text = str(text).lower()
        return any(kw in text for kw in keywords)
    df['mencion_precio'] = df['reviewText'].apply(lambda x: detect_aspect(x, keywords_precio))
    df['mencion_calidad'] = df['reviewText'].apply(lambda x: detect_aspect(x, keywords_calidad))
    df['mencion_envio'] = df['reviewText'].apply(lambda x: detect_aspect(x, keywords_envio))
    df.to_csv(output_csv, index=False)
    print(f"[OK] Archivo guardado en: {output_csv}")
    return df
