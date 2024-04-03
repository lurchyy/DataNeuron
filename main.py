from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
df = pd.read_csv('data.csv')

model = SentenceTransformer('distilbert-base-nli-mean-tokens')

text1_embeddings = model.encode(df['text1'].tolist())
text2_embeddings = model.encode(df['text2'].tolist())

similarity_scores = cosine_similarity(text1_embeddings, text2_embeddings)

for i in range(len(df)):
    print(f"Similarity between text1[{i}] and text2[{i}]: {similarity_scores[i][i]}")
df['similarity_score'] = similarity_scores.diagonal()
