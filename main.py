from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer('distilbert-base-nli-mean-tokens')
txt1 = str(input("Enter the first text: "))
txt2 = str(input("Enter the second text: "))
txt1_embedding = model.encode([txt1])
txt2_embedding = model.encode([txt2])


similarity = cosine_similarity(txt1_embedding, txt2_embedding)[0][0]

print(f"Similarity Score:{similarity:.2f}")

