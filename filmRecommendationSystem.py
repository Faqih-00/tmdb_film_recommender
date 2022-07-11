import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import word_tokenize
from sklearn.metrics.pairwise import cosine_distances
 
df = pd.read_csv('tmdb_preprocessed_dataset.csv')

class FilmRecommenderSystem:
    def __init__(self, data, content_col):
        self.df = pd.read_csv(data)
        self.content_col = content_col
        self.encoder = None
        self.bank = None

    def fit(self):
        self.encoder = CountVectorizer(
            stop_words='english', tokenizer=word_tokenize)
        self.bank = self.encoder.fit_transform(self.df[self.content_col])

    def recommend(self, idx, top=10):
        content = df.loc[idx, self.content_col]
        code = self.encoder.transform([content])
        dist = cosine_distances(code, self.bank)
        rec_idx = dist.argsort()[0, 1:(top+1)]
        return df.loc[rec_idx]


recsys = FilmRecommenderSystem(
    'tmdb_preprocessed_dataset.csv', content_col="metadata")
recsys.fit()
print(recsys.recommend(0)) # idx 0 = Avatar
