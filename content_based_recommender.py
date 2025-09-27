import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from data import get_metadata


# Content based recommendation of 10 most description similar movies (using Term Frequency-Inverse Document Frequency)
def main():
    metadata = get_metadata()
    tfidf = TfidfVectorizer(stop_words="english")
    metadata["overview"] = metadata["overview"].fillna("")

    tfidf_matrix = tfidf.fit_transform(metadata["overview"])

    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

    indices = pd.Series(metadata.index, index=metadata["title"]).drop_duplicates()

    print(get_recommendations("The Dark Knight Rises", metadata, cosine_sim, indices))


def get_recommendations(title, metadata, cosine_sim, movie_indices):
    idx = movie_indices[title]

    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]

    indices = [i[0] for i in sim_scores]

    return metadata["title"].iloc[indices]


if __name__ == "__main__":
    main()
