from ast import literal_eval

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from content_based_recommender import get_recommendations
from data import get_metadata_with_credits_and_keywords


def main():
    metadata = get_metadata_with_credits_and_keywords()
    features = ["cast", "crew", "keywords", "genres"]
    for feature in features:
        metadata[feature] = metadata[feature].apply(literal_eval)

    metadata["director"] = metadata["crew"].apply(get_director)

    features = ["cast", "keywords", "genres"]
    for feature in features:
        metadata[feature] = metadata[feature].apply(get_top_3_from_list)

    features = ["cast", "keywords", "genres", "director"]
    for feature in features:
        metadata[feature] = metadata[feature].apply(clean_data)

    metadata["soup"] = metadata.apply(create_soup, axis=1)

    count = CountVectorizer(stop_words="english")
    count_matrix = count.fit_transform(metadata["soup"])
    cosine_sim = cosine_similarity(count_matrix, count_matrix)

    metadata = metadata.reset_index()
    indices = pd.Series(metadata.index, index=metadata["title"])
    print(get_recommendations("The Dark Knight Rises", metadata, cosine_sim, indices))


def get_director(data):
    for x in data:
        if x["job"] == "Director":
            return x["name"]

    return np.nan


def get_top_3_from_list(l) -> list:
    if isinstance(l, list):
        names = [i["name"] for i in l]

        if len(names) > 3:
            names = names[:3]
        return names

    return []


def clean_data(data):
    if isinstance(data, list):
        return [str.lower(i.replace(" ", "")) for i in data]
    elif isinstance(data, str):
        return str.lower(data.replace(" ", ""))
    else:
        return ""


def create_soup(metadata):
    return (
        " ".join(metadata["keywords"])
        + " ".join(metadata["cast"])
        + " ".join(metadata["director"])
        + " ".join(metadata["genres"])
    )


if __name__ == "__main__":
    main()
