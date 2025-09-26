import pandas as pd


def get_metadata():
    metadata = pd.read_csv("data/movies_metadata.csv", low_memory=False)
    return metadata


def get_metadata_with_credits_and_keywords():
    metadata = pd.read_csv("data/movies_metadata.csv", low_memory=False)
    credits = pd.read_csv("data/credits.csv")
    keywords = pd.read_csv("data/keywords.csv")

    metadata = metadata.drop([19730, 29503, 35587])

    credits["id"] = credits["id"].astype("int")
    keywords["id"] = keywords["id"].astype("int")
    metadata["id"] = metadata["id"].astype("int")

    metadata = metadata.merge(credits, on="id")
    metadata = metadata.merge(keywords, on="id")

    return metadata
