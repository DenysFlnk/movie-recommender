from data import get_metadata


# Simple recommendation for best 20 movies based on weighted rating
def main():
    metadata = get_metadata()
    mean_c = metadata["vote_average"].mean()

    percentile_90_m = metadata["vote_count"].quantile(0.90)

    filtered_metadata = metadata.copy().loc[metadata["vote_count"] >= percentile_90_m]

    def weighted_rating(data, m=percentile_90_m, C=mean_c):
        v = data["vote_count"]
        R = data["vote_average"]

        return (v / (v + m) * R) + (m / (m + v) * C)

    filtered_metadata["score"] = filtered_metadata.apply(weighted_rating, axis=1)
    filtered_metadata = filtered_metadata.sort_values("score", ascending=False)
    print(filtered_metadata[["title", "vote_count", "vote_average", "score"]].head(20))


if __name__ == "__main__":
    main()
