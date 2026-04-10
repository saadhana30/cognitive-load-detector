import pandas as pd
from sklearn.cluster import KMeans

df = pd.read_csv("data/processed/features.csv")

# take only numeric columns
X = df.select_dtypes(include=['float64', 'int64'])

# clustering
kmeans = KMeans(n_clusters=3, random_state=42)
df['cognitive_load'] = kmeans.fit_predict(X)

# convert 0,1,2 → labels
df['cognitive_load'] = df['cognitive_load'].map({
    0: "low",
    1: "medium",
    2: "high"
})

df.to_csv("data/processed/labeled_features.csv", index=False)

print("DONE")