import pandas as pd
from sklearn.neighbors import NearestNeighbors

# -----------------------------
# Step 1: Load the OULAD dataset files
# -----------------------------
student_info = pd.read_csv("studentInfo.csv")
student_vle = pd.read_csv("studentVle.csv")
vle = pd.read_csv("vle.csv")

# -----------------------------
# Step 2: Merge studentVle with VLE to get activity type
# -----------------------------
merged = pd.merge(student_vle, vle, on="id_site")

# -----------------------------
# Step 3: Create interaction matrix (students × activity types)
# -----------------------------
interaction_matrix = merged.pivot_table(
    index="id_student",
    columns="activity_type",
    values="sum_click",
    aggfunc="sum",
    fill_value=0
)

# -----------------------------
# Step 4: Normalize interactions per student (row-wise)
# -----------------------------
interaction_normalized = interaction_matrix.div(interaction_matrix.sum(axis=1), axis=0)

# -----------------------------
# Step 5: Transpose to get activity-type × student matrix
# -----------------------------
item_matrix = interaction_normalized.T

# -----------------------------
# Step 6: Fit Nearest Neighbors (item-based collaborative filtering)
# -----------------------------
knn_model = NearestNeighbors(metric='cosine', algorithm='brute')
knn_model.fit(item_matrix.values)

# -----------------------------
# Step 7: Recommendation function using item similarity
# -----------------------------
def recommend_knn(student_id, top_n=3):
    if student_id not in interaction_normalized.index:
        return "Student ID not found."

    student_vector = interaction_normalized.loc[student_id]
    interacted_items = student_vector[student_vector > 0].index.tolist()

    if not interacted_items:
        return "No activity data for this student."

    scores = {}

    for item in interacted_items:
        try:
            item_idx = item_matrix.index.get_loc(item)
            distances, indices = knn_model.kneighbors([item_matrix.iloc[item_idx]], n_neighbors=top_n + 1)

            # Iterate over neighbors, skip the item itself
            for dist, idx in zip(distances[0][1:], indices[0][1:]):
                neighbor_item = item_matrix.index[idx]
                similarity = 1 - dist  # Convert cosine distance to similarity
                scores[neighbor_item] = scores.get(neighbor_item, 0) + similarity
        except KeyError:
            continue  # Skip if activity type not found

    # Remove items the student already interacted with
    for seen in interacted_items:
        scores.pop(seen, None)

    return pd.Series(scores).sort_values(ascending=False).head(top_n)

# -----------------------------
# Step 8: Example usage
# -----------------------------
if __name__ == "__main__":
    test_id = 28808
    print(f"KNN Recommendations for student {test_id}:")
    print(recommend_knn(student_id=test_id, top_n=3))
