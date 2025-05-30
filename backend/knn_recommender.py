import pandas as pd
from sklearn.neighbors import NearestNeighbors

student_info = pd.read_csv("studentInfo.csv")
student_vle = pd.read_csv("studentVle.csv")
vle = pd.read_csv("vle.csv")

## use the on id site to merge, merge studentVle with VLE to get activity type

merged = pd.merge(student_vle, vle, on="id_site")


 # interaction matrix crreation (students times the activity types))

interaction_matrix = merged.pivot_table(
    index="id_student",
    columns="activity_type",
    values="sum_click",
    aggfunc="sum",
    fill_value=0
)


#Normalize interactions per student (    row-wise)
interaction_normalized = interaction_matrix.div(interaction_matrix.sum(axis=1), axis=0)


#Transpose to get activity-type * student matrix

item_matrix = interaction_normalized.T


#=Fit KNNN

knn_model = NearestNeighbors(metric='cosine', algorithm='brute')
knn_model.fit(item_matrix.values)


# Recommendation function using item similarity

def recommend_knn(student_id, top_n=3):
    if student_id not in interaction_normalized.index:
        return "Student ID not found."

    student_vector = interaction_normalized.loc[student_id]
    interacted_items = student_vector[student_vector > 0].index.tolist()

    if not interacted_items:
        return "No activity data for this student."

    scores = {}
    neighbors_to_fetch = top_n + 1 if top_n is not None else 6  

    for item in interacted_items:
        try:
            item_idx = item_matrix.index.get_loc(item)
            distances, indices = knn_model.kneighbors(
                [item_matrix.iloc[item_idx]], n_neighbors=neighbors_to_fetch
            )

            for dist, idx in zip(distances[0][1:], indices[0][1:]):
                neighbor_item = item_matrix.index[idx]
                similarity = 1 - dist
                scores[neighbor_item] = scores.get(neighbor_item, 0) + similarity
        except KeyError:
            continue

    # Remove items the student already interacted with, when evaluating comment out the line below
    #for seen in interacted_items:
        #scores.pop(seen, None)

    return pd.Series(scores).sort_values(ascending=False).head(top_n)


# =Example usage

if __name__ == "__main__":
    test_id = 28808
    print(f"KNN Recommendations for student {test_id}:")
    print(recommend_knn(student_id=test_id, top_n=3))
