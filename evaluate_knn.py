from knn_recommender import interaction_normalized, recommend_knn
import pandas as pd
import numpy as np

def precision_recall_at_k_knn(recommend_func, student_id, k=3):
    try:
        
        recommended = recommend_func(student_id=student_id, top_n=k)
        if isinstance(recommended, str):  
            return None, None

        recommended = recommended.index.tolist()


        actual_vector = interaction_normalized.loc[student_id]
        actual = actual_vector[actual_vector > 0].index.tolist()





        true_positives = len(set(recommended) & set(actual))
        precision = true_positives / k
        recall = true_positives / len(actual) if actual else 0
        return precision, recall
    except Exception as e:
        return None, None
    

#


valid_students = interaction_normalized.index.tolist()
sample_ids = np.random.choice(valid_students, size=50, replace=False)

results = []

for sid in sample_ids:
    precision, recall = precision_recall_at_k_knn(recommend_knn, sid, k=3)
    if precision is not None:
        results.append({
            "student_id": sid,
            "precision@3": precision,
            "recall@3": recall
        })

# Create DataFrame of results
results_df = pd.DataFrame(results)
print("Evaluation Ssummary (KNN Model) ")
print(results_df.describe())