import pandas as pd
import numpy as np
from recommend_hybrid import recommend_hybrid
from recommender import interaction_normalized

# Define the evaluation function
def precision_recall_at_k_hybrid(recommend_func, student_id, profile, k=3):
    try:
        recommended = recommend_func(student_id=student_id, profile=profile, top_n=k)
        if isinstance(recommended, str) or recommended is None:
            return None, None

        recommended_items = recommended.index.tolist()
        actual_vector = interaction_normalized.loc[student_id]
        actual_items = actual_vector[actual_vector > 0].index.tolist()

        true_positives = len(set(recommended_items) & set(actual_items))
        precision = true_positives / k
        recall = true_positives / len(actual_items) if actual_items else 0
        return precision, recall

    except Exception as e:
        return None, None

# Sample evaluation
def evaluate_hybrid_model(k=3, sample_size=50):
    results = []
    valid_students = interaction_normalized.index.tolist()
    sample_ids = np.random.choice(valid_students, size=sample_size, replace=False)

    for sid in sample_ids:
        # Create a fake profile from studentInfo 
        # generate dummy demographic profile per student (fallback-safe)
        profile = {
            "AgeBand": "35-55",
            "HighestEducation": "HE Qualification",
            "Disability": "N",
            "Module": "AAA"
        }

        precision, recall = precision_recall_at_k_hybrid(recommend_hybrid, sid, profile, k=k)
        if precision is not None:
            results.append({
                "student_id": sid,
                f"precision@{k}": precision,
                f"recall@{k}": recall
            })

    results_df = pd.DataFrame(results)
    print(f"\n--- Evaluation Summary (Hybrid Model) ---")
    print(results_df.describe())
    return results_df


if __name__ == "__main__":
    evaluate_hybrid_model()
