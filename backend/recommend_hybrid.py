import pandas as pd
from recommender import recommend_for_new_student, interaction_normalized
from knn_recommender import recommend_knn

def recommend_hybrid(student_id, profile, top_n=3, weight_cold=0.4, weight_knn=0.6):
   
    #Combines cold-start and KNN collaborative filtering into a hybrid model.





    # Get the outptscores from both models
    cold_scores = recommend_for_new_student(profile, top_n=None)
    knn_scores = recommend_knn(student_id=student_id, top_n=None)

    if isinstance(knn_scores, str) or knn_scores is None:
        return pd.Series(cold_scores).sort_values(ascending=False).head(top_n)


    # Build unified list of activs 
    all_activities = list(interaction_normalized.columns)


    # Convert to full-length Series with all activities
    cold_series = pd.Series(cold_scores, index=all_activities).fillna(0)
    knn_series = pd.Series(knn_scores, index=all_activities).fillna(0)




    # Weighted combination
    hybrid_scores = (weight_cold * cold_series) + (weight_knn * knn_series)




    return hybrid_scores.sort_values(ascending=False).head(top_n)

#test run
if __name__ == "__main__":
    test_profile = {
        "AgeBand": "35-55",
        "HighestEducation": "HE Qualification",
        "Disability": "N",
        "Module": "AAA"
    }
    print(recommend_hybrid(student_id=28808, profile=test_profile, top_n=3))
