# ########### evaluate.py
# this file runs evaluations for both existing students (with past clicks)
# and cold-start students (new ones without past data) 

from recommender import (
    interaction_normalized,
    recommend_activities,
    recommend_for_new_student
)

import pandas as pd
import matplotlib.pyplot as plt

print("importing done")  # just to know things loaded okay

####Precision and Recall @K 
# this function helps us evaluate a recsys for a student
# we check how many of the top K things recommended were things the user actually used 
def precision_recall_at_k(recommend_func, student_id, k=3):
    try:
        # call the recommender function (like recommend_activities)
        recommended = recommend_func(student_id=student_id, top_n=k).index.tolist()

        # get the actual activities this student interacted with
        actual_vector = interaction_normalized.loc[student_id]
        actual = actual_vector[actual_vector > 0].index.tolist()

        # show what's going on
        print(f"\nStudent ID: {student_id}")
        print("Actual interacted activity types:", actual)
        print("Recommended activity types:", recommended)

        # how many overlaps? that's our true positives!
        true_positives = len(set(recommended) & set(actual))

        # calculate the metrics
        precision = true_positives / k  # how many recommended were right
        recall = true_positives / len(actual) if actual else 0  # how much of real stuff we retrieved

        print(f"  ✅ True positives: {true_positives} | Precision: {precision:.2f} | Recall: {recall:.2f}")
        return precision, recall

    except Exception as e:
        # in case something breaks (maybe student ID not found), show the error
        print(f"❌ Error for student {student_id}: {e}")
        return None, None


# un Evaluation for Existing Students 
def run_existing_user_evaluation():
    # take the first 50 students for testing — can adjust later
    test_students = interaction_normalized.index[:50]
    results = []

    for sid in test_students:
        # evaluate each student and collect the metrics
        p, r = precision_recall_at_k(recommend_activities, student_id=sid, k=3)
        if p is not None:
            results.append((sid, p, r))

    # save results into a table (CSV) for later
    df = pd.DataFrame(results, columns=['student_id', 'precision@3', 'recall@3'])
    df.to_csv("evaluation_results.csv", index=False)

    # show some summary stats (mean, std, min, etc)
    print("\n--- Evaluation Summary ---")
    print(df.describe())

    # --- Optional chart to visualize precision vs recall
    plt.figure(figsize=(6, 4))
    plt.scatter(df['precision@3'], df['recall@3'], alpha=0.7)
    plt.title("Precision vs Recall (Top-3)")
    plt.xlabel("Precision@3")
    plt.ylabel("Recall@3")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("precision_recall_plot.png")
    plt.show()


#  Cold Start Diversity 
# here we test if different types of new students get different results
def evaluate_cold_start():
    # some sample profiles for new students, changing age, education, disability, module etc
    demographics_test = [
        {"AgeBand": "35-55", "HighestEducation": "HE Qualification", "Disability": "N", "Module": "AAA"},
        {"AgeBand": "0-35", "HighestEducation": "A Level or Equivalent", "Disability": "Y", "Module": "BBB"},
        {"AgeBand": "55<=", "HighestEducation": "Post Graduate Qualification", "Disability": "N", "Module": "CCC"},
        {"AgeBand": "35-55", "HighestEducation": "HE Qualification", "Disability": "Y", "Module": "DDD"}
    ]

    all_activities = []

    print("\n--- Cold-Start Recommendations ---")
    for d in demographics_test:
        # get top 3 recs for each of these profiles
        recs = recommend_for_new_student(d, top_n=3)
        print(f"\nInput: {d}")
        print("Recommended Activities:", recs)
        all_activities += list(recs.keys())  # add to the list of total activities

    # check how diverse the results were  how many unique vs total
    diversity = len(set(all_activities)) / len(all_activities)
    print(f"\nCold-start diversity score: {diversity:.2f}")


#  Run everything if we call this file directly 
print("running evaluations")
if __name__ == "__main__":
    print("Running evaluation for existing users")
    run_existing_user_evaluation()

    print("Running cold-start evaluation")
    evaluate_cold_start()
