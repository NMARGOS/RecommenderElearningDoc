# bring in pandas for table , and sklearn for similarity calcs
import pandas as pd 
from sklearn.metrics.pairwise import cosine_similarity

#### load ithe csvsOULAD, we need 3 out of the 7
student_info = pd.read_csv("studentInfo.csv")  # info like age, edu, disabilitiess
student_vle = pd.read_csv("studentVle.csv")# click behavour like how many clicks on activities
vle = pd.read_csv("vle.csv")              # info about each activity type

# first we merge the click data with the activity type desc. using id_site 
merged = pd.merge(student_vle, vle, on="id_site")

# then we build the big matrix: rows are students, columns are activity types, values = sum of clicks
interaction_matrix = merged.pivot_table(
    index='id_student',
    columns='activity_type',
    values='sum_click',
    aggfunc='sum',
    fill_value=0      # fill missing combos w/ 0 (means no clicks!)
)

# normalize it row-wise (each student becomes a vector adding to 1), so we can compare
interaction_normalized = interaction_matrix.div(interaction_matrix.sum(axis=1), axis=0)

# compute pop. of activities so we know which are too popular
# this will help with re-ranking to avoid ONLY pushing the tooop activities
activity_popularity = interaction_matrix.sum(axis=0)         # total clicks per column (activity)
activity_popularity = activity_popularity / activity_popularity.max()  # scale 0-1

# we build similarity matrix BETWEEN activity types
# we compare cols of interaction matrix to see what acts are similar
activity_similarity = cosine_similarity(interaction_normalized.T)
similarity_df = pd.DataFrame(
    activity_similarity,
    index=interaction_normalized.columns,    # labels = activity types
    columns=interaction_normalized.columns
)

# Recommend activities for RETURNING students
def recommend_activities(student_id, top_n=3):
    # make sure this student exists in the data or else send msg 
    if student_id not in interaction_normalized.index:
        return "Student ID not found."

    # get that student's row (normalized click pattern)
    student_vector = interaction_normalized.loc[student_id]

    # get what activities this student clicked on before
    interacted = student_vector[student_vector > 0].index.tolist()

    # get similarity scores for everything based on those seen ones 
    scores = similarity_df[interacted].sum(axis=1)

    # uncommment bleow if you want to exclude known activities (during eval)
    # scores = scores.drop(labels=interacted)

    # sort and get the best 3
    recommendations = scores.sort_values(ascending=False).head(top_n)
    return recommendations


#b  Recommend for COLD START (new students!!) 
def recommend_for_new_student(profile, top_n=3):
    # pull info from profile sent from frontend
    age = profile.get("AgeBand")
    education = profile.get("HighestEducation")
    disability = profile.get("Disability")
    module = profile.get("Module")

    # join the demographics with the normalized clicks to filter users 
    merged_data = pd.merge(student_info, interaction_normalized, left_on="id_student", right_index=True)

    # only keep people with matching demo features 
    filtered = merged_data[
        (merged_data["age_band"] == age) &
        (merged_data["highest_education"] == education) &
        (merged_data["disability"] == disability)
    ]

    # get only the activity columns
    activity_cols = interaction_normalized.columns

    # take the average activity vector for these demo-similar students
    profile_avg = filtered[activity_cols].mean()

    # optionally blend in module-level activity behavior too
    if module:
        mod_filtered = merged_data[merged_data["code_module"] == module]
        module_avg = mod_filtered[activity_cols].mean()
        profile_avg = (profile_avg + module_avg) / 2   # simple avg of the two

    # fallback: if no one matched or was super sparse
    if profile_avg.isna().all():
        profile_avg = interaction_normalized.mean()

    # return top 3 most used activity types from this average profile
    return profile_avg.sort_values(ascending=False).head(top_n).to_dict()


# -Popularity penalty to avoid echo-chamberrr
def apply_popularity_penalty(recommendations, popularity, weight=0.3):
    # make a new dict of adjusted scores after subtracting popularity * weight
    adjusted = {}
    for activity, score in recommendations.items():
        penalty = weight * popularity.get(activity, 0)  # how popular it is
        adjusted[activity] = score - penalty            # lower the score if it's too popular
    # sort and return only top ones
    return dict(sorted(adjusted.items(), key=lambda x: x[1], reverse=True))
