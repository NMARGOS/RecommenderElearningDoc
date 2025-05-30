# import the stuff we need for the api to work ,maybe remove some extra ones
from flask import Flask, request, jsonify
# import our recommend funcshuns and data
from recommender import recommend_for_new_student, interaction_normalized, recommend_activities, activity_popularity, apply_popularity_penalty
from recommend_hybrid import recommend_hybrid 
from flask_cors import CORS  # added late (other iteration)coz it found helps with browser security errors

# make our flask app
app = Flask(__name__)
CORS(app)  # lets the frontend talk to this backend without CORS errors, added after b/c it wasnt working without it during first run

# set up route that listens for POST reqs to /recommend
@app.route("/recommend", methods=["POST"])
def recommend():
    
    data = request.get_json()# get json data from the form

    # try to get student ID, if it's there
    student_id = data.get("StudentID")  # could be blank
    # build the student profile with age, ed level etc
    profile = {
        "AgeBand": data.get("AgeBand"),
        "HighestEducation": data.get("HighestEducation"),
        "Disability": data.get("Disability"),
        "Module": data.get("Module")
    }

    # if we gotyt an ID and it exists in the data we will use se the hybrid model
    if student_id and int(student_id) in interaction_normalized.index:
        raw_result = recommend_hybrid(student_id=int(student_id), profile=profile, top_n=10)
        source = "hybrid"
    else:# otherwise, if its the new student we will neeed to use cold start (demographics)
        raw_result = recommend_for_new_student(profile, top_n=10)
        source = "cold_start"

    # fix scores by subtracting some points for populer items. This was also added in the endswith
    # essentially the penalty function with the 0.3 weight that can be adjusted
    # maybe in the future have the weight from the front end to be customizable?
    reranked = apply_popularity_penalty(raw_result, activity_popularity, weight=0.3)
    # only keep the top 3 after we are reweighting
    final_result = dict(list(reranked.items())[:3])

    # make thee json to send back to the frontend
    response = {
        "model": source,
        "recommendations": [{"ActivityType": k, "Score": float(v)} for k, v in final_result.items()]
    }
    return jsonify(response)

# start ,. the flask server       on port 5000
if __name__ == "__main__":
    app.run(host='0.0.0.0',debug=True, port=5000)
