Data set (too large to upload) can be found on https://www.kaggle.com/datasets/anlgrbz/student-demographics-online-education-dataoulad/data


E-Learning Recommender System – Deployment & Usage Guide


This guide will help you build, run, and access the hybrid recommender system (backend + frontend) using Docker.

----------------------------------------------------!!!!!!!!!!!!!!!!!!!!!!!!IMPORTANT!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!-------------------------------------

Due to file size constraints on github studentVLE must be downloaded and placed in the backend/ file for the code to work

The URL to download the file: https://www.kaggle.com/datasets/anlgrbz/student-demographics-online-education-dataoulad?select=studentVle.csv


--------------------------------------------------------------------------------------------------------------------------


<pre> ## 1. Project Structure (Expected Layout) ``` RecommenderElearning/ ├── backend/ │ ├── api.py │ ├── recommender.py │ ├── recommend_hybrid.py │ ├── knn_recommender.py │ ├── requirements.txt │ ├── studentInfo.csv │ ├── studentVle.csv │ ├── vle.csv │ └── Dockerfile ├── frontend/ │ ├── bin/ │ ├── Controllers/ │ ├── Models/ │ ├── obj/ │ ├── Properties/ │ ├── Services/ │ ├── wwwroot/ │ ├── .dockerignore │ ├── appsettings.json │ ├── appsettings.Development.json │ ├── Dockerfile │ ├── Program.cs │ ├── RecommenderAPI.csproj │ ├── RecommenderAPI.csproj.user │ ├── RecommenderAPI.http │ ├── RecommenderAPI.sln │ └── WeatherForecast.cs └── docker-compose.yml ``` </pre>

2. Prerequisites
----------------
- Docker installed and running
- (RRecommended) Docker Desktop with WSL2 backend configured with at least 4 GB of memory

3. How to Run the Project
-------------------------
Open terminal and run:

    docker-compose up --build

This builds and starts both backend (Flask) and frontend containers using Docker.

4. Access the Application
-------------------------
Once containers are running:

Open your browser and go to:
    http://localhost:8080

You’ll see the learning recommendation interface.

5. Making API Requests Manually (Optional)
------------------------------------------
Using curl:

    curl -X POST http://localhost:5000/recommend ^
         -H "Content-Type: application/json" ^
         -d "{\\"AgeBand\\": \\"35-55\\", \\"HighestEducation\\": \\"HE Qualification\\", \\"Disability\\": \\"N\\", \\"Module\\": \\"AAA\\"}"

6. Shutting Down
----------------
To stop the system, press Ctrl+C or run:

    docker-compose down

7. Troubleshooting
------------------
Issue: ERR_EMPTY_RESPONSE
Solution: Make sure backend is running with host='0.0.0.0'

Issue: Frontend can't reach backend
Solution: Ensure frontend uses http://backend:5000 in Docker context

Issue: Backend crashes on startup
Solution: Increase Docker memory in .wslconfig if using WSL2

Issue: Missing modules
Solution: Ensure all code files are copied in the Docker build stage
