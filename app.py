# from imports import *
# from sentence_transformers import SentenceTransformer
# from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
# from sklearn.metrics.pairwise import cosine_similarity

# app=Flask(__name__)
# sentence_model = SentenceTransformer('all-mpnet-base-v2')
# ner_pipeline = pipeline("ner")
# model_name = "bert-base-uncased"
# role_tokenizer = AutoTokenizer.from_pretrained(model_name)
# role_model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=25) # Adjust num_labels

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/predict', methods=['GET', 'POST'])
# def predict():
#     # Initialize variables with default values
#     Job_Role = ""
#     Score = "Needed Job Description"
#     fetched_data = []
#     skills = []
#     jobs = []

#     if request.method == 'POST':
#         try:
#             data = CustomData(
#                 Resume_file=request.files['Resume_file'],
#                 Job_description_file=request.files['Job_description_file']
#             )
#             Score = data.Savedata()

#             preprocess = PreprocessPipeline()
#             RoleModel_input = preprocess.run()

#             Modelpredict = ModelPipeline()
#             Job_Role = Modelpredict.predictrole(RoleModel_input)

#             Scorepredict = ScorePipeline()
#             ScoreModel_input = Scorepredict.scoreprocess()
#             Score = Scorepredict.predictscore(ScoreModel_input)

#             Recom = Recommend()
#             skills = Recom.Get_skills(Job_Role)
#             jobs = Recom.Get_jobs(Job_Role)

#             Webscrap = WebScraping()
#             fetched_data = Webscrap.GetList(Job_Role)

#         except Exception as e:
#             # Handle exceptions gracefully
#             return str(e), 500

#     return render_template(
#         "results.html",
#         Job_Role=Job_Role,
#         Score=Score,
#         fetched_data=fetched_data,
#         skills=skills,
#         jobs=jobs
#     )


# if __name__=="__main__":
#     app.run(host="0.0.0.0",debug=True)


from flask import Flask, request, render_template, jsonify
from sentence_transformers import SentenceTransformer
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics.pairwise import cosine_similarity
from src.components.data_transformation import Preprocessing
import os
import sys
import tempfile
import fitz  # PyMuPDF
import docxpy

# Import your pipeline classes
from src.pipeline.predict_pipeline import PreprocessPipeline, ModelPipeline, ScorePipeline, SkillPipeline, Recommend, WebScraping, CustomData

app = Flask(__name__)

# Load models outside of routes
sentence_model = SentenceTransformer('all-mpnet-base-v2')
ner_pipeline = pipeline("ner")
model_name = "bert-base-uncased"
role_tokenizer = AutoTokenizer.from_pretrained(model_name)
role_model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=25)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    Job_Role = ""
    Score = "Needed Job Description"
    fetched_data = []
    skills = []
    jobs = []

    if request.method == 'POST':
        try:
            resume_file = request.files['Resume_file']
            job_desc_file = request.files['Job_description_file']

            # --- Extract Text from Files ---
            preprocessor = Preprocessing()  # Use the class
            resume_text = preprocessor.extract_resume_text(resume_file)
            job_desc_text = preprocessor.extract_resume_text(job_desc_file) if job_desc_file else ""

            # --- Calculate Matching Score ---
            score_pipeline = ScorePipeline()
            Score = score_pipeline.predictscore(resume_text, job_desc_text) if job_desc_text else "Needed Job Description"

            # --- Extract Skills ---
            skill_pipeline = SkillPipeline()
            resume_skills = skill_pipeline.extract_skills(resume_text)
            job_desc_skills = skill_pipeline.extract_skills(job_desc_text)
            skills = list(set(resume_skills + job_desc_skills))

            # --- Predict Job Role ---
            preprocess_pipeline = PreprocessPipeline()
            role_model_input = preprocess_pipeline.run(resume_text)  # Pass text, not file object
            model_pipeline = ModelPipeline()
            Job_Role = model_pipeline.predictrole(role_model_input)

            # --- Get Job Recommendations ---
            recom = Recommend()
            jobs = recom.Get_jobs(Job_Role)

            # --- Web Scraping ---
            webscrap = WebScraping()
            fetched_data = webscrap.GetList(Job_Role)

            # --- Clean up ---
            custom_data = CustomData(Resume_file=None, Job_description_file=None)  # Pass None, the files are already read
            custom_data.Deletefiles()

            return render_template(
                "results.html",
                Job_Role=Job_Role,
                Score=Score,
                fetched_data=fetched_data,
                skills=skills,
                jobs=jobs
            )
        except Exception as e:
            error_message = f"An error occurred: {str(e)}"
            print(error_message)  # Log the error
            return jsonify(error=error_message), 500  # Return a JSON error response

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
