# from imports import *
# import nltk
# import ast

# nltk.download('punk_tab')
# nltk.download('punkt')

# # Check if the stopwords resource is available
# from nltk.corpus import stopwords  # This should print the list of English stopwords
# nltk.download('stopwords')

# class PreprocessPipeline:
#     def __init__(self):
#         pass

#     def run(self):
#         try:
#             resume_path = os.path.join("artifacts", "Resume_file.pdf")
#             preprocessor = Preprocessing()
#             Text = preprocessor.extract_resume_text(resume_path)
#             # if(Text==""): print("Why not given")
#             Tokens = preprocessor.preprocess_text(Text)
#             Model_input = preprocessor.pos_filter(Tokens)
#             return Model_input
#         except Exception as e:
#             raise CustomException(e,sys)

# class ModelPipeline:
#     def __init__(self):
#         pass

#     def predictrole(self,Model_input):
#         try:
#             labels = {6: 'Data Science', 12: 'HR', 0: 'Advocate', 1: 'Arts', 24: 'Web Designing', 16: 'Mechanical Engineer', 22: 'Sales', 14: 'Health and fitness', 5: 'Civil Engineer', 15: 'Java Developer', 4: 'Business Analyst', 21: 'SAP Developer', 2: 'Automation Testing', 11: 'Electrical Engineering', 18: 'Operations Manager', 20: 'Python Developer', 8: 'DevOps Engineer', 17: 'Network Security Engineer', 19: 'PMO', 7: 'Database', 13: 'Hadoop', 10: 'ETL Developer', 9: 'DotNet Developer', 3: 'Blockchain', 23: 'Testing'}
#             # tokenizer = AutoTokenizer.from_pretrained("./models/")
#             # model = AutoModelForSequenceClassification.from_pretrained("./models/")
#             model_name = "bert-base-uncased" 
#             tokenizer = AutoTokenizer.from_pretrained(model_name)
#             model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(labels))
#             tokens = tokenizer.encode_plus(Model_input,max_length=512, truncation=True,padding="max_length",return_tensors="pt")
#             outputs = model(**tokens)
#             predicted_label = outputs.logits.argmax().item()
#             output=labels[predicted_label]
#             return output
#         except Exception as e:
#             raise CustomException(e,sys)


# class ScorePipeline:
#     def __init__(self):
#         pass
    
#     def scoreprocess(self):
#         try:
#             resume_path = os.path.join("artifacts", "Resume_file.pdf")
#             job_description_path = os.path.join("artifacts", "Job_description_file.pdf")
#             preprocessor = Preprocessing()
#             Resume_Text = preprocessor.extract_resume_text(resume_path)
#             Job_description_Text = preprocessor.extract_resume_text(job_description_path)
#             # Resume_Named_Entities = preprocessor.extract_named_entities(Resume_Text)
#             # job_description_keywords = preprocessor.extract_keywords(Job_description_Text)
#             Score_Model_input=[Resume_Text,Job_description_Text]
#             return Score_Model_input
#         except Exception as e:
#             raise CustomException(e,sys)

#     def predictscore(self,Score_Model_input):
#         try:
#             ob = CountVectorizer()
#             matrix = ob.fit_transform(Score_Model_input)
#             similarity_matrix =  cosine_similarity(matrix)
#             score = similarity_matrix[0][1] * 100
#             score = round(score,2)
#             return score
#         except Exception as e:
#             raise CustomException(e,sys)


# class Recommend:
#     def __init__(self):
#         self.Role_recommen_file = pd.read_csv("artifacts/Recom.csv")

#     def Get_skills(self, Job_Role):
#         category_data = self.Role_recommen_file[self.Role_recommen_file['Category'] == Job_Role]
#         skills = ast.literal_eval(category_data['Skills'].iloc[0])[:5]
#         return skills

#     def Get_jobs(self, Job_Role):
#         category_data = self.Role_recommen_file[self.Role_recommen_file['Category'] == Job_Role]
#         job_roles = ast.literal_eval(category_data['Job_Roles'].iloc[0])[:5]
#         return job_roles

# class WebScraping:
#     def __init__(self):
#         pass

#     def internshala(self,job_url):
#         try:
#             job_url = requests.get(job_url)
#             soup = BeautifulSoup(job_url.content, 'html.parser')
#             job_cards = soup.find_all('div', class_='individual_internship')
#             jobs_list = []
#             for card in job_cards:

#                 job_title_elem = card.find('h3', class_='heading_4_5 profile')
#                 company_name_elem = card.find('h4', class_='heading_6 company_name')
#                 location_elem = card.find('a', class_='location_link')
#                 start_date_elem = card.find('div', id='start-date-first')
#                 salary = card.find('div', class_='salary')
#                 experience = card.find('div', class_='desktop-text')
#                 apply = card.find('a')

#                 if not all([job_title_elem, company_name_elem, location_elem, start_date_elem, salary, experience]):
#                     continue

#                 job_title = job_title_elem.text.strip()
#                 company_name = company_name_elem.text.strip()
#                 location = location_elem.text.strip()
#                 start_date = start_date_elem.text.replace('Starts\xa0','').strip()
#                 salary = salary.text.strip()
#                 experience = experience.text.strip()
#                 apply = "https://internshala.com" + apply['href']

#                 job_info = {
#                     'Job_Title': job_title,
#                     'Company_Name': company_name,
#                     'Location': location,
#                     'Start_Date': start_date,
#                     'Salary': salary,
#                     'experience': experience,
#                     'apply' : apply
#                 }
#                 jobs_list.append(job_info)
#             return jobs_list
#         except Exception as e:
#             raise CustomException(e,sys)
        
    
#     def fresherworld(self,job_url,Job_Role):
#         try:
#             r = requests.get(job_url)
#             soup = BeautifulSoup(r.content, 'html.parser')
#             job_cards = soup.find_all('div', class_='col-md-12 col-lg-12 col-xs-12 padding-none job-container jobs-on-hover top_space')
#             jobs_list = []
#             for card in job_cards:
#                 job_title_elem = card.find('span', class_='wrap-title seo_title')
#                 company_name_elem = card.find('h3', class_='latest-jobs-title font-16 margin-none inline-block company-name')
#                 location_elem = card.find('span', class_='job-location display-block modal-open job-details-span')
#                 start_date_elem = card.find('span', class_='desc')
#                 apply = card.get('job_display_url')
#                 experience = card.find('span', class_='experience job-details-span')
#                 if not all([job_title_elem, company_name_elem, location_elem, start_date_elem, apply, experience]):
#                     continue
#                 # job_title = Job_Role
#                 # job_title = job_title_elem.text.strip()
#                 # company_name = "amzur technologies"
#                 company_name = company_name_elem.text.strip()
#                 location = location_elem.text.strip()
#                 start_date = start_date_elem.text.strip()
#                 experience = experience.text.strip()
#                 salary="Not Mentioned"
#                 job_info = {
#                     'Job_Title': Job_Role,
#                     'Company_Name': company_name,
#                     'Location': location,
#                     'Start_Date': start_date,
#                     'Salary': salary,
#                     'experience': experience,
#                     'apply': apply
#                 }
#                 jobs_list.append(job_info)
#             return jobs_list
#         except Exception as e:
#             raise CustomException(e,sys)
        
#     def GetList(self,Job_Role):
#         try:
#             Job_Role=Job_Role.replace(' ','-').lower()+"-jobs/"
#             jobs_list_internshala=[]
#             # jobs_list_fresherworld=[]

#             job_url_string_internshala='https://internshala.com/jobs/'+Job_Role
#             jobs_list_internshala=self.internshala(job_url_string_internshala)
#             # job_url_string_fresherworld='https://www.freshersworld.com/jobs/jobsearch/python-developer-jobs-for-be-btech?course=16'
#             # job_url_string_fresherworld='https://www.freshersworld.com/jobs/jobsearch/'+Job_Role+'-jobs-for-be-btech?course=16'
#             # jobs_list_fresherworld=self.fresherworld(job_url_string_fresherworld,Job_Role)
#             # print(len(jobs_list_fresherworld))
#             # print(jobs_list_freskoherworld[0])
#             return jobs_list_internshala
        
#         except Exception as e:
#             raise CustomException(e,sys)



# class CustomData:
#     def __init__(self, Resume_file, Job_description_file):
#         self.Resume_file = Resume_file
#         self.Job_description_file = Job_description_file

#     def Savedata(self):
#         self.Resume_file.save(os.path.join("artifacts", "Resume_file.pdf"))
#         if self.Job_description_file:
#             self.Job_description_file.save(os.path.join("artifacts", "Job_description_file.pdf"))
#         else:
#             return "Job_description_file is not uploaded ..... !"
#         return ""
    
#     def Deletefiles(self):
#         try:
#             if os.path.exists(os.path.join("artifacts", "Resume_file.pdf")):
#                 os.remove(os.path.join("artifacts", "Resume_file.pdf"))
#             if os.path.exists(os.path.join("artifacts", "Job_description_file.pdf")):
#                 os.remove(os.path.join("artifacts", "Job_description_file.pdf"))    
#         except Exception as e:
#             raise CustomException(e,sys)


# if __name__ == "__main__":
#     pass

import os
import sys
import ast
import requests
from sentence_transformers import SentenceTransformer
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from src.exception import CustomException  # Make sure this exists
import fitz  # PyMuPDF
import docxpy
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from bs4 import BeautifulSoup

# nltk.download('punkt')  # Uncomment these if you haven't downloaded the data
# nltk.download('stopwords')
# nltk.download('averaged_perceptron_tagger')  # Required for pos_filter


class Preprocessing:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))

    def extract_resume_text(self, resume_file):
        """
        Extracts text from a resume file (PDF or DOCX).  Handles file objects.

        Args:
            resume_file:  A file object (from Flask request.files)

        Returns:
            str: The extracted text, or an empty string on error.
        """
        text = ""
        try:
            if resume_file.filename.lower().endswith(".pdf"):
                # Read the file content from the FileStorage object
                file_content = resume_file.read()
                with fitz.open(stream=file_content, filetype="pdf") as doc:
                    for page in doc:
                        text += page.get_text()
                resume_file.seek(0)
            elif resume_file.filename.lower().endswith((".doc", ".docx")):
                import tempfile

                temp_file = tempfile.NamedTemporaryFile(delete=False)
                resume_file.save(temp_file.name)  # Save to a temp file
                text = docxpy.process(temp_file.name)
                os.unlink(temp_file.name)  # Delete the temp file
            else:
                return ""  # Handle other file types or no file

            # Reset the file pointer to the beginning after reading
            resume_file.seek(0)
            return text
        except Exception as e:
            print(f"Error extracting text: {e}")
            return ""

    def preprocess_text(self, text):
        """
        Preprocesses the text by removing punctuation, converting to lowercase,
        removing stop words, and tokenizing.

        Args:
            text (str): The text to preprocess.

        Returns:
            list: A list of cleaned tokens.
        """
        text = re.sub(r'[^\w\s]', '', text)
        text = text.lower()
        tokens = word_tokenize(text)
        tokens = [word for word in tokens if word not in self.stop_words]
        return tokens

    def pos_filter(self, tokens):
        """
        Filters tokens based on their part-of-speech tags, keeping only nouns,
        proper nouns, and adjectives.

        Args:
            tokens (list): A list of tokens.

        Returns:
            list: A list of filtered tokens.
        """
        tagged_tokens = nltk.pos_tag(tokens)
        filtered_tokens = [
            word for word, pos in tagged_tokens
            if pos.startswith('NN') or pos == 'JJ' or pos == 'NNP'
        ]
        return filtered_tokens


class PreprocessPipeline:
    def __init__(self):
        self.preprocessor = Preprocessing()

    def run(self, resume_text):
        try:
            tokens = self.preprocessor.preprocess_text(resume_text)
            model_input = " ".join(self.preprocessor.pos_filter(tokens))  # Join tokens into a string
            return model_input
        except Exception as e:
            raise CustomException(e, sys)


class ModelPipeline:
    def __init__(self):
        self.model_name = "bert-base-uncased"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, num_labels=25
        )  # Adjust based on your labels

    def predictrole(self, model_input):
        try:
            labels = {
                6: 'Data Science', 12: 'HR', 0: 'Advocate', 1: 'Arts', 24: 'Web Designing',
                16: 'Mechanical Engineer', 22: 'Sales', 14: 'Health and fitness',
                5: 'Civil Engineer', 15: 'Java Developer', 4: 'Business Analyst',
                21: 'SAP Developer', 2: 'Automation Testing', 11: 'Electrical Engineering',
                18: 'Operations Manager', 20: 'Python Developer', 8: 'DevOps Engineer',
                17: 'Network Security Engineer', 19: 'PMO', 7: 'Database', 13: 'Hadoop',
                10: 'ETL Developer', 9: 'DotNet Developer', 3: 'Blockchain', 23: 'Testing'
            }
            tokens = self.tokenizer.encode_plus(model_input, max_length=512,
                                                truncation=True, padding="max_length",
                                                return_tensors="pt")
            outputs = self.model(**tokens)
            predicted_label = outputs.logits.argmax().item()
            output = labels[predicted_label]
            return output
        except Exception as e:
            raise CustomException(e, sys)


class ScorePipeline:
    def __init__(self):
        self.sentence_model = SentenceTransformer('all-mpnet-base-v2')

    def predictscore(self, resume_text, job_description_text):
        try:
            resume_embedding = self.sentence_model.encode(resume_text)
            job_desc_embedding = self.sentence_model.encode(job_description_text)
            similarity_score = cosine_similarity([resume_embedding], [job_desc_embedding])[0][
                0]
            score = round(similarity_score * 100, 2)
            return score
        except Exception as e:
            raise CustomException(e, sys)


class SkillPipeline:
    def __init__(self):
        self.ner_pipeline = pipeline("ner")

    def extract_skills(self, text):
        try:
            ner_results = self.ner_pipeline(text)
            skills = set(
                entity['word'] for entity in ner_results
                if entity['entity'].startswith('SKILL') or entity['entity'].startswith('TECH')
            )
            return list(skills)
        except Exception as e:
            raise CustomException(e, sys)


class Recommend:
    def __init__(self):
        self.Role_recommen_file = pd.read_csv(
            "artifacts/Recom.csv")  # You might want to load this in app.py and pass it here.

    def Get_skills(self, Job_Role):
        try:
            category_data = self.Role_recommen_file[self.Role_recommen_file['Category'] == Job_Role]
            if not category_data.empty:
                skills = ast.literal_eval(category_data['Skills'].iloc[0])[:5]
                return skills
            else:
                return []
        except Exception as e:
            print(f"Error in Get_skills: {e}")
            return []

    def Get_jobs(self, Job_Role):
        try:
            category_data = self.Role_recommen_file[self.Role_recommen_file['Category'] == Job_Role]
            if not category_data.empty:
                job_roles = ast.literal_eval(category_data['Job_Roles'].iloc[0])[:5]
                return job_roles
            else:
                return []
        except Exception as e:
            print(f"Error in Get_jobs: {e}")
            return []


class WebScraping:
    def __init__(self):
        pass

    def internshala(self, job_url):
        try:
            job_url_response = requests.get(
                job_url)  # Changed variable name to avoid shadowing
            soup = BeautifulSoup(job_url_response.content, 'html.parser')
            job_cards = soup.find_all('div', class_='individual_internship')
            jobs_list = []
            for card in job_cards:
                job_title_elem = card.find('h3', class_='heading_4_5 profile')
                company_name_elem = card.find('h4', class_='heading_6 company_name')
                location_elem = card.find('a', class_='location_link')
                start_date_elem = card.find('div', id='start-date-first')
                salary = card.find('div', class_='salary')
                experience = card.find('div', class_='desktop-text')
                apply = card.find('a')

                if not all(
                    [job_title_elem, company_name_elem, location_elem, start_date_elem, salary,
                     experience, apply]):
                    continue

                job_title = job_title_elem.text.strip()
                company_name = company_name_elem.text.strip()
                location = location_elem.text.strip()
                start_date = start_date_elem.text.replace('Starts\xa0', '').strip()
                salary_text = salary.text.strip()
                experience_text = experience.text.strip()
                apply_url = "https://internshala.com" + apply['href']

                job_info = {
                    'Job_Title': job_title,
                    'Company_Name': company_name,
                    'Location': location,
                    'Start_Date': start_date,
                    'Salary': salary_text,
                    'experience': experience_text,
                    'apply': apply_url
                }
                jobs_list.append(job_info)
            return jobs_list
        except Exception as e:
            raise CustomException(e, sys)

    def fresherworld(self, job_url, job_role):
        try:
            response = requests.get(job_url)  # Changed variable name
            soup = BeautifulSoup(response.content, 'html.parser')
            job_cards = soup.find_all('div',
                                     class_='col-md-12 col-lg-12 col-xs-12 padding-none job-container jobs-on-hover top_space')
            jobs_list = []
            for card in job_cards:
                job_title_elem = card.find('span', class_='wrap-title seo_title')
                company_name_elem = card.find(
                    'h3', class_='latest-jobs-title font-16 margin-none inline-block company-name')
                location_elem = card.find(
                    'span', class_='job-location display-block modal-open job-details-span')
                start_date_elem = card.find('span', class_='desc')
                apply = card.get('job_display_url')
                experience = card.find('span', class_='experience job-details-span')
                if not all([job_title_elem, company_name_elem, location_elem, start_date_elem, apply,
                            experience]):
                    continue
                company_name = company_name_elem.text.strip()
                location = location_elem.text.strip()
                start_date = start_date_elem.text.strip()
                experience_text = experience.text.strip()
                salary = "Not Mentioned"
                job_info = {
                    'Job_Title': job_role,
                    'Company_Name': company_name,
                    'Location': location,
                    'Start_Date': start_date,
                    'Salary': salary,
                    'experience': experience_text,
                    'apply': apply
                }
                jobs_list.append(job_info)
            return jobs_list
        except Exception as e:
            raise CustomException(e, sys)

    def GetList(self, job_role):
        try:
            job_role_url = job_role.replace(' ', '-').lower() + "-jobs/"
            jobs_list_internshala = []
            job_url_string_internshala = 'https://internshala.com/jobs/' + job_role_url
            jobs_list_internshala = self.internshala(job_url_string_internshala)
            return jobs_list_internshala
        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    def __init__(self, Resume_file, Job_description_file):
        self.Resume_file = Resume_file
        self.Job_description_file = Job_description_file

    def Savedata(self):
        try:
            if self.Resume_file:
                resume_path = os.path.join("artifacts", "Resume_file.pdf")
                self.Resume_file.save(resume_path)
            else:
                resume_path = None
            if self.Job_description_file:
                job_description_path = os.path.join("artifacts", "Job_description_file.pdf")
                self.Job_description_file.save(job_description_path)
            else:
                job_description_path = None
            return resume_path, job_description_path  # Return file paths
        except Exception as e:
            raise CustomException(e, sys)

    def Deletefiles(self):
        try:
            if os.path.exists(os.path.join("artifacts", "Resume_file.pdf")):
                os.remove(os.path.join("artifacts", "Resume_file.pdf"))
            if os.path.exists(os.path.join("artifacts", "Job_description_file.pdf")):
                os.remove(os.path.join("artifacts", "Job_description_file.pdf"))
        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    pass