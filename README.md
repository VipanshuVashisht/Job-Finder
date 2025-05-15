# Job Finder: An Advanced Real-Time Job Recommender and Resume Analyzer

Welcome to the **Recruiter AI** project repository!

## Project Overview

This project focuses on implementing a resume screening and job recommendation system using a domain adaptation approach based on Encoder-Decoder Model i.e BERT. The primary goal is to extract latent features from job posts and resumes using BERT. Subsequently, the system classifies resumes based on the matched job roles.

## Modules

### 1. Predicting Category

- Description: Predict the category of a resume based on its content.
- Key Components:
  - Data loading and preprocessing
  - Glove Tokenizer
  - Message Passing Layer
  - BERT For Resume Classification which consists of 23 labels.

### 2. Matching Score Between Resume and Description

- Description: Compute a matching score between a given resume and a job description.
- Key Components:
  - Text preprocessing
  - Named Entity Extraction
  - Keyword Extraction
  - Score Computation

### 3. Job and Skill Recommendation System

- Description: Recommend relevant job roles and skills to candidates based on their resume content and the predicted job category.

### 4. Real-time Job Web Scraping

- Description: Continuously scrape up-to-date job listings from various websites, suitable for the predicted job category.
- Key Components:
  - Web scraping techniques
  - Data enrichment
 
### 5. Results

- Home Page
  ![image](https://github.com/user-attachments/assets/915f1636-ed81-438e-9f30-6d69277a8d90)

- Upload Resume and Job Description
  ![image](https://github.com/user-attachments/assets/cd4b7601-20d0-4319-bb8f-13b1d58f57ab)

- Predicted Result
  ![image](https://github.com/user-attachments/assets/c1cb2a62-09d8-4c1d-8e25-3cf3f06d478d)

- Skill Recommendation
  ![image](https://github.com/user-attachments/assets/c0b97575-7590-4d80-85f7-52a18a010013)

- Job Opportunities
  ![image](https://github.com/user-attachments/assets/0231bc3b-d036-401a-9449-771558f1c819)





