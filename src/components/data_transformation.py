from imports import *
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

class Preprocessing:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))

    def extract_resume_text(self, resume_file):
        """
        Extracts text from a resume file (PDF or DOCX).  Handles file objects.
        """
        text = ""
        try:
            if resume_file.filename.lower().endswith(".pdf"):
                import fitz
                file_content = resume_file.read()
                with fitz.open(stream=file_content, filetype="pdf") as doc:
                    for page in doc:
                        text += page.get_text()
                resume_file.seek(0)
            elif resume_file.filename.lower().endswith((".doc", ".docx")):
                import docxpy
                temp_file = tempfile.NamedTemporaryFile(delete=False)
                resume_file.save(temp_file.name)
                text = docxpy.process(temp_file.name)
                os.unlink(temp_file.name)
            else:
                return ""
            return text
        except Exception as e:
            print(f"Error extracting text: {e}")
            return ""

    def preprocess_text(self, text):
        text = re.sub(r'[^\w\s]', '', text)
        text = text.lower()
        tokens = word_tokenize(text)
        tokens = [word for word in tokens if word not in self.stop_words]
        return tokens

    def pos_filter(self, tokens):
        tagged_tokens = nltk.pos_tag(tokens)
        filtered_tokens = [
            word for word, pos in tagged_tokens
            if pos.startswith('NN') or pos == 'JJ' or pos == 'NNP'
        ]
        return filtered_tokens

    def extract_named_entities(text):
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(text)
        named_entities = [ent.text for ent in doc.ents if ent.label_ != "CARDINAL" and ent.label_ != "DATE"]
        processed_list = []
        for word in named_entities:
            processed_word = word.replace('\n', '').lower()
            processed_list.append(processed_word)
        return named_entities
    
    def extract_keywords(text):
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(text)
        keywords = []
        for token in doc:
            if token.pos_ in ["NOUN", "PROPN", "ADJ"]:
                lemma = token.lemma_.lower().strip()
                keywords.append(lemma)
        return keywords