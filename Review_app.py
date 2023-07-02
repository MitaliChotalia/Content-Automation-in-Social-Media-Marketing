import streamlit as st
import pandas as pd
import re
import os
import base64
import transformers
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.lm import MLE, Vocabulary
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.tokenize import word_tokenize
import smtplib
from email.mime.text import MIMEText
from transformers import AutoTokenizer, AutoModelWithLMHead
from transformers import AutoTokenizer, AutoModel
from api import gettext
import torch
from sklearn.metrics.pairwise import cosine_similarity
import transformers
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from perplexity import calper
# from similarity import res






# Load sentiment analysis model
sentiment_analysis = pipeline("sentiment-analysis")

# Streamlit app
st.title('Email Response Generator')


similarity = 'bert-base-uncased'
tokenizer_sim = AutoTokenizer.from_pretrained(similarity)
model4 = AutoModel.from_pretrained(similarity)

# Define a function to calculate the similarity score between the review and response
def calculate_similarity_score(response, review):
    # Load the BERT model and tokenizer
    # Tokenize the review and response
    inputs = tokenizer_sim([response, review], padding=True, truncation=True, max_length=512, return_tensors='pt')
    # Pass the inputs through the model to get the output embeddings
    outputs = model4(**inputs)
    response_embedding = outputs.last_hidden_state[0][-1]
    review_embedding = outputs.last_hidden_state[0][-2]
    # Calculate the cosine similarity between the embeddings
    similarity_score = cosine_similarity([response_embedding.detach().numpy()], [review_embedding.detach().numpy()])[0][0]
    return similarity_score



def generate_email(email_body):
    
    # Read the email template from a file
    
    
    # Replace the placeholders in the email template with the user input
    # email_body = email_template.replace('[Recipient]', recipient_name)
    # email_body = email_body.replace('[Your Name]', sender_name)
    
    # Split the email body into sentences
    sentences = re.split(r'(?<=[^A-Z].[.?]) +(?=[A-Z])', email_body)
    
    # Find the position of the "Subject" and "Dear" keywords in the email body
    # subject_pos = email_body.find("Subject:")
    # dear_pos = email_body.find("Dear")
    
    # Separate the email body based on the "Subject" and "Dear" keywords
    # subject = email_body[subject_pos:dear_pos]
    # body = email_body[dear_pos:]
    body=email_body
    
    # Add line breaks after each sentence
    body = '\n'.join(sentences)
    image_path="D:/College/SEM VI/IPD/mark/Food Town.png"
    # msg.attach(MIMEText(body, 'plain'))
    # Add the image to the email
    # if image_path:
    #     with open(image_path, "rb") as image_file:
    #         encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    #     body += f'<br><img src="data:image/png;base64,{encoded_string}">'
    # filenames = ["D:/College/SEM VI/IPD/mark/Food Town.png"]
    # for filename in filenames:
    #     with open(filename, "rb") as attachment:
    #         part = MIMEBase('application', 'octet-stream')
    #         part.set_payload((attachment).read())
    #         encoders.encode_base64(part)
    #         part.add_header('Content-Disposition', "attachment; filename= %s" % filename)
    #         body.attach(part)
        # Predict sentiment
    sentiment = sentiment_analysis(review)[0]['label']
    
    # Select email response based on sentiment and keyword overlap
    review_keywords = set(review.split())
    if sentiment.lower()=='postive':
        # data_subset = data[data['sentiment'] == rev] 
        formatted_email = f"Thank you for your feedback.\nDear {name},\n\n{body}\n\nBest regards,\nThe Food Town"
    else:
        formatted_email = f"Dear {name},\n\n{body}\n\nBest regards,\nThe Food Town"

    return formatted_email

email = st.text_input("Enter your email address")
name = st.text_input("Enter your name")
# rev = st.selectbox('Response Type',('General Review','Quality Review','Service Review','Value Review','Hygiene Review','Management Review','Atmosphere Review'))
review = st.text_area("Enter your review here", height=200)
if st.button("Generate Email"):


    response=gettext(review)
    generated_text = response.choices[0].text.strip()
    print(generated_text)
    email_body = generate_email(generated_text)
    st.write(calper(email_body))
    st.write("Generated Email:")
    st.write(email_body)

    


    if email != "":
        sender_email = "bhagyashah45@gmail.com"
        sender_password = "szqcjnjphxvustzf"
        message = MIMEText(email_body)
        message['Subject'] = f"Thank you for your feedback."
        message['From'] = sender_email
        message['To'] = email
        
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login(sender_email, sender_password)
            smtp.send_message(message)
            st.write("Email sent to ", email)