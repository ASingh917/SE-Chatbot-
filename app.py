import os
import pandas as pd
from flask import Flask, render_template, request, jsonify
from difflib import get_close_matches
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Initialize Flask app
app = Flask(__name__)

# Load the dataset
df = pd.read_csv('legal_advisory_faq_extended.csv')

# TF-IDF Vectorizer for question matching
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df["Question"])

# Function to suggest corrections
def suggest_question(user_query, questions):
    suggestions = get_close_matches(user_query, questions, n=3, cutoff=0.3)
    if suggestions:
        return f"Did you mean: {', '.join(suggestions)}?"
    else:
        return "I couldn't find a close match. Please rephrase your question."

# Chatbot function
def chatbot(query):
    query_vec = vectorizer.transform([query])
    similarity = cosine_similarity(query_vec, tfidf_matrix)
    index = similarity.argmax()
    if similarity.max() > 0.3:  # Threshold for relevance
        response = df.iloc[index]["Answer"]
        return response
    else:
        questions = df["Question"].tolist()
        suggestion = suggest_question(query, questions)
        return suggestion

# Home route to serve the UI
@app.route('/')
def home():
    return render_template('index.html')

# API endpoint to handle chatbot queries
@app.route('/ask', methods=['POST'])
def ask():
    user_query = request.json.get("query")
    if user_query:
        response = chatbot(user_query)
        return jsonify({"response": response})
    else:
        return jsonify({"response": "Please provide a query."}), 400

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
