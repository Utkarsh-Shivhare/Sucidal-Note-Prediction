import pickle
from flask import Flask, request, jsonify
import string
import nltk
from nltk.corpus import stopwords
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from flask_cors import CORS  # import the CORS extension

def modi(x):
    x = str(x).lower()
    x = re.sub(r'http\://[a-zA-Z0-9\-\.]+\.[a-zA-Z]{2,3}(/\S*)',"webaddress",x)
    x = re.sub(r'^\(?[\d](3)\)?[\s.]?[\d](3)\)?[\s.]?[\d](4)\)$','phonenumber',x)
    x = re.sub(r'\d+(\.\d+)?','num',x)
    x = re.sub(r'[^\w\d\s]','',x)
    x = re.sub(r'^\S+?$',"",x)
    stop_words = set(stopwords.words('english'))
    x_processed = lambda x: " ".join(term for term in x.split() if term not in stop_words)
    x = x_processed(x)
    return x

df= pd.read_csv("sucidal.csv")

# Load the trained model from the pickle file
with open('sucidal text prediction', 'rb') as f:
    model = pickle.load(f)

# Initialize a Flask app
app = Flask(__name__)
CORS(app)  # add the CORS extension to the app

# Define a route for the API endpoint
@app.route('/predict', methods=['POST'])
def predict():
    # Get the text input from the request
    text = request.json['text']
    
    # Preprocess the text input (e.g. remove stop words, URLs, etc.)
    text=modi(text)
    df["text"]=df["text"].apply(lambda x: modi(x))

    # Convert the preprocessed text into numerical features
    tf_vec=TfidfVectorizer(max_features=12713,ngram_range=(1,3),analyzer='char')
    pre=tf_vec.fit_transform(df["text"])
    vec=tf_vec.transform([text])
    

    # Make a prediction using the trained model
    prediction = model.predict(vec)
    print(prediction)
    prediction=prediction[0]
    print(vec.reshape(1,-1))

    # Return the prediction as a JSON response
    response = {'prediction': bool(prediction)}
    print("prediction:",prediction)
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)

