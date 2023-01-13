from flask import Flask,request,render_template
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import pandas as pd
import regex as re
import os


app = Flask(__name__)
model_path = os.getcwd()+'/Review_Score/model.pkl'
vector_path = os.getcwd()+'/Review_Score/vector.pkl'
model = pickle.load(open(model_path, 'rb'))
#from sklearn.feature_extraction.text import TfidfVectorizer

#vector = TfidfVectorizer(ngram_range=(1,2))
vector = pickle.load(open(vector_path,'rb'))
lemmatizer = WordNetLemmatizer()
nltk_stopwords = set(stopwords.words('english'))

def preprocess(raw_text, flag):
    #Removes html tags
    # Removing special characters and digits
    sentence = re.sub("<.*?>|[^a-zA-Z]", " ", raw_text)
    
    # change sentence to lower case
    sentence = sentence.lower()

    # tokenize into words
    tokens = sentence.split()
    
    # remove stop words                
    clean_tokens = [t for t in tokens if t not in nltk_stopwords]
    clean_tokens = [lemmatizer.lemmatize(word) for word in clean_tokens]
    return pd.Series([" ".join(clean_tokens)])

@app.route('/')
def home():
	return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    txt = request.form['review']
    txt = preprocess(txt,"lemm")
    my_prediction = model.predict(vector.transform(txt))[0]
        
    return render_template('results.html', prediction=my_prediction,txt =request.form['review'] )
    
if __name__ == '__main__':
    app.run(debug=True)


