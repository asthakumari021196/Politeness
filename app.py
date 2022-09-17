
from flask import Flask, request, render_template
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import  naive_bayes, svm
import numpy as np
import data_polarity

# Declare a Flask app
app = Flask(__name__)

# Main function here
# ------------------

@app.route('/', methods=['GET', 'POST'])
def main():
    sent = ""
    polarity = ""
    # If a form is submitted
    if request.method == "POST":
        # Unpickle classifier        
        # Get values through input bars
        print("enter Text")
        text = request.form.get("Text")
        sent = []
        sent.append(str(text))
        len_of_text = len(sent[0].split())
        if len_of_text < 3:
            sent = 'Enter another sentence, sentence is too short'
            

        else:
            dp = data_polarity.Polarity()
            polarity = dp.calculatePolarity(sent[0])
            
            Tfidf_vect = pickle.load(open("vectorizer.pickle", "rb"))

            Test_X_Tfidf = Tfidf_vect.transform(sent)
            print("text_x_tfidf ", Test_X_Tfidf)
            SVM  = pickle.load(open('svm.pkl', 'rb'))
            print("model ", SVM)
            print("Model loaded")

            # Get prediction
            prediction = SVM.predict(Test_X_Tfidf)
            if prediction == 0:
                sent = "impolite"
            else:
                sent = "polite"    
                
    else:
        sent = ""
        
    return render_template("website.html", sent = sent, polarity = polarity)
# Running the app
if __name__ == '__main__':
    app.run(debug = True)