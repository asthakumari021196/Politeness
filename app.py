
from flask import Flask, request, render_template
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import  naive_bayes, svm
import numpy as np
import mlflow
import mlflow.pyfunc

import data_polarity

# Declare a Flask app
app = Flask(__name__)

# Main function here
# ------------------

@app.route('/', methods=['GET', 'POST'])
def main():

    # If a form is submitted
    if request.method == "POST":
        # Unpickle classifier        
        # Get values through input bars
        print("enter Text")
        text = request.form.get("Text")
        sent = []
        sent.append(str(text))
        len_of_text = len(sent[0].split())

        mlflow.set_tracking_uri("http://127.0.0.1:5000")
        registery_uri = 'sqlite:///mlflow.db'
        mlflow.tracking.set_tracking_uri(registery_uri)

        # mlflow.set_experiment('case-study-one')
        model_svm  = mlflow.pyfunc.load_model(model_uri=f"models:/SVM/Staging")
        model_naive  = mlflow.pyfunc.load_model(model_uri=f"models:/NaiveBayes/Staging")


        if len_of_text < 3:
            err = 'Enter another sentence, sentence is too short'
            polarity = ""
            sent_naive = ""
            sent_svm = ""
     

        else:
            dp = data_polarity.Polarity()
            polarity = dp.calculatePolarity(sent[0])
            
            Tfidf_vect = pickle.load(open("vectorizer.pickle", "rb"))

            Test_X_Tfidf = Tfidf_vect.transform(sent)
            print("text_x_tfidf ", Test_X_Tfidf)


            # Get prediction
            prediction_svm = model_svm.predict(Test_X_Tfidf)
            prediction_naive = model_naive.predict(Test_X_Tfidf)

            err = ""
            if prediction_svm == 0:
                sent_svm = "impolite"
            else:
                sent_svm = "polite"    

            if prediction_naive == 0:
                sent_naive = "impolite"
            else:
                sent_naive = "polite"                 
                
    else:
        sent_naive = ""
        sent_svm = ""
        polarity = ""
        err = ""
        text = ""
        
    return render_template("website.html", text = text, sent_svm = sent_svm, polarity = polarity, sent_naive = sent_naive, err = err)
# Running the app
if __name__ == '__main__':
    app.run(host='0.0.0.0',port = 80,debug = True)
