import pandas as pd
import time

from sklearn.preprocessing import LabelEncoder
from sklearn import model_selection, naive_bayes, svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

from sklearn.feature_extraction.text import TfidfVectorizer

import mlflow
import mlflow.sklearn
import pickle

from urllib.parse import urlparse

# import tensorflow as tf
# from tensorflow.keras import regularizers


class PolitenessDetector:
    def __init__(self,filename) -> None:
        dataRead = pd.read_csv(filename)
        df = pd.DataFrame(dataRead)
        self.data = df.head(200000)

    def NaiveBayesModel(self):
        Naive = naive_bayes.MultinomialNB()
        Naive.fit(self.Train_X_Tfidf,self.Train_Y)
        pickle.dump(Naive, open('naive.pkl', 'wb'))
        mlflow.log_param("test data shape", self.Test_X_Tfidf.shape[0])


        # predict the labels on validation dataset
        predictions_NB = Naive.predict(self.Test_X_Tfidf)
        
        # Use accuracy_score function to get the accuracy
        acc_score = accuracy_score(predictions_NB, self.Test_Y)*100
        preci_score = precision_score(predictions_NB, self.Test_Y)
        recall = recall_score(predictions_NB, self.Test_Y)
        f1 = f1_score(predictions_NB, self.Test_Y)

        mlflow.log_metric("accuracy_naive",acc_score )
        mlflow.log_metric("precision_naive",preci_score )
        mlflow.log_metric("recall_naive",recall )
        mlflow.log_metric("f1_naive",f1 )


        print("Naive Bayes Accuracy Score -> ",acc_score)  
        print("Naive Bayes Precision Score -> ",preci_score)  
        print("Naive Bayes Recall Score -> ",recall)  
        print("Naive Bayes F1 Score -> ",f1)  

        return Naive

    def SVMModel(self):
        # Classifier - Algorithm - SVM
        # fit the training dataset on the classifier
        SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
        print('model fitting')

        SVM.fit(self.Train_X_Tfidf,self.Train_Y)
        pickle.dump(SVM, open('svm.pkl', 'wb'))

        mlflow.log_param("test data shape", self.Test_X_Tfidf.shape[0])
        # predict the labels on validation dataset
        print('prediction started')


        predictions_SVM = SVM.predict(self.Test_X_Tfidf)
        
        # Use accuracy_score function to get the accuracy
        acc_score = accuracy_score(predictions_SVM, self.Test_Y)*100
        preci_score = precision_score(predictions_SVM, self.Test_Y)
        recall = recall_score(predictions_SVM, self.Test_Y)
        f1 = f1_score(predictions_SVM, self.Test_Y)

        mlflow.log_metric("accuracy_svm",acc_score )
        mlflow.log_metric("precision_svm",preci_score )
        mlflow.log_metric("recall_svm",recall )
        mlflow.log_metric("f1_svm",f1 )

        print("SVM Accuracy Score -> ",acc_score)  
        print("SVM Precision Score -> ",preci_score)  
        print("SVM Recall Score -> ",recall)  
        print("SVM F1 Score -> ",f1)  

        return SVM
        

    def detect(self):
        df = self.data
        Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(df['txt'],df['polite'],test_size=0.3)
        Encoder = LabelEncoder()
        Train_Y = Encoder.fit_transform(Train_Y)
        Test_Y = Encoder.fit_transform(Test_Y)
        # sample = ["open the box and put cheese in it", "can you close the door", "It is appreciated if it could be improved","email this to the tax office","pass me the book"," Thank You for the help"," would you like some coffee ?"]
        # sampley = [0,0,1,0,0,1,1]
        Tfidf_vect = TfidfVectorizer(max_features=5000)
        Tfidf_vect.fit(df['txt'].astype(str))
        pickle.dump(Tfidf_vect, open("vectorizer.pickle", "wb"))

        Train_X_Tfidf = Tfidf_vect.transform(Train_X.astype(str))
        Test_X_Tfidf = Tfidf_vect.transform(Test_X.astype(str))
        # sample_X_TFidf = Tfidf_vect.transform(sample)

        self.Train_X_Tfidf = Train_X_Tfidf
        self.Test_X_Tfidf = Test_X_Tfidf
        # self.Test_X_Tfidf = sample_X_TFidf

        self.Train_Y = Train_Y
        self.Test_Y = Test_Y
        # self.Test_Y = sampley

        print("test",Train_X_Tfidf.shape)

        # print("test",Test_X_Tfidf.shape)
                # (1) update the data in the running real time
        mlflow.set_tracking_uri("http://127.0.0.1:5000")

        # (2) this is model registry
        registery_uri = 'sqlite:///mlflow.db'

        # (3) update the date in the real time from the above sqlitedb
        mlflow.tracking.set_tracking_uri(registery_uri)
        
        # Running the MLflow -> SVM Model and Naive Bayes
        with mlflow.start_run():
            print("model run")
            model1 = self.SVMModel()
            model2 =self.NaiveBayesModel()


            tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

            # Model registry does not work with file store
            if tracking_url_type_store != "file":

            # Register the model
            # There are other ways to use the Model Registry, which depends on the use case,
            # please refer to the doc for more information:
            # https://mlflow.org/docs/latest/model-registry.html#api-workflow
                mlflow.sklearn.log_model(model1, "model_1", registered_model_name="SVM")
                mlflow.sklearn.log_model(model2, "model_2", registered_model_name="NaiveBayes")

            else:
                mlflow.sklearn.log_model(model1, "model1")    
                mlflow.sklearn.log_model(model2, "model2")    





file = "C:/Users/kasth/github/Politeness/cleaned_file.csv"
d = PolitenessDetector(file)   
start_time = time.time()
d.detect()
print('execution time')
print("--- %s seconds ---" % (time.time() - start_time))