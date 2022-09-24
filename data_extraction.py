import pandas as pd
import time

import data_cleaning
import data_polarity
# import impolite_detection


class DataSentiment():
    def __init__(self,filename) -> None:
        dataRead = pd.read_csv(filename)
        df = pd.DataFrame(dataRead)
        self.data = df

    def describe_data(self,tag):
        print("the no. of records are : ",self.data.shape)    
        print("the polarity distribution is this : ", self.data[[{tag}]].value_counts())

    def data_transform(self):
        self.data = self.data[self.data["is_useful"] == 1]
        self.data = self.data[['txt','p_tag']]
        self.data.drop_duplicates(subset=['txt'])
        # insuring that null values are not present
        self.data = self.data[pd.notnull(self.data)]
        dc = data_cleaning.DataCleaning()
        new_df = dc.data_preprocessing(self.data)

        dp = data_polarity.Polarity()
        new_df['polarity']  = new_df[['txt']].applymap(lambda x: dp.calculatePolarity(x))

        new_df.to_csv('cleaned_file.csv')
        print('Value Counts: ')
        print(new_df['polarity'].value_counts())
       
        # print(self.data['polarity'].value_counts())
        
        # self.data['p_id_tag'] = self.data['p_tag'].apply(lambda x: int(x.replace("P_", "")))
        # X,y = dc.data_preprocessing(self.data)
        # X_smote,y_smote = dc.data_preprocessing(self.data)

        # id = impolite_detection.PolitenessDetector(X_smote,y_smote)
        # id.detector()
        return new_df





fil = "C:/Users/kasth/github/Politeness/data/politeness.csv"
d = DataSentiment(fil)   
start_time = time.time()
d.data_transform() 
print('execution time')
print("--- %s seconds ---" % (time.time() - start_time))







# 1.3 million rows