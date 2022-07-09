
import pandas as pd
from textblob import TextBlob




class Polarity():
    
    def getPolarity(self,text):
        return TextBlob(text).sentiment.polarity 

    def getAnalysis(self,score):
        if score < 0:
            return 'negative'

        elif score == 0:
            return 'neutral'  

        else:
            return 'positive'


    def calculatePolarity(self,text):
        polarityScore = self.getPolarity(text)
        polarity= self.getAnalysis(polarityScore)

        return  polarity


