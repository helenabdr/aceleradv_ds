import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline

from model_training import ModelTraining
from preprocessing import Preprocessing
from data import DataSource
from metrics import Metrics

class ModelInference:
    
    def __init__(self):
        self.model = ModelTraining()
        self.data = DataSource()
        self.preprocessing = Preprocessing()

    def predict(self):
        '''
        Predict values using model trained.
        :return: pd.Series with predicted values.
        '''
        
        print('Loading Data')
        num_inscricao, test_df = self.data.read_data(etapa_treino=False)
        
        print('Preprocessing Data')
        X_test = self.preprocessing.process(test_df, etapa_treino=False)
        
        #Predict y result
        print('Predicting')
        
        #Call the trained model
        model, features = self.model.model_training()
        
        #Predict the y
        y_pred = model.predict(X_test[features])
        
        #Create dataframe that receive the notes
        print('Saving Files')
        df_answer = pd.DataFrame({'NU_INSCRICAO': num_inscricao, 'NU_NOTA_MT': y_pred}).to_csv('answer.csv', index=False)

        return y_pred
    