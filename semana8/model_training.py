from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import pandas as pd

from data import DataSource
from preprocessing import Preprocessing

class ModelTraining:
    
    def __init__(self):
        self.data = DataSource()
        self.preprocessing = Preprocessing()

    def model_training(self):
        
        '''
        Train the model
        '''
        
        pre = Preprocessing()
        print('Loading data')
        
        df = self.data.read_data(etapa_treino = True)
        
        print('Training preprocessing')
        #Dataset splited and processed
        X, y, features = pre.process(df, etapa_treino = True)
        
        #Standardized with scaler
        scaler = StandardScaler()
        scaled = scaler.fit_transform(X, y)
    
        #Create model
        linear_regression_model = linear_model.LinearRegression()
        rf = RandomForestRegressor()
        
        #Train data
        model = rf.fit(X, y)
        
        return model, features
        
        '''
        #Prepare the model with input scaling
        pipeline = Pipeline(steps=[('scaler', StandardScaler()), ('model', linear_regression_model)])
        # fit pipeline
        pipe = pipeline.fit(X, y)
        '''