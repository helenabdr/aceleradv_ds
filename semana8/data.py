import pandas as pd

class DataSource:
    
    def __init__(self):
        self.path_train = '/home/helena/codenation/enem-2/train.csv'
        self.path_test = 'test.csv'
        #self.path_label = 'answer.csv'

    def read_data(self, etapa_treino=True):
        '''
            Read data from data sources
            :param etapa_treino: Boolean specifing if is train or test.
            :return: pd.DataFrame with values and pd.Series with labels
        '''

        if etapa_treino:
            df = pd.read_csv(self.path_train)
            return df

        df = pd.read_csv(self.path_test)
        
        num_inscricao = df.NU_INSCRICAO
        
        return num_inscricao, df