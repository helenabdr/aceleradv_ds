import pandas as pd
from data import DataSource
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE
from sklearn.tree import DecisionTreeRegressor

class Preprocessing:

    def _init_(self):
        self.data = DataSource()

    def process(self, df, etapa_treino = True):

        columns = ['CO_UF_RESIDENCIA', 'SG_UF_RESIDENCIA', 'NU_IDADE',
       'TP_SEXO', 'TP_COR_RACA', 'TP_NACIONALIDADE', 'TP_ST_CONCLUSAO',
       'TP_ANO_CONCLUIU', 'TP_ESCOLA', 'TP_ENSINO', 'IN_TREINEIRO',
       'TP_DEPENDENCIA_ADM_ESC', 'TP_PRESENCA_CN', 'TP_PRESENCA_CH', 'TP_PRESENCA_LC',
       'CO_PROVA_CN', 'CO_PROVA_CH', 'CO_PROVA_LC', 'CO_PROVA_MT', 'NU_NOTA_MT',
       'NU_NOTA_CN', 'NU_NOTA_CH', 'NU_NOTA_LC', 'TP_LINGUA',
       'TP_STATUS_REDACAO', 'NU_NOTA_COMP1', 'NU_NOTA_COMP2', 'NU_NOTA_COMP3',
       'NU_NOTA_COMP4', 'NU_NOTA_COMP5', 'NU_NOTA_REDACAO', 'Q001', 'Q002',
       'Q006', 'Q024', 'Q025', 'Q026', 'Q027', 'Q047']

        print('Creating Dataframe for Data Manipulation')

        if etapa_treino:
            df = df[columns]

        cons = pd.DataFrame({'column': df.columns,
                            'missing_perc': (df.isna().sum() / df.shape[0])*100,
                            'dtype': df.dtypes })

        print(cons)

        #Remove columns with high percentage of missing data
        cons = cons[cons['missing_perc'] < 50.00]
        df = df[cons.column]

        #Select only numeric columns
        df = df.select_dtypes(include = 'number')

        '''
        Try 1:
        
        for x in df.columns:
            df[x].fillna(df[x].mean(), inplace = True)
        '''
        
        #Try 2
        df.fillna(0, inplace = True)
        
        if etapa_treino:

            #Separating out the features
            X_train = df.drop(['NU_NOTA_MT'], axis=1)

            #Separating out the target
            y_train = df['NU_NOTA_MT']

            print('Selecting the features ...')

            #Apply RFE
            selector = RFE(DecisionTreeRegressor(), n_features_to_select=5)
            selector = selector.fit(X_train, y_train)

            selected = []

            #Summarize all features
            for i, j in zip(X_train, range(len(X_train))):

                if selector.support_[j]:
                    selected.append(i)

            print('Selected features:', selected)

            return X_train[selected], y_train, selected

        else:

            return df

