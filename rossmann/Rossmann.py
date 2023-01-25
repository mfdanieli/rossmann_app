import pickle
import inflection
import pandas as pd
import numpy as np
import math
import datetime

class Rossmann( object ):
    def __init__( self ):
#         self.home_path=''
        self.competition_distance_scaler   = pickle.load( open( 'parameter/competition_distance_scaler.pkl', 'rb') )
        self.competition_since_scaler      = pickle.load( open( 'parameter/competition_since_scaler.pkl', 'rb') )
        self.year_scaler                   = pickle.load( open( 'parameter/year_scaler.pkl', 'rb') )
        self.store_type_scaler             = pickle.load( open( 'parameter/store_type_scaler.pkl', 'rb') )
        
        
    def data_cleaning( self, df1 ): 
        
        ## 1.1. Rename Columns
        cols_new = []
        for cols in df1.columns:
            col_aux = inflection.underscore(cols)
            cols_new.append(col_aux)

        # rename
        df1.columns = cols_new

        ## 1.3. Data Types
        # correct date type (from object to datetime)
        df1['date'] = pd.to_datetime( df1['date'] )

        ## 1.5. Fillout NA
        #competition_distance        
        df1['competition_distance'] = df1['competition_distance'].fillna(200000)

        df1['competition_open_since_month'] = df1['competition_open_since_month'].fillna(df1['date'].dt.month)

        df1['competition_open_since_year'] = df1['competition_open_since_year'].fillna(df1['date'].dt.year)

 
        return df1 


    def feature_engineering( self, df2 ):

        # year
        df2['year'] = df2['date'].dt.year

        # month
        df2['month'] = df2['date'].dt.month

        # day
        df2['day'] = df2['date'].dt.day

        # year week >> to see temporal evolution per week
        df2['year_week'] = df2['date'].dt.strftime( '%Y-%W' )

        # week of year
        df2['week_of_year'] = df2['date'].dt.weekofyear

        # -----------
        # competition since -> months since competition is open:

        # a. pegando month e year como string para concatenar no jeito mais simples
        df2['competition_open_since_month'] = df2['competition_open_since_month'].astype(int)
        df2['competition_open_since_year'] = df2['competition_open_since_year'].astype(int)
        df2['competition_open_since_month'] = df2['competition_open_since_month'].astype(str)
        df2['competition_open_since_year'] = df2['competition_open_since_year'].astype(str)
        df2['competition_open_since'] =df2['competition_open_since_month'] + '-' + df2['competition_open_since_year'] 
        # b. como o resultado no passo anterior é objeto, preciso passar p/ date
        df2['competition_open_since'] = pd.to_datetime( df2['competition_open_since'])

        # c. Getting the diff between to dates in form of months
        df2['competition_since'] = ((df2.date - df2.competition_open_since)/np.timedelta64(1, 'M'))


        # replacing a,b... with names of categories
        # assortment
        df2['assortment'] = df2['assortment'].apply( lambda x: 'basic' if x == 'a' else 'extra' if x == 'b' else 'extended' )

        # state holiday
        df2['state_holiday'] = df2['state_holiday'].apply( lambda x: 'public_holiday' if x == 'a' else 'easter_holiday' if x == 'b' else 'christmas' if x == 'c' else 'regular_day' )

        # 3.0. PASSO 03 - FILTRAGEM DE VARIÁVEIS
        ## 3.1. Filtragem das Linhas

        ## 3.2. Selecao das Colunas
        cols_drop = ['open']
        df2 = df2.drop( cols_drop, axis=1 )
        
        return df2


    def data_preparation( self, df5 ):

        ## 5.2. Rescaling 
        # competition distance
        df5['competition_distance'] = self.competition_distance_scaler.fit_transform( df5[['competition_distance']].values )
    
        # competition since
        df5['competition_since'] = self.competition_since_scaler.fit_transform( df5[['competition_since']].values )

        # year
        df5['year'] = self.year_scaler.fit_transform( df5[['year']].values )

        ### 5.3.1. Encoding
        # state_holiday - One Hot Encoding
        df5 = pd.get_dummies( df5, prefix=['state_holiday'], columns=['state_holiday'] )

        # store_type - Label Encoding
        df5['store_type'] = self.store_type_scaler.fit_transform( df5['store_type'] )

        # assortment - Ordinal Encoding
        assortment_dict = {'basic': 1,  'extra': 2, 'extended': 3}
        df5['assortment'] = df5['assortment'].map( assortment_dict )

        
        ### 5.3.3. Nature Transformation
        # day of week
        df5['day_of_week_sin'] = df5['day_of_week'].apply( lambda x: np.sin( x * ( 2. * np.pi/7 ) ) )
        df5['day_of_week_cos'] = df5['day_of_week'].apply( lambda x: np.cos( x * ( 2. * np.pi/7 ) ) )

        # month
        df5['month_sin'] = df5['month'].apply( lambda x: np.sin( x * ( 2. * np.pi/12 ) ) )
        df5['month_cos'] = df5['month'].apply( lambda x: np.cos( x * ( 2. * np.pi/12 ) ) )

        # day 
        df5['day_sin'] = df5['day'].apply( lambda x: np.sin( x * ( 2. * np.pi/30 ) ) )
        df5['day_cos'] = df5['day'].apply( lambda x: np.cos( x * ( 2. * np.pi/30 ) ) )

        # week of year
        df5['week_of_year_sin'] = df5['week_of_year'].apply( lambda x: np.sin( x * ( 2. * np.pi/52 ) ) )
        df5['week_of_year_cos'] = df5['week_of_year'].apply( lambda x: np.cos( x * ( 2. * np.pi/52 ) ) )
        
        
        cols_selected = ['store', 'promo', 'store_type', 'assortment', 'competition_distance', 'promo2', 'competition_since', 'day_of_week_sin',  'day_of_week_cos', 'month_cos', 'month_sin', 'day_sin', 'day_cos', 'week_of_year_cos','week_of_year_sin']
        
        return df5[ cols_selected ]
    
    
    def get_prediction( self, model, original_data, test_data ):
        # prediction
        pred = model.predict( test_data )
        
        # join pred into the original data
        original_data['prediction'] = np.expm1( pred )
        
        return original_data.to_json( orient='records', date_format='iso' )