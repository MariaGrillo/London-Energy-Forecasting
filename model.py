import pandas as pd
import numpy as np
from fbprophet import Prophet

class EnergyModel:

    def __init__(self):

        self.model = None

    def preprocess_training_data(self, df):

        df = df.rename(columns={'day':'ds', 'consumption':'y'})
        df.ds = pd.to_datetime(df.ds)
        return df, None
        

    def fit(self, X, y):

        m = Prophet(weekly_seasonality=False).add_seasonality(name='Weekly', period=7, fourier_order=15)
        self.model = m.fit(X)
        

    def preprocess_unseen_data(self, df):

        df = df.rename(columns={'day':'ds', 'consumption':'y'})
        df.ds = pd.to_datetime(df.ds)
        return df

    def predict(self, X):

        df_dates = self.model.make_future_dataframe(periods=27, include_history=True)
        predictions = self.model.predict( df_dates )
        return predictions['yhat'][-27:]
