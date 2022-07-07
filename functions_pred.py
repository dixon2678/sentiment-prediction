import nltk
from nltk.corpus import stopwords
import pandas as pd
import fsspec
import gcsfs
import nltk
from google.cloud import storage
from google.oauth2 import service_account
import os
from sklearn.metrics import auc, accuracy_score, confusion_matrix, mean_squared_error, f1_score
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold, RandomizedSearchCV, train_test_split
import xgboost as xgb
import pickle
from google.cloud import bigquery
from sklearn.feature_extraction.text import CountVectorizer
from nltk.sentiment import SentimentIntensityAnalyzer
from datetime import datetime
import json
from nltk.stem.porter import *
nltk.download('punkt')
nltk.download('vader_lexicon')
nltk.download('stopwords')
import string
gcp_json_credentials_dict = json.loads(os.environ['creds'])
credentials = service_account.Credentials.from_service_account_info(gcp_json_credentials_dict)
client = storage.Client(credentials=credentials, project='final-347314')
bucket = client.get_bucket('csv-etl-fyp')

date = datetime.today().strftime('%Y-%m-%d')

def load_ml_model():
  blob = bucket.blob('model/ml/xgb-base.pkl')
  blob.download_to_filename('/loaded-xgb.pkl')
  loaded_file = open('/loaded-xgb.pkl', 'rb')
  model = pickle.load(loaded_file)
  return model

def load_prediction_data():
  client = bigquery.Client(credentials=credentials, project='final-347314')
  sql = """
  SELECT * FROM `final-347314.main.rss_coin_name`
  """
  df = client.query(sql).to_dataframe()
  return df

def load_vectorizer():
  blob = bucket.blob('model/cv/cv.pkl')
  blob.download_to_filename('/cv.pkl')
  loaded_file = open('/cv.pkl', 'rb')
  cv = pickle.load(loaded_file)
  return cv

def preprocessing_pipeline(data, cv):

  data = data.dropna()
  data = [x.lower() for x in data]

  st = PorterStemmer()
  data = [st.stem(x) for x in data]

  stop = set(stopwords.words('english'))
  data = [t for t in data if t not in stop and t not in string.punctuation]
  data = cv.transform(data)
  return data

def upload_to_gcs(path):
  
  blob = bucket.blob('model/prediction/prediction' + date + '.csv')
  blob.upload_from_filename(path)
