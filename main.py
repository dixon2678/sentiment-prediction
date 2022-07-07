import requests
import pandas as pd
import os
from functions_pred import load_ml_model, load_prediction_data, load_vectorizer, preprocessing_pipeline, upload_to_gcs
from datetime import datetime
from google.cloud import bigquery
from flask import Flask
date = datetime.today().strftime('%Y-%m-%d')
app = Flask(__name__)

@app.route("/")
def el_job():
    model = load_ml_model()
    df = load_prediction_data()
    cv = load_vectorizer()
    final_df = preprocessing_pipeline(df['title'], cv)
    pred = model.predict(final_df)
    df['pred'] = pred
    path = 'prediction' + date + '.csv'
    df.to_csv(path)
    upload_to_gcs(path)
  
# Main
if __name__ == "__main__":
    print("Started")
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
 
