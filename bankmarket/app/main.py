import pandas as pd
from flask import Flask, request, jsonify
import pickle

from bankmarket.scripts.train import X

le_fpath = '/Users/gayathrigurram/Desktop/logreg/bankmarket/model/label_encoder.pkl'
ohe_fpath = '/Users/gayathrigurram/Desktop/logreg/bankmarket/model/ohe_encoder.pkl'
model_fpath = '/Users/gayathrigurram/Desktop/logreg/bankmarket/model/log_reg_model.pkl'

with open(le_fpath,'rb') as le_file:
    label_encoder = pickle.load(le_file)

with open(ohe_fpath,'rb') as ohe_file:
    ohe_encoder = pickle.load(ohe_file)

with open(model_fpath,'rb') as model_file:
    model = pickle.load(model_file)

app = Flask(__name__)

@app.route('/')

def home():
    return "Welcome to bank marketing API!"

@app.route('/bankmarketingpredict', methods=['GET'])

def get_prediction():
    try:
      features = request.args.to_dict()
      input_data = pd.DataFrame([features])

      print("Received Query Parameters:",features)
      print("Converted data frame:",input_data)

      input_data = input_data.reindex(columns= X.columns, fill_value=0)
      prediction = model.predict(input_data)
      prediction_prob = model.predict_proba(input_data)[:,1]
      return jsonify(
          {
              "status": "success",
              "input_features": features,
              "prediction": int(prediction[0]),
              "prediction_probablity": float(prediction_prob[0])
          }
      ),200
    except Exception as e:
        return jsonify(
            {
                "status": "error",
                "message": str(e)
            }
        ),500

if __name__ == '__main__':
    app.run(debug=True)
