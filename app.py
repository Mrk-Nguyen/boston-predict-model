'''Main web app'''
import pickle
import logging
import os
import json

import pandas as pd
from flask import Flask, request, jsonify

try:
    with open('model/model.pkl', 'rb') as f:
        MODEL = pickle.load(f)
        MODEL_LOADED = True
except OSError:
    logging.exception("Cannot open model")
    MODEL_LOADED = False

with open('model/feature_sequence.txt', 'r') as f:
    COLUMNS = json.loads(f.readline())

PORT = int(os.environ.get('PORT', 8080))

APP = Flask(__name__)


@APP.route('/', methods=['GET'])
def server_check():
    '''Homepage Route'''
    return "Server is up and running."


@APP.route('/score', methods=['POST'])
def score():
    '''Score Endpoint: Accepts JSON data in the format of:

    [{
        "CRIM": 0.5405,
        "ZN": 20.0,
        "INDUS": 3.97,
        "CHAS": 0.0,
        "NOX": 0.575,
        "RM": 7.47,
        "AGE": 52.6,
        "DIS": 2.872,
        "TAX": 264.0,
        "PTRATIO": 13.0,
        "B": 390.3,
        "LSTAT": 3.16
    }]

    '''
    content = request.json
    try:
        data_df = pd.DataFrame(content)[COLUMNS]
    except SystemError:
        logging.exception("Invalid data submitted")
        return jsonify(status='error, invalid data submitted', score=-1)

    if MODEL_LOADED:
        predicted_score = MODEL.predict(data_df)
        return jsonify(status='ok', scores=list(predicted_score))

    return jsonify(status='error, model failed to load', score=-1)


if __name__=='__main__':
    APP.run( debug=True, host='0.0.0.0', port=PORT )
