"""Unit test script"""

import pandas as pd
import unittest
import pickle
import json

from app import APP

class TestWebAPPEndpoints(unittest.TestCase):

    def setUp(self):
        self._app = APP.test_client()
        with open('test_resources/data.json','r') as f:
            self._data = json.loads(f.readline())
        with open('test_resources/score.json','r') as f:
            self._score = json.loads(f.readline())
    
    def test_200(self):
        '''test_200: a request for / shall return 200 OK'''
        response = self._app.get('/')
        self.assertEqual(response.status,'200 OK', 'Failed to submit GET to homepage')

    def test_404(self):
        '''test_404: a request for null shall return 404 NOT FOUND'''
        response = self._app.get('/null')
        self.assertEqual(response.status,'404 NOT FOUND','Failed to get 404 to bad request')
    
    def test_score(self):
        '''test_score: a request to score data should return expected output'''
        response = self._app.post('/score',json=self._data)
        print(response.data)

class TestModelOutput(unittest.TestCase):
    
    def setUp(self):
        with open('model/model.pkl','rb') as f:
            self._model = pickle.load(f)

        with open('test_resources/data.json','r') as f:
            records = json.loads(f.readline())
            self._data = pd.DataFrame(records)

        with open('test_resources/score.json','r') as f:
            self._expected_score = json.loads(f.readline())
    
    def test_model_output(self):
        '''test_model_output: Model should predict expected score'''
        score = self._model.predict(self._data)
        self.assertEqual(score,self._expected_score, 'Model did not output expected score')

if __name__ == '__main__':
    unittest.main()