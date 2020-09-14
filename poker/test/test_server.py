import unittest
import os
from flask import Flask,current_app, Response as BaseResponse, json
from flask_testing import TestCase
from server import app

class BaseConfiguration(object):
    DEBUG = False
    TESTING = False
    MONGO_PATH = 'mongodb://' + '172.16.0.3' + ':27017/'

class TestConfiguration(BaseConfiguration):
    TESTING = True

class BaseTestCase(TestCase):
    """A base test case for flask-tracking."""

    def create_app(self):
        app.config.from_object(TestConfiguration())
        app.config['TESTING'] = True
        app.config['WTF_CSRF_ENABLED'] = False
        return app

    def setUp(self):
        print('Testing Server')

        self.app = self.create_app()
        c = self.app.test_client()
        c.post('/api/player/name', data=json.dumps(dict(name='Bubba')),follow_redirects=True)
        # db.create_all()

    def tearDown(self):
        pass
    #     db.session.remove()
    #     db.drop_all()

    def test_main_page(self):
        c = self.app.test_client()
        response = c.get('/health', follow_redirects=True)
        self.assertEqual(response.status_code, 200)
        response = c.get('/garbage', follow_redirects=True)
        self.assertEqual(response.status_code, 404)
        response = c.get('/api/reset')
        assert response.content_length > 400
        # step_response = c.post('/api/step', data=json.dumps(dict(action='check',betsize=0)),follow_redirects=True)
        # assert response.content_length > 400
        # step_response = c.post('/api/step', data=json.dumps(dict(action='check',betsize=0)),follow_redirects=True)

    # def test_hand(self):
    #     print('test bet call')
    #     c = self.app.test_client()
    #     response = c.get('/api/reset')
        # step_response = c.post('/api/step', data=json.dumps(dict(action='bet',betsize=1)),follow_redirects=True)
        # print(step_response.data)
        # step_response = c.post('/api/step', data=json.dumps(dict(action='call',betsize=1)),follow_redirects=True)
        # print(step_response.data)

    def test_player_results(self):
        c = self.app.test_client()
        response = c.get('/api/player/stats')
        print(response.data)

    def test_model_load(self):
        c = self.app.test_client()
        response = c.post('/api/model/load',data=json.dumps(dict(path=os.path.join(os.getcwd(),'checkpoints/RL_actor'))),follow_redirects=True)
        print(response.data)

if __name__ == '__main__':
    unittest.main()