import unittest
from flask import Flask,current_app, Response as BaseResponse, json
from flask.testing import FlaskClient
from flask_testing import TestCase
from werkzeug.utils import cached_property
import os
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
        self.app = self.create_app()
        c = self.app.test_client()
        c.post('/api/player', data=json.dumps(dict(name='Bubba')),follow_redirects=True)
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
        # print(response.content_length)
        assert response.content_length > 400
        # print(response.data)
        step_response = c.post('/api/step', data=json.dumps(dict(action='check',betsize=0)),follow_redirects=True)
        # print(response.content_length)
        assert response.content_length > 400
        # print(step_response.data)

    def test_hand(self):
        print('test bet call')
        c = self.app.test_client()
        response = c.get('/api/reset')
        step_response = c.post('/api/step', data=json.dumps(dict(action='bet',betsize=1)),follow_redirects=True)
        print(step_response.data)
        step_response = c.post('/api/step', data=json.dumps(dict(action='call',betsize=1)),follow_redirects=True)
        print(step_response.data)

class Response(BaseResponse):
    @cached_property
    def json(self):
        return json.loads(self.data)

class TestClient(FlaskClient):
    def open(self, *args, **kwargs):
        if 'json' in kwargs:
            kwargs['data'] = json.dumps(kwargs.pop('json'))
            kwargs['content_type'] = 'application/json'
        return super(TestClient, self).open(*args, **kwargs)

if __name__ == '__main__':
    unittest.main()