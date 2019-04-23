import os
basedir = os.path.abspath(os.path.dirname(__file__))


class Config(object):
    SECRET_KEY = os.urandom(24)
    SQLALCHEMY_DATABASE_URI = 'sqlite:///' + \
        os.path.join(basedir, 'week3.db')
    # flask-login uses sessions which require a secret Key
    SQLALCHEMY_TRACK_MODIFICATIONS = True
