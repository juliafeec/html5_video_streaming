from app import application, classes, db
from flask import *
from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileRequired
from wtforms import SubmitField
from werkzeug import secure_filename
from flask_login import login_user

from boto.s3.key import Key
import boto
import boto3
import os
from base64 import b64decode
import redis
import pickle
from collections import Counter
from user_definition import key_id, access_key

app = Flask(__name__)

redis_db = redis.StrictRedis(host="localhost", port=6379, db=0)

with open('extracted_dict.pickle','rb') as f:
    feature_dict = pickle.load(f)

counter = Counter([x.split("_")[0] for x in feature_dict.keys()])
del feature_dict


class UploadFileForm(FlaskForm):
    """Class for uploading file when submitted"""
    file_selector = FileField('File', validators=[FileRequired()])
    submit = SubmitField('Submit')


def upload(name, listfile):
    """upload a file from a client machine."""
    for file in listfile:
        filename = secure_filename(file.filename)
        file_content = file.stream.read()
    
    
        counter[name]+=1
        filename_extension = filename.rsplit(".")[1]
        filename = "{}_{}.{}".format(name, counter[name], filename_extension)
    
        with open("photos/" + filename, 'wb') as f:
            f.write(file_content)
      
        redis_db.set('create_embs', 1)
    
        bucket_name = 'msds603camera' # Change it to your bucket.
        s3_connection = boto.connect_s3(aws_access_key_id=key_id,
                                        aws_secret_access_key=access_key)
        bucket = s3_connection.get_bucket(bucket_name)
        k = Key(bucket)
        k.key = filename
        k.set_contents_from_string(file_content)
        key = bucket.lookup(filename)
        key.set_acl('public-read-write')


@application.route('/new_user_form', methods=['GET', 'POST'])
def new_user_form():
    """Let a new user to register"""
    if request.method == "POST":
        # when a user fills the new user form
        name = request.form['name']
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']

        # we look to check if the user is already created
        user = classes.User.query.filter_by(email=email).first()

        if user is not None:
            login_user(user)
            flash("Email already has an account associated.")
            return render_template("index_newform.html")
        else:
            # we create the user in the database
            user = classes.User(name, username, email, password)
            db.session.add(user)
            db.session.commit()

        return redirect(url_for("login"))

    return render_template("index_newform.html")


@application.route('/main_page', methods=['GET', 'POST'])
def main_page():
    """Main page of the application"""
    if request.method == "POST":
        name = request.form['name']
        listofFiles = request.files.getlist("img_file")
        # we need to call upload here
        upload(name, listofFiles)
    return render_template("main_page2.html")


@application.route('/list', methods=['GET', 'POST'])
def list_images():
    """Get a list of images to search"""
    bucket_name = 'msds603camera'
    s3_connection = boto.connect_s3()
    bucket = s3_connection.get_bucket(bucket_name)
    unsorted_keys = []
    for key in bucket.list('images/'):
        unsorted_keys.append([key.name, key.last_modified])
    return render_template('list.html', items=unsorted_keys)


@application.route('/image', methods=['POST'])
def image():
    print('capturing image from webcam')
    image_path = os.path.join(app.instance_path, 'images')
    js_image_data = request.form['image']
    header, encoded = js_image_data.split(",", 1)
    data = b64decode(encoded)
    # f_name = ('%s.jpeg' % time.strftime('%Y%m%d-%H%M%S'))
    f_name = '%s.jpeg' % 'detected_image'
    with open(os.path.join(image_path, f_name), 'wb') as f:
        f.write(data)
    return Response("%s saved" % f_name)


@application.route("/svg")
def svg():
    """Add the svg overlay to stream"""
    svg_overlay = redis_db.get('overlay').decode('utf-8')
    svg_string = "data:image/svg+xml;utf8,"+svg_overlay
    return svg_string


@application.route("/demo")
def demo():
    """Web for demo"""
    return render_template("demo.html")


@application.route("/test_stream")
def test_stream():
    """Web for test stream"""
    return render_template("main_page_demo.html")


@application.route('/', methods=['GET', 'POST'])
def login():
    """User login page"""
    if request.method == "POST":
        # we check if we have an existing or new user
        email = request.form['email']
        password = request.form['password']

        # Look for it in the database.
        user = classes.User.query.filter_by(email=email).first()

        # Login and validate the user.
        if user is not None and user.check_password(password):
            login_user(user)
            return redirect(url_for("main_page"))
        else:
            flash("Invalid username and password combination!")
            return render_template("login.html")

    return render_template("login.html")
