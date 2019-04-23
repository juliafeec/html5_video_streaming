from app import application, classes, db
# from flask import Flask, render_template, send_from_directory, request, redirect
from flask import *
from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileRequired
from wtforms import SubmitField
from werkzeug import secure_filename
from flask_login import login_user

from boto.s3.key import Key
import boto


class UploadFileForm(FlaskForm):
    """Class for uploading file when submitted"""
    file_selector = FileField('File', validators=[FileRequired()])
    submit = SubmitField('Submit')

def upload(file):
    """upload a file from a client machine."""
    filename = secure_filename(file.filename)
    # filename : filename of FileField
    # secure_filename secures a filename before storing it directly on the filesystem.
    file_content = file.stream.read()

    bucket_name = 'msds603camera' # Change it to your bucket.
    s3_connection = boto.connect_s3(aws_access_key_id="AKIA2UZ37BVQN2YYZWFJ",
                                    aws_secret_access_key='w3H6P3eriY668gNttNxphHsm+0nwO2PwnRVWD77q')
    bucket = s3_connection.get_bucket(bucket_name)
    k = Key(bucket)
    k.key = filename
    k.set_contents_from_string(file_content)


@application.route('/new_user_form', methods=['GET', 'POST'])
def new_user_form():
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
	if request.method == "POST":
		name = request.form['name']
		imgFile = request.files['img_file']
		# we need to call upload here
		upload(imgFile)

	return render_template("main_page2.html")


@application.route('/', methods=['GET', 'POST'])
def login():
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

