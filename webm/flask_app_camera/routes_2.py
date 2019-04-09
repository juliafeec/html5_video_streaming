from app import application
from flask import render_template, redirect, url_for
from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileRequired
from wtforms import SubmitField
from werkzeug import secure_filename

from boto.s3.key import Key
import boto


class UploadFileForm(FlaskForm):
    """Class for uploading file when submitted"""
    file_selector = FileField('File', validators=[FileRequired()])
    submit = SubmitField('Submit')


@application.route('/index')
@application.route('/')
def index():
    """Index Page : Renders index.html with author name."""
    #return ("<h1> Hello World </h1>")
    return(render_template("index.html", author='camera'))

#export AWS_ACCESS_KEY_ID=<YOUR REAL ACCESS KEY>
#export AWS_SECRET_ACCESS_KEY=<YOUR REAL SECRET KEY>
#from os import environ as os_env
#
#ses = boto.ses.connect_to_region(
#   'us-west-2',
#    aws_access_key_id=os_env['AWS_ACCESS_KEY_ID'],
#    aws_secret_access_key=os_env['AWS_SECRET_ACCESS_KEY']'
#)
@application.route('/upload', methods=['GET', 'POST'])
def upload():
    """upload a file from a client machine."""
    file = UploadFileForm()  # file : UploadFileForm class instance
    if file.validate_on_submit():  # Check if it is a POST request and if it is valid.
        f = file.file_selector.data  # f : Data of FileField
        filename = secure_filename(f.filename)
        # filename : filename of FileField
        # secure_filename secures a filename before storing it directly on the filesystem.
        file_content = f.stream.read()

        bucket_name = 'msds603camera' # Change it to your bucket.
        s3_connection = boto.connect_s3(aws_access_key_id="AKIA2UZ37BVQGUF5O4XB",
                                        aws_secret_access_key='DuI84JbZtURkalRwyiy1yWUV2wvwR63jDp3kWf3b')
        bucket = s3_connection.get_bucket(bucket_name)
        k = Key(bucket)
        k.key = filename
        k.set_contents_from_string(file_content)

        return redirect(url_for('index'))  # Redirect to / (/index) page.
    return render_template('upload.html', form=file)