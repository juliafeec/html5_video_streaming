from flask import Flask, render_template, send_from_directory, request
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


def upload(file):
    """upload a file from a client machine."""
    filename = secure_filename(file.filename)
    file_content = file.stream.read()

    bucket_name = 'msds603camera'
    key_id = "AKIA2UZ37BVQGUF5O4XB"
    acc_key = "DuI84JbZtURkalRwyiy1yWUV2wvwR63jDp3kWf3b"
    s3_connection = boto.connect_s3(aws_access_key_id=key_id,
                                    aws_secret_access_key=acc_key)
    bucket = s3_connection.get_bucket(bucket_name)
    k = Key(bucket)
    k.key = filename
    k.set_contents_from_string(file_content)
    key = bucket.lookup(filename)
    key.set_acl('public-read-write')

app = Flask(__name__)


@app.route('/main_page', methods=['GET', 'POST'])
def main_page():
	if request.method == "POST":
		print("Here, in POST!")
		name = request.form['name']
		imgFile = request.files['img_file']
		upload(imgFile)
	return render_template("main_page2.html")

@app.route('/', methods=['GET', 'POST'])
def login():
    return render_template("login.html")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)
