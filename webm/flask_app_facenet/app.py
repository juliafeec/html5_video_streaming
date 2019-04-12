from flask import Flask, render_template, send_from_directory, request
from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileRequired
from wtforms import SubmitField
from werkzeug import secure_filename

from boto.s3.key import Key
import boto

import redis

redis_db = redis.StrictRedis(host="localhost", port=6379, db=0)
app = Flask(__name__)

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
    s3_connection = boto.connect_s3(aws_access_key_id="AKIA2UZ37BVQGUF5O4XB",
                                    aws_secret_access_key='DuI84JbZtURkalRwyiy1yWUV2wvwR63jDp3kWf3b')
    bucket = s3_connection.get_bucket(bucket_name)
    k = Key(bucket)
    k.key = filename
    k.set_contents_from_string(file_content)


app = Flask(__name__)


@app.route('/main_page', methods=['GET', 'POST'])
def main_page():
	if request.method == "POST":
		print("Here, in POST!")
		name = request.form['name']
		imgFile = request.files['img_file']
		# we need to call upload here
		upload(imgFile)


	return render_template("main_page2.html")

@app.route("/svg")
def svg():
    # with open('overlay.xml', 'r') as f:
        # svg_overlay = f.read()
    svg_overlay = redis_db.get('overlay').decode('utf-8')
    svg_string = "data:image/svg+xml;utf8,"+svg_overlay
    return svg_string


@app.route('/', methods=['GET', 'POST'])
def login():
    return render_template("login.html")


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)
