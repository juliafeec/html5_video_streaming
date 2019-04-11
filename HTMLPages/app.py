from flask import Flask, render_template, send_from_directory, request

app = Flask(__name__)

@app.route('/main_page', methods=['GET', 'POST'])
def main_page():
    return render_template("main_page2.html")

@app.route('/login', methods=['GET', 'POST'])
def login():
    return render_template("login.html")

@app.route('/handle_form_data', methods=['POST'])
def handle_form_data():
    if request.method == "POST":
       name = request.form['name']
       print(name)
      # print(request.files)
      # upload(imagepath)
    
    return render_template("main_page2.html")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)
