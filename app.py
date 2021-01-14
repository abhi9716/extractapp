import os
from flask import Flask, request, redirect, url_for, render_template, send_from_directory,jsonify
from werkzeug.utils import secure_filename
# from PyPDF2 import PdfFileReader, PdfFileWriter
from process import get_data
from collections import defaultdict 
UPLOAD_FOLDER = os.path.dirname(os.path.abspath(__file__)) + '/uploads/'

ALLOWED_EXTENSIONS = {'pdf', 'txt'}

app = Flask(__name__, static_url_path="/static")
DIR_PATH = os.path.dirname(os.path.realpath(__file__))
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# limit upload size upto 8mb
app.config['MAX_CONTENT_LENGTH'] = 8 * 1024 * 1024


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            print('No file attached in request')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            print('No file selected')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            print(file.filename)
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for("processpdf",filename=filename))
    
    return render_template('index.html')



@app.route('/processpdf/<filename>/result',methods=['GET', 'POST'])
def processpdf(filename):
    text=get_data(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    # print(text)
    return jsonify(text)


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8000))
    app.run(host='0.0.0.0', port=port)
