import os
from flask import Flask, flash, request, redirect, url_for, send_from_directory, render_template, jsonify
import json
from werkzeug.utils import secure_filename
from sudoku import image_to_matrix, solve_sudoku

UPLOAD_FOLDER = 'uploads/'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('uploaded_file',
                                    filename=filename))
    return render_template('home.html')


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    matrix = image_to_matrix(UPLOAD_FOLDER + filename)
    matrix_normal = [ls.tolist() for ls in matrix]
    return render_template('sudoku.html', matrix=json.dumps(matrix_normal))

@app.route('/solve', methods=['POST'])
def solve():
    string = request.form['matrix']
    return jsonify(solve_sudoku(string))
app.run(threaded=False)