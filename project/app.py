from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
from flask_sqlalchemy import SQLAlchemy
import requests
import os
import base64
import uuid
import re
import logging
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = "greenmed"
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///greenmed.db'  # SQLite URI
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)


class User(db.Model):
    user_id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(255), nullable=False)
    password = db.Column(db.String(20), nullable=False)
    cpassword = db.Column(db.String(20), nullable=False)


class Cam(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    cam = db.Column(db.BLOB, nullable=False)
    date = db.Column(db.DateTime, nullable=False, default=db.func.now())


def create_db():
    with app.app_context():
        db.create_all()


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def predict_image_fastapi(image_data, model_version):
    port_mapping = {'v5': 8000, 'v8': 8001, 'v9': 8002}  # Mapping of model versions to port numbers
    port = port_mapping.get(model_version)
    print(port)
    if port is None:
        flash("Invalid model version selected", "error")
        return None

    url = f"http://localhost:{port}/predict"
    files = {'image': image_data}
    response = requests.post(url, files=files)

    if response.status_code == 200:
        prediction = response.json()
        print(prediction)
        return prediction
    
    else:
        flash('Failed to get predictions from FastAPI')
        return None


@app.route('/')
def index():
    if 'username' in session:
        return render_template('home.html', username=session['username'])
    else:
        return redirect(url_for('login'))


@app.route('/', methods=['GET', 'POST'])
def home():
    return render_template('home.html')


@app.route('/browse', methods=['GET', 'POST'])
def browse():
    if request.method == 'POST':
        model_version = request.form['model_version']
        if 'my_image' in request.files:
            file = request.files['my_image']
            if file.filename != '':
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                with open(filepath, 'rb') as f:
                    image_data = f.read()
                predictions = predict_image_fastapi(image_data, model_version)
                if predictions:
                    if 'predictions' in predictions and predictions['predictions']:
                        predicted_class = predictions['predictions'][0]['leaf_name']
                        session['filename'] = filename
                        session['predicted_class'] = predicted_class
                    else:
                        predicted_class = 'Undefined class'
                        session['filename'] = filename
                        session['predicted_class'] = predicted_class
                    return redirect(url_for('predict'))
                else:
                    flash('Failed to get predictions from FastAPI')
                    return redirect(url_for('home'))
        flash('No file part')
        return redirect(request.url)
    return render_template('browse.html')


@app.route('/upload_webcam', methods=['POST'])
def upload_webcam():
    # Get the model version from the form
    model_version = request.form.get('model_version')
    if not model_version:
        return jsonify({'error': 'Model version not provided'}), 400

    # Get the snapshot from the form and convert it to a file-like object
    snapshot_data_uri = request.form.get('snapshot')
    if not snapshot_data_uri:
        return jsonify({'error': 'Snapshot data not provided'}), 400

    # Regular expression to extract base64 data
    data_uri_pattern = re.compile(r'base64,(.*)$')
    image_data_match = data_uri_pattern.search(snapshot_data_uri)

    if image_data_match:
        # Decode the base64 image data
        image_data = base64.b64decode(image_data_match.group(1))

        # Generate a unique filename
        unique_filename = str(uuid.uuid4()) + ".jpg"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)

        # Save the decoded image data to a file
        with open(filepath, 'wb') as f:
            f.write(image_data)

        # Use the model version for prediction logic
        predictions = predict_image_fastapi(image_data, model_version)
        if predictions:
            if 'predictions' in predictions and predictions['predictions']:
                    predicted_class = predictions['predictions'][0]['leaf_name']
                    session['filename'] = unique_filename
                    session['predicted_class'] = predicted_class
            else:
                predicted_class = 'Undefined class'
                session['filename'] = unique_filename
                session['predicted_class'] = predicted_class
            return redirect(url_for('predict'))
        else:
            flash('Failed to get predictions from FastAPI')
            return redirect(url_for('home'))

    flash('No file part')
    return jsonify({'error': 'Invalid snapshot data'}), 400

@app.route('/predict')
def predict():
    filename = session.get('filename')
    predicted_class = session.get('predicted_class')

    if filename is None or predicted_class is None:
        flash('Error: No filename or predicted class found in session')
        return redirect(url_for('home'))

    # Read the image file and encode it as base64
    with open(os.path.join(app.config['UPLOAD_FOLDER'], filename), 'rb') as f:
        image_data = base64.b64encode(f.read()).decode('utf-8')

    return render_template('predict.html', filename=filename, predicted_class=predicted_class, predicted_image=image_data)


@app.route('/login', methods=['POST', 'GET'])
def login():
    if 'username' in session:
        flash("You are already logged in", "info")
        return redirect(url_for('home'))
    if request.method == 'POST':
        username = request.form['name']
        password = request.form['password']
        if not username or not password:
            flash("Both username and password are required", "error")
            return redirect(url_for('login'))
        user = User.query.filter_by(name=username, password=password).first()
        if user:
            session['username'] = username
            flash("Login successful", "success")
            return redirect(url_for('home'))
        else:
            flash("Incorrect username or password", "error")
            return redirect(url_for('login'))
    return render_template('login.html')


@app.route('/logout')
def logout():
    session.pop('username', None)
    flash("You have been logged out", "success")
    return redirect(url_for('login'))


@app.route('/signup', methods=['POST', 'GET'])
def signup():
    if request.method == 'POST':
        username = request.form['name']
        password = request.form['password']
        cpassword = request.form['cpassword']
        if not username or not password or not cpassword:
            flash("All fields are required", "error")
        elif password != cpassword:
            flash("The two passwords do not match", "error")
        else:
            existing_user = User.query.filter_by(name=username).first()
            if existing_user:
                flash("Username already exists", "error")
            else:
                new_user = User(name=username, password=password, cpassword=cpassword)
                db.session.add(new_user)
                db.session.commit()
                flash("Registration successful.", "success")
                return redirect(url_for('home'))
    return render_template('signup.html')


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/cameraupload')
def cameraupload():
    return render_template('cameraupload.html')


if __name__ == '__main__':
    create_db()
    app.run(debug=True)

