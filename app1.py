"""
app1.py

Flask application for Heart Attack Prediction using retinal images.

Routes:
  /first or /       : Render the landing page (first.html).
  /login            : Render the login page (login.html).
  /chart            : Render the chart page (chart.html).
  /upload           : Render the clustering upload page (index1.html).
  /success (POST)   : Handle image upload for clustering and plot EM clusters.
  /index            : Render the prediction page (index.html).
  /predict (POST)   : Handle image upload for heart attack risk prediction.

Utilities:
  save_img          : Save uploaded image securely.
  load_image        : Load and predict with the TensorFlow model.
  prepare_image     : Preprocess the image for model inference.

"""

from flask import Flask, render_template, request, url_for, current_app
import os
import secrets
from werkzeug.utils import secure_filename
from PIL import Image

# Custom modules for image classification and clustering
import label_image
import image_fuzzy_clustering as fem

app = Flask(__name__)

# Prediction model placeholder (lazy-loaded)
model = None

# Configure upload folder for clustering images
UPLOAD_FOLDER = os.path.join(app.root_path, 'static', 'images')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def load_image(image_path):
    """
    Run classification on the given image using the label_image module.

    Args:
        image_path (str): Path to the image file.

    Returns:
        str: Raw prediction label.
    """
    return label_image.main(image_path)


def prepare_image(image, target_size):
    """
    Preprocess PIL image for model inference:
      - Convert to RGB
      - Resize to target_size (width, height)
      - Convert to numpy array and expand dims
      - Apply ImageNet preprocessing

    Args:
        image (PIL.Image): Input image
        target_size (tuple): Desired (width, height)

    Returns:
        np.ndarray: Preprocessed tensor ready for model predict
    """
    # Ensure image is RGB
    if image.mode != "RGB":
        image = image.convert("RGB")

    # Resize and convert to array
    image = image.resize(target_size)
    from tensorflow.keras.preprocessing.image import img_to_array
    import numpy as np
    from tensorflow.keras.applications import imagenet_utils

    image_array = img_to_array(image)
    image_array = np.expand_dims(image_array, axis=0)
    image_array = imagenet_utils.preprocess_input(image_array)
    return image_array


def save_img(img_file, filename):
    """
    Save uploaded image to the static/images directory.

    Args:
        img_file (FileStorage): Uploaded file object
        filename (str): Original filename to preserve extension

    Returns:
        str: Full filesystem path of saved image
    """
    secure_name = secure_filename(filename)
    save_path = os.path.join(current_app.root_path, 'static', 'images', secure_name)
    image = Image.open(img_file)
    image.save(save_path)
    return save_path


@app.route('/')
@app.route('/first')
def first():
    """Render the landing page."""
    return render_template('first.html')


@app.route('/login')
def login():
    """Render the login page."""
    return render_template('login.html')


@app.route('/chart')
def chart():
    """Render the analysis chart page."""
    return render_template('chart.html')


@app.route('/upload')
def upload():
    """Render the clustering upload interface."""
    return render_template('index1.html')


@app.route('/success', methods=['POST'])
def success():
    """
    Handle clustering POST request:
      - Read cluster count and image file
      - Save original image
      - Generate and save EM-clustered image
      - Render results page
    """
    if request.method == 'POST':
        # Number of clusters requested by user
        cluster_count = int(request.form.get('cluster'))
        # Uploaded file
        uploaded_file = request.files['file']
        # Save original image
        original_path = save_img(uploaded_file, uploaded_file.filename)
        # Generate EM clustering plot and save as em_img.jpg in same folder
        fem.plot_cluster_img(original_path, cluster_count)
        return render_template('success.html')


@app.route('/index')
def index():
    """Render the prediction interface."""
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """
    Handle risk prediction POST request:
      - Save uploaded image temporarily
      - Perform model inference
      - Append human-readable risk explanation
      - Clean up temp file and return response text
    """
    if request.method == 'POST':
        f = request.files['file']
        temp_path = secure_filename(f.filename)
        f.save(temp_path)

        # Run label_image prediction
        raw_label = load_image(temp_path)
        raw_label = raw_label.title()

        # Mapping of labels to detailed risk info
        risk_map = {
            '1': '→ Age 30–35, SBP 140–160 mmHg, DBP 80–90 mmHg, BMI 27–29, HbA1c 4–5.6: Very Low Risk (20%)',
            '2': '→ Age 35–40, SBP 150–166 mmHg, DBP 85–95 mmHg, BMI 29–31, HbA1c 7.5–10.5: Mild Risk (40%)',
            '3': '→ Age 35–40, SBP 120–136 mmHg, DBP 75–55 mmHg, BMI 18–25, HbA1c 5.5–6.5: No Risk (Healthy)',
            '4': '→ Age 45–60, SBP 160–176 mmHg, DBP 95–100 mmHg, BMI 30–35, HbA1c 13.4–14.9: High Risk (60%)',
            '0': '→ Age 20–25, SBP 111–126 mmHg, DBP 80–85 mmHg, BMI 18–25, HbA1c 5.4–7.0: No Risk (Healthy)'
        }
        explanation = risk_map.get(raw_label, '')
        result_text = f"{raw_label} {explanation}"

        # Remove temporary file
        os.remove(temp_path)
        return result_text
    return None


if __name__ == '__main__':
    # Run Flask in debug mode for development
    app.run(debug=True)
