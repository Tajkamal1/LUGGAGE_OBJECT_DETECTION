import os
import cv2
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from ultralytics import YOLO
from werkzeug.utils import secure_filename
import numpy as np

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
RESULTS_FOLDER = 'static/results'

# Corrected: Create RESULTS_FOLDER within static
os.makedirs(os.path.join(app.root_path, RESULTS_FOLDER), exist_ok=True)
os.makedirs(os.path.join(app.root_path, UPLOAD_FOLDER), exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULTS_FOLDER'] = RESULTS_FOLDER

# Load the YOLO model once when the app starts
# NOTE: Update this path to your trained model, e.g., 'runs/detect/train/weights/best.pt'
model = YOLO('yolov8n.pt') 

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def detect_objects_yolo_seg(image_path):
    """
    Detects objects in an image using YOLO and draws bounding boxes with OpenCV.
    Returns the processed image as a NumPy array and a list of detected items.
    """
    try:
        # Load the image using OpenCV
        img = cv2.imread(image_path)
        if img is None:
            return None, [], []

        # Run inference
        results = model.predict(source=img, save=False)
        
        # Get the first result object (assuming single image)
        result_obj = results[0]
        
        # Get the original image to draw on
        processed_img = result_obj.plot()
        
        # Extract detected items
        detected_objects = []
        if len(result_obj.boxes) > 0:
            for cls_id in result_obj.boxes.cls:
                class_name = result_obj.names[int(cls_id)]
                detected_objects.append(class_name)
        
        threat_items = [item for item in detected_objects if item in ['knife', 'gun', 'scissors']]

        return processed_img, detected_objects, threat_items
    except Exception as e:
        print(f"Error during object detection: {e}")
        return None, [], []

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            original_filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(original_filepath)

            # Process the image and get results
            processed_image_array, detected_objects, threat_items = detect_objects_yolo_seg(original_filepath)
            
            if processed_image_array is None:
                return 'Error processing image.', 500

            processed_filename = f'detected_{filename}'
            processed_filepath = os.path.join(app.config['RESULTS_FOLDER'], processed_filename)
            
            # Save the processed image
            cv2.imwrite(processed_filepath, processed_image_array)

            return render_template(
                'result.html',
                original_image=filename,
                processed_image=processed_filename,
                detected_objects=detected_objects,
                threat_items=threat_items
            )
    return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/static/results/<filename>')
def result_file(filename):
    return send_from_directory(app.config['RESULTS_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)