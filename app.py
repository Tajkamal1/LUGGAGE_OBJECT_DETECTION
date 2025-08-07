import os
import cv2
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from ultralytics import YOLO
from werkzeug.utils import secure_filename
import numpy as np

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
RESULTS_FOLDER = 'static/results'

# Create necessary directories
os.makedirs(os.path.join(app.root_path, RESULTS_FOLDER), exist_ok=True)
os.makedirs(os.path.join(app.root_path, UPLOAD_FOLDER), exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULTS_FOLDER'] = RESULTS_FOLDER

# Load both YOLO models
model1= YOLO('yolov8n.pt')
model2=YOLO('yolov8x.pt')
model3 = YOLO('runs/detect/train4/weights/best.pt')
model4 = YOLO('runs/detect/train3/weights/best.pt')

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def detect_objects_yolo_seg(image_path):
    """
    Run both models on the image and combine their results.
    Returns: processed image array, list of detected objects, list of threat items
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            return None, [], []

        detected_objects = []
        threat_items = []

        # Run both models and plot predictions on the same image
        for model in [model1,model2,model3,model4]:
            results = model.predict(source=img, save=False)[0]
            
            if len(results.boxes) > 0:
                for box in results.boxes:
                    cls_id = int(box.cls[0])
                    class_name = results.names[cls_id]
                    detected_objects.append(class_name)
                    if class_name in ['knife', 'gun', 'scissors','blade','shuriken','spring','paperclip','zipper']:
                        threat_items.append(class_name)

            # Plot current model predictions onto the image
            img = results.plot()

        # Remove duplicates
        detected_objects = list(set(detected_objects))
        threat_items = list(set(threat_items))

        return img, detected_objects, threat_items

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

            # Detect objects
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
