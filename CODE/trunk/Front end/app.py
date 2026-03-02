from flask import Flask, render_template, request, redirect, url_for, session, flash
from werkzeug.security import generate_password_hash, check_password_hash
import mysql.connector, os, re
import pandas as pd
import numpy as np
from torchvision import models, transforms
from PIL import Image
import uuid
import torch
from torchvision import transforms, models
import matplotlib.pyplot as plt
import pymysql
from werkzeug.utils import secure_filename
from torch import nn
import os
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F
from ultralytics import YOLO
import cv2


app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Change this in production!





mydb = pymysql.connect(
    host="localhost",
    user="root",
    password="root",
    port=3306,
    database='Durian'
)

mycursor = mydb.cursor()

def executionquery(query,values):
    mycursor.execute(query,values)
    mydb.commit()
    return

def retrivequery1(query,values):
    mycursor.execute(query,values)
    data = mycursor.fetchall()
    return data

def retrivequery2(query):
    mycursor.execute(query)
    data = mycursor.fetchall()
    return data



@app.route('/')
def index():
    prediction = session.pop('prediction', None)  # Flash prediction from POST
    return render_template('index.html', prediction=prediction)



@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/register', methods=["GET", "POST"])
def register():
    if request.method == "POST":
        # Retrieve form data
        name = request.form['name']  # Added name field
        email = request.form['email']
        password = request.form['password']
        c_password = request.form['c_password']
        
        # Check if passwords match
        if password == c_password:
            # Query to check if the email already exists (case-insensitive)
            query = "SELECT UPPER(email) FROM users"
            email_data = retrivequery2(query)
            email_data_list = [i[0] for i in email_data]
            
            # If the email is unique, insert the new user
            if email.upper() not in email_data_list:
                query = "INSERT INTO users (name, email, password) VALUES (%s, %s, %s)"
                values = (name, email, password)  # Include name in the insert query
                executionquery(query, values)
                return render_template('login.html', message="Successfully Registered!")
            
            # If email already exists
            return render_template('register.html', message="This email ID already exists!")
        
        # If passwords do not match
        return render_template('register.html', message="Confirm password does not match!")
    
    # If GET request
    return render_template('register.html')


@app.route('/login', methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form['email']
        password = request.form['password']
        
        # Query to check if email exists
        query = "SELECT UPPER(email) FROM users"
        email_data = retrivequery2(query)
        email_data_list = [i[0] for i in email_data]  # Simplified list comprehension

        if email.upper() in email_data_list:
            # Query to fetch the password for the provided email
            query = "SELECT password, name FROM users WHERE email = %s"
            values = (email,)
            user_data = retrivequery1(query, values)  # Assuming this returns a list of tuples
            
            if user_data:
                stored_password, name = user_data[0]  # Extract the password and name
                
                # Check if password matches (case-insensitive)
                if password == stored_password:
                    # Store the email and name in a session or global variable
                    session['user_email'] = email  # Store in session for security
                    session['user_name'] = name
                    
                    # Pass the user's name to the home page directly
                    return render_template('home.html', user_name=name)  # Pass the user name to home page
                
                # If passwords do not match
                return render_template('login.html', message="Invalid Password!")
            
            # If no data found for the user (which shouldn't happen here)
            return render_template('login.html', message="This email ID does not exist!")
        
        # If email doesn't exist
        return render_template('login.html', message="This email ID does not exist!")
    
    # If GET request
    return render_template('login.html')


@app.route('/home')
def home():
    # Check if user is logged in by verifying session
    if 'user_email' not in session:
        return redirect(url_for('login'))  # Redirect to login page if not logged in
    
    user_name = session.get('user_name')  # Retrieve user name from session
    return render_template('home.html', user_name=user_name)



@app.route('/model_accuracy')
def model_accuracy():
    if 'user_email' not in session:
        return redirect(url_for('login'))
    return render_template('model_accuracy.html')



# === CONFIG ===
# === CONFIG: UPLOAD FOLDER ===
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# === CREATE UPLOADS FOLDER ===
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# === ALLOWED EXTENSIONS ===
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'gif', 'webp'}
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Class names (same as in your model)
class_names = ['ALGAL_LEAF_SPOT', 'ALLOCARIDARA_ATTACK', 'HEALTHY_LEAF', 'LEAF_BLIGHT', 'PHOMOPSIS_LEAF_SPOT']

# Image transform (same as training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# === LOAD MODEL ONCE (outside route for efficiency) ===
def load_leaf_model(model_path='models/best_model.pth'):
    model = models.densenet121(pretrained=False)  # No need for pretrained if using saved weights
    model.classifier = torch.nn.Sequential(
        torch.nn.Linear(model.classifier.in_features, 1024),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.5),
        torch.nn.Linear(1024, 512),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.5),
        torch.nn.Linear(512, len(class_names))
    )
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    return model

# Load model globally
leaf_model = load_leaf_model()

# === PREDICTION FUNCTION ===
def predict_leaf_image(image_path):
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0)  # Add batch dim
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_tensor = input_tensor.to(device)
    leaf_model.to(device)
    
    with torch.no_grad():
        output = leaf_model(input_tensor)
        probabilities = F.softmax(output, dim=1)
        confidence, predicted_idx = torch.max(probabilities, 1)
    
    predicted_class = class_names[predicted_idx.item()]
    confidence_percent = confidence.item() * 100
    
    return predicted_class, confidence_percent, image


# === DISEASE INFORMATION DATABASE ===


# === DISEASE INFORMATION DATABASE ===
disease_info = {
    'ALGAL_LEAF_SPOT': {
        'name': 'Algal Leaf Spot',
        'description': 'A fungal disease caused by Cephaleuros virescens that appears as raised, velvety lesions on leaves.',
        'symptoms': 'Small, greenish-gray spots that enlarge and become reddish-brown with velvety texture.',
        'causes': 'High humidity, poor air circulation, and wet foliage conditions.',
        'recommendations': [
            'Prune affected leaves and improve air circulation',
            'Apply copper-based fungicides (e.g., copper oxychloride)',
            'Avoid overhead irrigation',
            'Maintain proper plant spacing'
        ],
        'fertilizer': 'Balanced NPK fertilizer (10-10-10) with added micronutrients',
        'severity': 'Moderate'
    },
    'ALLOCARIDARA_ATTACK': {
        'name': 'Allocaridara Pest Attack',
        'description': 'Infestation by Allocaridara insects causing physical damage to leaves.',
        'symptoms': 'Chewed leaf edges, holes in leaves, yellowing, and stunted growth.',
        'causes': 'Presence of Allocaridara beetles or other leaf-eating insects.',
        'recommendations': [
            'Apply neem oil spray (2% solution) every 7 days',
            'Use insecticidal soap for severe infestations',
            'Introduce natural predators (ladybugs, lacewings)',
            'Remove and destroy heavily infested leaves'
        ],
        'fertilizer': 'High nitrogen fertilizer (20-10-10) to promote new growth',
        'severity': 'High'
    },
    'HEALTHY_LEAF': {
        'name': 'Healthy Leaf',
        'description': 'No disease detected. The leaf shows normal, healthy characteristics.',
        'symptoms': 'Vibrant green color, uniform texture, no spots or discoloration.',
        'causes': 'Proper care and maintenance.',
        'recommendations': [
            'Continue current care routine',
            'Regular monitoring for early detection',
            'Maintain balanced fertilization',
            'Ensure adequate sunlight and water'
        ],
        'fertilizer': 'Maintain with balanced slow-release fertilizer (14-14-14)',
        'severity': 'None'
    },
    'LEAF_BLIGHT': {
        'name': 'Leaf Blight',
        'description': 'Fungal infection causing rapid wilting and death of leaf tissue.',
        'symptoms': 'Brown or black lesions, yellow halos, rapid spreading of dead tissue.',
        'causes': 'Phytophthora or Alternaria fungi, excessive moisture, poor drainage.',
        'recommendations': [
            'Apply systemic fungicide (e.g., mancozeb or chlorothalonil)',
            'Remove and burn infected leaves',
            'Improve soil drainage',
            'Avoid water splashing on leaves'
        ],
        'fertilizer': 'Low nitrogen, high phosphorus/potassium fertilizer (5-20-20)',
        'severity': 'High'
    },
    'PHOMOPSIS_LEAF_SPOT': {
        'name': 'Phomopsis Leaf Spot',
        'description': 'Fungal disease caused by Phomopsis species affecting leaf vitality.',
        'symptoms': 'Circular brown spots with yellow margins, often coalescing.',
        'causes': 'Wet conditions, plant stress, infected plant debris.',
        'recommendations': [
            'Apply fungicide containing thiophanate-methyl',
            'Prune to improve air circulation',
            'Clean up fallen leaves and debris',
            'Water at base to keep foliage dry'
        ],
        'fertilizer': 'Complete fertilizer with calcium and magnesium',
        'severity': 'Moderate to High'
    }
}

# === UPDATED PREDICTION ROUTE ===
@app.route('/predict_leaf', methods=['GET', 'POST'])
def predict_leaf():
    if 'user_email' not in session:
        return redirect(url_for('login'))

    if request.method == 'POST':
        file = request.files['image']
        if file and file.filename:
            filename = secure_filename(file.filename)
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(image_path)

            pred_class, confidence, _ = predict_leaf_image(image_path)
            
            # Get disease information
            disease_data = disease_info.get(pred_class, {})
            
            # CORRECT PATH FOR STATIC
            image_url = f"uploads/{filename}"

            return render_template('predict_leaf.html',
                                   prediction=pred_class,
                                   confidence=round(confidence, 2),
                                   image_url=image_url,
                                   disease_data=disease_data)

    return render_template('predict_leaf.html')

# === LOAD YOLO MODEL ONCE ===
yolo_model = YOLO("models/best.pt")

# @app.route('/predict_fruit', methods=['GET', 'POST'])
# def predict_fruit():
#     if 'user_email' not in session:
#         return redirect(url_for('login'))

#     if request.method == 'POST':
#         file = request.files['image']
#         if file and file.filename:
#             filename = secure_filename(file.filename)
#             image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#             file.save(image_path)

#             # === YOLO PREDICTION ===
#             results = yolo_model.predict(source=image_path, conf=0.20, save=False)
#             result = results[0]

#             # Draw boxes
#             img_with_boxes = result.plot()
#             output_path = os.path.join(app.config['UPLOAD_FOLDER'], f"detected_{filename}")
#             cv2.imwrite(output_path, img_with_boxes)

#             # Extract detections
#             detections = []
#             for box in result.boxes:
#                 cls_id = int(box.cls[0])
#                 conf = float(box.conf[0]) * 100
#                 class_name = yolo_model.names[cls_id]
#                 detections.append({
#                     'class_name': class_name.upper(),
#                     'confidence': round(conf, 2)
#                 })

#             image_url = f"uploads/detected_{filename}"

#             return render_template('predict_fruit.html',
#                                    image_url=image_url,
#                                    detections=detections)

#     return render_template('predict_fruit.html')




# === FRUIT DISEASE INFORMATION DATABASE ===
fruit_disease_info = {
    'DAMAGE': {
        'name': 'Physical Damage',
        'description': 'Mechanical injury to durian fruit caused by impact, pests, or improper handling.',
        'symptoms': 'Bruises, cracks, cuts, or deformities on fruit surface.',
        'causes': 'Poor handling during harvest, transport damage, bird/animal attacks, or hail damage.',
        'recommendations': [
            'Harvest carefully using proper tools',
            'Implement cushioning during transport',
            'Use protective netting against birds/animals',
            'Separate damaged fruits immediately to prevent rot spread'
        ],
        'treatment': 'Apply antifungal wax to damaged areas, store in cool dry place',
        'severity': 'Moderate',
        'market_value': 'Reduced by 40-60%',
        'storage_advice': 'Consume within 2-3 days, do not store with healthy fruits'
    },
    'FUNGUS': {
        'name': 'Fungal Infection',
        'description': 'Fungal growth on durian fruit, often starting at wounds or stem end.',
        'symptoms': 'White/green mold, soft rotting spots, musty odor, discoloration.',
        'causes': 'Phytophthora or other fungal pathogens, high humidity, poor air circulation.',
        'recommendations': [
            'Apply fungicide (mancozeb or copper-based) before storage',
            'Improve ventilation in storage area',
            'Maintain humidity below 85%',
            'Remove infected fruits immediately'
        ],
        'treatment': 'Discard severely infected fruits, treat mild cases with antifungal spray',
        'severity': 'High',
        'market_value': 'Reduced by 70-90%',
        'storage_advice': 'Not suitable for long storage, separate immediately'
    },
    'WORM': {
        'name': 'Worm Infestation',
        'description': 'Insect larvae (usually durian borer) feeding inside the fruit.',
        'symptoms': 'Small entry holes, frass (insect waste) around holes, internal tunneling.',
        'causes': 'Durian fruit borer (Mudaria magniplaga) or other insect larvae.',
        'recommendations': [
            'Use pheromone traps for adult insects',
            'Apply insecticide during flowering/fruit set',
            'Bag young fruits with protective covers',
            'Remove and destroy infested fruits'
        ],
        'treatment': 'Cut out infested portions if detected early, otherwise discard',
        'severity': 'High',
        'market_value': 'Reduced by 80-100%',
        'storage_advice': 'Not suitable for sale, dispose properly to prevent spread'
    }
}

# === UPDATED YOLO PREDICTION ROUTE ===
@app.route('/predict_fruit', methods=['GET', 'POST'])
def predict_fruit():
    if 'user_email' not in session:
        return redirect(url_for('login'))

    if request.method == 'POST':
        file = request.files['image']
        if file and file.filename:
            filename = secure_filename(file.filename)
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(image_path)

            # === YOLO PREDICTION ===
            results = yolo_model.predict(source=image_path, conf=0.20, save=False)
            result = results[0]

            # Draw boxes
            img_with_boxes = result.plot()
            output_path = os.path.join(app.config['UPLOAD_FOLDER'], f"detected_{filename}")
            cv2.imwrite(output_path, img_with_boxes)

            # Extract detections
            detections = []
            detected_classes = set()
            for box in result.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0]) * 100
                class_name = yolo_model.names[cls_id].upper()
                detected_classes.add(class_name)
                detections.append({
                    'class_name': class_name,
                    'confidence': round(conf, 2)
                })

            # Get disease info for detected classes
            disease_data = {}
            for cls in detected_classes:
                if cls in fruit_disease_info:
                    disease_data[cls] = fruit_disease_info[cls]

            image_url = f"uploads/detected_{filename}"

            return render_template('predict_fruit.html',
                                   image_url=image_url,
                                   detections=detections,
                                   disease_data=disease_data)

    return render_template('predict_fruit.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))

if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)  # For image uploads
    app.run(debug=True)