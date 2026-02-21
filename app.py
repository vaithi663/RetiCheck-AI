from flask import Flask, request, jsonify, render_template
import numpy as np
from PIL import Image, ImageOps, ImageFilter
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.cm as cm
import base64
from io import BytesIO

app = Flask(__name__)

# 1. Load AI Model
try:
    model = load_model("models/keras_model.h5", compile=False)
    class_names = open("models/labels.txt", "r").readlines()
    print("✅ Chef (AI Model) is ready!")
except Exception as e:
    print(f"⚠️ Chef is missing: {e}")
    model = None

cdss_data = {
    "Normal": {"status": "Healthy Retina", "triage": "SAFE", "advice": "No major diseases detected.", "action": "General drops."},
    "Diabetic": {"status": "Diabetic Retinopathy", "triage": "URGENT", "advice": "High risk of vision loss.", "action": "Check Sugar/BP."},
    "Glaucoma": {"status": "Glaucoma Suspect", "triage": "REFERRAL", "advice": "Optic nerve damage detected.", "action": "Refer for IOP Test."}
}

# 2. XAI Heatmap Functions
# 2. XAI Heatmap Functions (ACCURATE VERSION)
def generate_robust_heatmap(img_array, model, class_index, patch_size=16, step=8):
    try:
        base_pred = model.predict(img_array, verbose=0)[0][class_index]
        patches, coords = [], []
        img_size = 224
        
        # Scanning the image with a smaller, finer box
        for y in range(0, img_size, step):
            for x in range(0, img_size, step):
                y_end, x_end = min(y + patch_size, img_size), min(x + patch_size, img_size)
                if (y_end - y) < 4 or (x_end - x) < 4: continue
                
                occ_img = img_array.copy()
                occ_img[0, y:y_end, x:x_end, :] = -1 # Blackout small region
                patches.append(occ_img[0])
                coords.append((y, y_end, x, x_end))
                
        if not patches: return None
        
        # Predict all patches
        preds = model.predict(np.array(patches), batch_size=32, verbose=0)[:, class_index]
        
        heatmap = np.zeros((img_size, img_size))
        for i, (y, y_end, x, x_end) in enumerate(coords):
            drop = base_pred - preds[i]
            # STRICT ACCURACY: Only mark regions where hiding it drops confidence significantly (>2%)
            if drop > 0.02: 
                heatmap[y:y_end, x:x_end] += drop
                
        heatmap = np.maximum(heatmap, 0)
        max_val = np.max(heatmap)
        if max_val > 0: 
            heatmap /= max_val
            
        # Smooth the heatmap so it looks like medical imaging, not blocky pixels
        heatmap_img = Image.fromarray(np.uint8(255 * heatmap)).filter(ImageFilter.GaussianBlur(radius=6))
        return np.array(heatmap_img) / 255.0
    except Exception as e: 
        print("Heatmap Error:", e)
        return None

def overlay_heatmap(img, heatmap, alpha=0.5):
    heatmap = np.uint8(255 * heatmap)
    jet = cm.get_cmap("jet")
    jet_heatmap = tf.keras.preprocessing.image.array_to_img(jet(np.arange(256))[:, :3][heatmap]).resize((img.size[0], img.size[1]))
    superimposed_img = tf.keras.preprocessing.image.img_to_array(jet_heatmap) * alpha + tf.keras.preprocessing.image.img_to_array(img)
    return tf.keras.preprocessing.image.array_to_img(superimposed_img)

# 3. Routes
@app.route('/')
def login(): return render_template('index.html')

@app.route('/patient')
def patient(): return render_template('patient.html')

@app.route('/upload')
def upload(): return render_template('upload.html')

@app.route('/result')
def result(): return render_template('result.html')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['image']
    image = Image.open(file.stream).convert("RGB")
    
    size = (224, 224)
    image_resized = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = (np.asarray(image_resized).astype(np.float32) / 127.5) - 1

    prediction = model.predict(data, verbose=0)
    index = np.argmax(prediction)
    raw_class = class_names[index].strip()
    class_name = raw_class[2:] if raw_class[0].isdigit() else raw_class
    confidence = float(prediction[0][index]) * 100

    disease = "Normal"
    if "Diabetic" in class_name: disease = "Diabetic"
    elif "Glaucoma" in class_name: disease = "Glaucoma"

    # Convert Original Image to Base64
    buf = BytesIO()
    image_resized.save(buf, format="JPEG")
    orig_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')

    # Generate Heatmap (Only if Disease)
    heat_b64 = ""
    heatmap_generated = False
    if disease != "Normal":
        heatmap = generate_robust_heatmap(data, model, index)
        if heatmap is not None:
            final_heatmap = overlay_heatmap(image_resized, heatmap)
            buf2 = BytesIO()
            final_heatmap.save(buf2, format="JPEG")
            heat_b64 = base64.b64encode(buf2.getvalue()).decode('utf-8')
            heatmap_generated = True

    report = cdss_data[disease]

    return jsonify({
        'disease': class_name,
        'confidence': f"{confidence:.1f}%",
        'status': report['status'],
        'triage': report['triage'],
        'advice': report['advice'],
        'action': report['action'],
        'orig_img': orig_b64,
        'heat_img': heat_b64,
        'heat_gen': heatmap_generated
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)