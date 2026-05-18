"""
Flask API for DeepFake Guardian - Simplified Version
"""
import os
import sys
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
import torch
from PIL import Image
from torchvision import transforms

# Setup paths
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from vit_cnn_model import HybridModel

# Create Flask app
app = Flask(__name__)
CORS(app)

# Configure uploads
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['UPLOAD_FOLDER'] = os.path.join(REPO_ROOT, 'temp_uploads')
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load model
print("Loading model...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
try:
    model = HybridModel(
        cnn_backbone='resnet18',
        cnn_pretrained=False,
        vit_pretrained=False,
        fusion_strategy='concat'
    ).to(device)
    
    checkpoint_path = os.path.join(REPO_ROOT, 'checkpoints', 'best.pth')
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state'])
        print("✅ Checkpoint loaded successfully!")
    else:
        print("⚠️ No checkpoint found. Model will use random weights.")
    
    model.eval()
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    import traceback
    traceback.print_exc()

# Image transform
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Routes
@app.route('/')
def index():
    """Serve frontend HTML"""
    static_path = os.path.join(REPO_ROOT, 'static', 'index.html')
    if os.path.exists(static_path):
        with open(static_path, 'r', encoding='utf-8') as f:
            return f.read()
    else:
        return jsonify({'error': 'index.html not found'}), 404

@app.route('/health')
def health():
    """Health check"""
    return jsonify({
        'status': 'healthy',
        'model': 'ResNet18+ViT',
        'device': str(device),
        'fusion_strategy': 'concat'
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Predict if image is real or fake"""
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No image selected'}), 400
        
        # Save temporarily
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # Predict
            image = Image.open(filepath).convert('RGB')
            image_tensor = transform(image).unsqueeze(0).to(device)
            
            with torch.no_grad():
                logit = model(image_tensor)
                prob = torch.sigmoid(logit).item()
            
            prediction = "Real" if prob > 0.5 else "Fake"
            confidence = prob if prob > 0.5 else (1 - prob)
            
            return jsonify({
                'prediction': prediction,
                'confidence': float(confidence),
                'probability_real': float(prob),
                'probability_fake': float(1 - prob),
                'logit': float(logit.item())
            }), 200
        finally:
            if os.path.exists(filepath):
                os.remove(filepath)
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("\n" + "="*60)
    print("DeepFake Detector - API Server")
    print("="*60)
    print(f"Open: http://127.0.0.1:5000 or http://localhost:5000")
    print("="*60 + "\n")
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
