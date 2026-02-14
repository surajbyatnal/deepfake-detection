# 🚀 DeepFake Detector - Full Setup & Connection Guide

## Overview
This guide walks through setting up the complete frontend-backend connection for the DeepFake Detection system.

---

## 📋 System Architecture

```
┌─────────────────┐
│   Frontend      │
│  (index.html)   │
│  Port: 5000/    │
│  any port       │
└────────┬────────┘
         │
         │ HTTP/JSON
         │ /predict
         │
┌────────▼────────┐
│   Flask API     │
│  (api/app.py)   │
│  Port: 5000     │
└────────┬────────┘
         │
         │ Python
         │
┌────────▼────────────────┐
│  ML Model               │
│  (model/vit_cnn_model)  │
│  - ResNet18/50 (CNN)    │
│  - ViT-B/16 (Transformer)
└─────────────────────────┘
```

---

## 🔧 Installation Steps

### Step 1: Install Dependencies

```powershell
# Navigate to project directory
cd c:\Users\suraj\OneDrive\Documents\deepfake-detector

# Install all required packages
pip install -r requirements.txt
```

**Key packages added:**
- `Flask>=2.3.0` - Web framework
- `Flask-CORS>=4.0.0` - Cross-Origin Resource Sharing
- `Werkzeug>=2.3.0` - WSGI utilities

### Step 2: Verify Model Files

Ensure checkpoint file exists:
```
checkpoints/best.pth  ✓ (Should be present)
```

---

## ▶️ Running the System

### Method 1: Full Stack (Recommended)

**Terminal 1 - Start Flask API Server:**
```powershell
python api/app.py
```

Expected output:
```
============================================================
Starting DeepFake Guardian API Server
============================================================
Open your browser to: http://127.0.0.1:5000
============================================================

Loading DeepFake Detection Model...
Model loaded successfully!
Device: cpu (or cuda)
Checkpoint epoch: 4
Best validation AUC: 0.9932

 * Running on http://0.0.0.0:5000
```

**Terminal 2 - (Optional) View logs:**
```powershell
# Monitor for requests
# Keep the first terminal window open to see incoming requests
```

**Step 3: Access Frontend**

Open browser and navigate to:
```
http://127.0.0.1:5000
```

---

## 🧪 Testing the Connection

### Test 1: Check Backend Health

```javascript
// Open browser console (F12) and run:
fetch('http://127.0.0.1:5000/health').then(r => r.json()).then(console.log)
```

Expected response:
```json
{
  "status": "healthy",
  "model": "ResNet18 + ViT-B/16",
  "device": "cpu"
}
```

### Test 2: Test Image Upload

1. Go to http://127.0.0.1:5000
2. Click "Upload Image for Analysis"
3. Select any image file (JPG, PNG, etc.)
4. Wait for analysis results
5. Check browser console (F12) for detailed logs

### Test 3: Check Connection Status Indicator

- **Green indicator** = Backend connected ✓
- **Red indicator** = Backend offline ✗

---

## 🔌 API Endpoints

### 1. **GET `/`**
- **Purpose:** Serve frontend HTML
- **Response:** Static HTML page
- **Example:** `http://127.0.0.1:5000`

### 2. **POST `/predict`**
- **Purpose:** Analyze image for deepfakes
- **Request:**
  - Method: `POST`
  - Content-Type: `multipart/form-data`
  - Body: Image file as `image` parameter

- **Response:**
```json
{
  "prediction": "Real",
  "confidence": 0.95,
  "probability_real": 0.95,
  "probability_fake": 0.05,
  "logit": 2.944
}
```

### 3. **GET `/health`**
- **Purpose:** Check API status
- **Response:**
```json
{
  "status": "healthy",
  "model": "ResNet18 + ViT-B/16",
  "device": "cpu"
}
```

---

## 🌐 Frontend-Backend Connection

### Connection Flow

```
1. User uploads image via web interface
   ↓
2. Frontend JavaScript captures file
   ↓
3. FormData created with image
   ↓
4. Fetch request sent to /predict endpoint
   ↓
5. Backend receives request
   ↓
6. Image preprocessed (resize, normalize)
   ↓
7. Model inference (CNN + ViT)
   ↓
8. Prediction + confidence calculated
   ↓
9. JSON response sent back
   ↓
10. Frontend displays results with visualization
```

### CORS Configuration

The API automatically handles cross-origin requests:

```python
CORS(app, 
     resources={r"/*": {"origins": "*"}},
     allow_headers=["Content-Type", "Authorization"],
     methods=["GET", "POST", "OPTIONS"])
```

This allows:
- ✓ Frontend on different port to access API
- ✓ Browser CORS preflight requests
- ✓ File uploads with proper headers

---

## 🐛 Troubleshooting

### Issue 1: "Cannot connect to server"

**Problem:** Red indicator, upload fails

**Solutions:**
1. Verify Flask server is running:
```powershell
# Should see "Running on http://0.0.0.0:5000"
```

2. Check if port 5000 is available:
```powershell
# Check what's using port 5000
Get-NetTCPConnection -LocalPort 5000
```

3. Try accessing health endpoint directly:
```
http://127.0.0.1:5000/health
```

### Issue 2: Model loading fails

**Problem:** Server starts but crashes when loading model

**Solutions:**
1. Verify checkpoint file exists:
```powershell
Test-Path "checkpoints/best.pth"
```

2. Check memory (model requires ~2GB RAM):
```powershell
Get-Process | Where-Object {$_.WorkingSet -gt 1GB}
```

3. Use CPU instead of GPU:
```powershell
# Force CPU by setting environment variable
$env:CUDA_VISIBLE_DEVICES = "-1"
python api/app.py
```

### Issue 3: Upload button not working

**Problem:** Click upload but nothing happens

**Solutions:**
1. Check browser console for errors (F12 → Console)
2. Verify image file size < 16MB
3. Check browser network tab for failed requests
4. Try different image format (PNG, JPG)

### Issue 4: "Invalid JSON response"

**Problem:** Server responds but can't parse

**Solutions:**
1. Check Flask logs for errors in first terminal
2. Verify image format is valid
3. Check that `/predict` endpoint is working

---

## 📊 Frontend Features

### Tab 1: DeepFake Detector
- Upload images via file picker or drag-drop
- Real-time preview
- Detection results with confidence meter
- Probability breakdown (Real vs Fake)

### Tab 2: System Architecture
- Model architecture visualization
- Layer-by-layer information
- Hybrid CNN+ViT design explanation

### Tab 3: Model Comparison
- Comparison with other detection methods
- Accuracy metrics table
- Performance characteristics

### Tab 4: Performance Metrics
- Training curves (AUC & Loss)
- Epoch-by-epoch progress
- Final model performance stats

---

## 💡 Advanced Usage

### Run on Different Host

To access from other machines on network:

```powershell
# Modify app.py last line:
app.run(host='0.0.0.0', port=5000, debug=False)

# Then access from other machine:
# http://YOUR_IP:5000
```

### Enable Debug Mode (Development Only)

```powershell
# Set debug=True in api/app.py
app.run(host='0.0.0.0', port=5000, debug=True)
```

**Note:** Debug mode auto-reloads on code changes but is slower.

### Change Model Backbone

In `api/app.py`, modify:
```python
model = HybridModel(
    cnn_backbone='resnet50',  # or 'resnet18'
    cnn_pretrained=True,
    vit_pretrained=True
).to(device)
```

---

## ✅ Verification Checklist

- [ ] All packages installed (`pip install -r requirements.txt`)
- [ ] Flask server running on port 5000
- [ ] Model checkpoint loaded successfully
- [ ] Frontend accessible at http://127.0.0.1:5000
- [ ] Connection status indicator shows green
- [ ] Health endpoint returns 200 status
- [ ] Image upload triggers prediction
- [ ] Results display with confidence metrics
- [ ] No errors in browser console
- [ ] No errors in Flask server terminal

---

## 📞 Support

For issues:
1. Check Flask server logs
2. Open browser console (F12)
3. Check browser Network tab
4. Verify all dependencies installed
5. Ensure port 5000 is available

---

## 🎉 You're Ready!

Your DeepFake Detection system is now fully connected and ready to use!

- **Frontend:** Handles user interface and image upload
- **API:** Processes requests and manages model inference
- **Backend:** Hybrid CNN+ViT model for accurate detection

Happy detecting! 🚀
