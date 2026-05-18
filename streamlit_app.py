"""
Streamlit app for DeepFake Detection using Hybrid CNN + ViT model.
"""

import os
import sys
import torch
import streamlit as st
from PIL import Image
from torchvision import transforms
import numpy as np

# Add root to path
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from vit_cnn_model import HybridModel

# Page configuration
st.set_page_config(
    page_title="DeepFake Detection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styling
st.markdown("""
    <style>
    .main-title {
        text-align: center;
        color: #FF6B6B;
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .result-box {
        padding: 20px;
        border-radius: 10px;
        margin: 20px 0;
    }
    .result-real {
        background-color: #D4EDDA;
        border-left: 5px solid #28A745;
    }
    .result-fake {
        background-color: #F8D7DA;
        border-left: 5px solid #DC3545;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="main-title">🔍 DeepFake Detection System</div>', unsafe_allow_html=True)
st.markdown("---")

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Model Configuration")
    
    cnn_backbone = st.selectbox(
        "CNN Backbone",
        ["resnet50", "resnet18"],
        help="ResNet50 is more accurate but slower. ResNet18 is faster but less accurate."
    )
    
    fusion_strategy = st.selectbox(
        "Fusion Strategy",
        ["concat", "add", "mul", "attention", "cross_modal", "bilinear"],
        help="How to combine CNN and ViT features"
    )
    
    confidence_threshold = st.slider(
        "Confidence Threshold",
        0.5, 1.0, 0.5,
        help="Minimum confidence to report as real/fake"
    )
    
    st.markdown("---")
    st.info(
        "💡 **Tips:**\n"
        "- Upload high-quality images for better results\n"
        "- Both real and deepfake images work\n"
        "- Results are based on neural network predictions"
    )

# Initialize session state for model caching
@st.cache_resource
def load_model(backbone, fusion_strat):
    """Load model with caching"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    st.info(f"🔧 Loading model on {device}...")
    
    try:
        # Determine projection dims based on fusion strategy
        if fusion_strat in ['add', 'mul', 'attention', 'cross_modal']:
            proj_dim = 512
        else:
            proj_dim = 512
        
        model = HybridModel(
            cnn_backbone=backbone,
            cnn_pretrained=False,
            vit_pretrained=False,
            cnn_proj_dim=proj_dim,
            vit_proj_dim=proj_dim,
            fusion_strategy=fusion_strat
        ).to(device)
        
        model.eval()
        return model, device
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None, None

# Image preprocessing
def preprocess_image(image, device):
    """Preprocess image for model"""
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image.astype('uint8'))
    
    image = image.convert('RGB')
    return transform(image).unsqueeze(0).to(device)

# Main app
def main():
    # Load model
    model, device = load_model(cnn_backbone, fusion_strategy)
    
    if model is None or device is None:
        st.error("Failed to load model. Please check the configuration.")
        return
    
    # Two columns for upload and preview
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📤 Upload Image")
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=["jpg", "jpeg", "png", "bmp"],
            help="Supported formats: JPG, PNG, BMP"
        )
    
    with col2:
        st.subheader("👁️ Preview")
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, use_column_width=True)
    
    st.markdown("---")
    
    # Prediction
    if uploaded_file is not None:
        if st.button("🚀 Analyze Image", use_container_width=True, type="primary"):
            try:
                with st.spinner("🔍 Analyzing image..."):
                    image = Image.open(uploaded_file)
                    image_tensor = preprocess_image(image, device)
                    
                    with torch.no_grad():
                        logit = model(image_tensor)
                        probability_real = torch.sigmoid(logit).item()
                        probability_fake = 1.0 - probability_real
                    
                    # Determine prediction
                    if probability_real > confidence_threshold:
                        prediction = "✅ REAL"
                        confidence = probability_real
                        result_class = "result-real"
                    elif probability_fake > confidence_threshold:
                        prediction = "❌ DEEPFAKE"
                        confidence = probability_fake
                        result_class = "result-fake"
                    else:
                        prediction = "❓ UNCERTAIN"
                        confidence = max(probability_real, probability_fake)
                        result_class = "result-box"
                
                # Display results
                st.markdown("---")
                st.subheader("📊 Results")
                
                # Result box
                result_html = f"""
                <div class="result-box {result_class}">
                    <h2 style="margin: 0; font-size: 2rem;">{prediction}</h2>
                    <p style="margin: 10px 0; font-size: 1.1rem;">Confidence: <strong>{confidence:.2%}</strong></p>
                </div>
                """
                st.markdown(result_html, unsafe_allow_html=True)
                
                # Detailed metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Real Probability", f"{probability_real:.4f}")
                with col2:
                    st.metric("Fake Probability", f"{probability_fake:.4f}")
                with col3:
                    st.metric("Raw Logit", f"{logit.item():.4f}")
                
                # Progress bars
                st.markdown("**Probability Distribution:**")
                col1, col2 = st.columns(2)
                with col1:
                    st.progress(probability_real, text=f"Real: {probability_real:.2%}")
                with col2:
                    st.progress(probability_fake, text=f"Fake: {probability_fake:.2%}")
                
                # Model info
                st.markdown("---")
                st.markdown("**Model Configuration:**")
                info_cols = st.columns(3)
                with info_cols[0]:
                    st.write(f"**Backbone:** {cnn_backbone}")
                with info_cols[1]:
                    st.write(f"**Fusion:** {fusion_strategy}")
                with info_cols[2]:
                    st.write(f"**Device:** {device}")
                    
            except Exception as e:
                st.error(f"❌ Error during prediction: {str(e)}")
                import traceback
                st.text(traceback.format_exc())
    else:
        st.info("👆 Please upload an image to get started!")

if __name__ == "__main__":
    main()
