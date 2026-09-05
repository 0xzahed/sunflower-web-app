import streamlit as st
import numpy as np
from PIL import Image
import os
import onnxruntime as ort

# Page configuration
st.set_page_config(
    page_title="Sunflower Disease Detection",
    page_icon="🌻",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Modern CSS — glassmorphism + gradient design system
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

    /* ===== ROOT VARIABLES ===== */
    :root {
        --primary: #2E7D32;
        --primary-light: #4CAF50;
        --primary-dark: #1B5E20;
        --accent: #FFB300;
        --accent-light: #FFD54F;
        --bg-dark: #0D1B0E;
        --bg-card: rgba(255,255,255,0.08);
        --glass-border: rgba(255,255,255,0.15);
        --text-main: #E8F5E9;
        --text-muted: #A5D6A7;
        --danger: #EF5350;
        --warning: #FFB74D;
        --success: #66BB6A;
    }

    /* ===== GLOBAL RESET ===== */
    .stApp, .main, section[data-testid="stMain"] {
        font-family: 'Inter', -apple-system, sans-serif !important;
    }

    /* Animated gradient background */
    .stApp {
        background: linear-gradient(-45deg, #0D1B0E, #1B3A1E, #0F2810, #1E4020, #0D1B0E);
        background-size: 400% 400%;
        animation: gradientShift 20s ease infinite;
    }
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    /* Max-width container */
    .main .block-container {
        max-width: 1400px;
        padding: 2rem 2.5rem;
        margin: 0 auto;
    }

    /* ===== HERO HEADER ===== */
    .hero-container {
        position: relative;
        text-align: center;
        padding: 2.5rem 1rem 2rem;
        margin-bottom: 1.5rem;
        border-radius: 24px;
        background: linear-gradient(135deg, rgba(46,125,50,0.25), rgba(255,179,0,0.12));
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid rgba(255,255,255,0.12);
        overflow: hidden;
    }
    .hero-container::before {
        content: '';
        position: absolute;
        top: -50%; left: -50%;
        width: 200%; height: 200%;
        background: radial-gradient(circle, rgba(255,213,79,0.08) 0%, transparent 50%);
        animation: pulse 8s ease-in-out infinite;
    }
    @keyframes pulse {
        0%, 100% { transform: scale(1); opacity: 0.5; }
        50% { transform: scale(1.2); opacity: 0.8; }
    }
    .hero-emoji {
        font-size: 3.5rem;
        display: inline-block;
        animation: float 3s ease-in-out infinite;
        filter: drop-shadow(0 4px 12px rgba(255,179,0,0.4));
    }
    @keyframes float {
        0%, 100% { transform: translateY(0); }
        50% { transform: translateY(-8px); }
    }
    .hero-title {
        font-size: clamp(1.8rem, 4.5vw, 2.8rem);
        font-weight: 800;
        background: linear-gradient(135deg, #FFD54F, #66BB6A, #FFB300);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 0.5rem 0 0.3rem;
        letter-spacing: -0.02em;
        position: relative;
        z-index: 1;
    }
    .hero-subtitle {
        font-size: clamp(0.9rem, 2vw, 1.15rem);
        color: var(--text-muted);
        font-weight: 400;
        position: relative;
        z-index: 1;
    }
    .hero-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        padding: 0.35rem 1rem;
        border-radius: 100px;
        background: rgba(102,187,106,0.15);
        border: 1px solid rgba(102,187,106,0.3);
        color: var(--success);
        font-size: 0.8rem;
        font-weight: 600;
        margin-top: 0.8rem;
        position: relative;
        z-index: 1;
    }
    .hero-badge .dot {
        width: 8px; height: 8px;
        border-radius: 50%;
        background: var(--success);
        animation: blink 2s ease-in-out infinite;
    }
    @keyframes blink {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.3; }
    }

    /* ===== GLASS CARDS ===== */
    .glass-card {
        background: var(--bg-card);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid var(--glass-border);
        border-radius: 20px;
        padding: 1.5rem;
        margin: 1rem 0;
        transition: all 0.3s ease;
    }
    .glass-card:hover {
        border-color: rgba(102,187,106,0.3);
        box-shadow: 0 8px 32px rgba(46,125,50,0.15);
    }

    /* ===== RESULT BOX ===== */
    .result-box {
        padding: 2rem;
        border-radius: 20px;
        color: white;
        text-align: center;
        margin: 1.5rem 0;
        position: relative;
        overflow: hidden;
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid rgba(255,255,255,0.2);
    }
    .result-box::after {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0; bottom: 0;
        background: radial-gradient(circle at 50% 0%, rgba(255,255,255,0.15), transparent 70%);
        pointer-events: none;
    }
    .result-box.reliable {
        background: linear-gradient(135deg, rgba(46,125,50,0.85), rgba(27,94,32,0.9));
        box-shadow: 0 8px 40px rgba(46,125,50,0.3);
    }
    .result-box.unreliable {
        background: linear-gradient(135deg, rgba(255,152,0,0.85), rgba(245,124,0,0.9));
        box-shadow: 0 8px 40px rgba(255,152,0,0.3);
    }
    .result-label {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.15em;
        opacity: 0.8;
        margin-bottom: 0.5rem;
        position: relative; z-index: 1;
    }
    .disease-name {
        font-size: clamp(1.6rem, 4vw, 2.4rem);
        font-weight: 800;
        margin: 0.5rem 0;
        letter-spacing: -0.01em;
        position: relative; z-index: 1;
    }
    .confidence-score {
        font-size: clamp(1.1rem, 2.5vw, 1.6rem);
        font-weight: 600;
        margin: 0.3rem 0;
        position: relative; z-index: 1;
    }
    .confidence-bar {
        height: 8px;
        background: rgba(255,255,255,0.2);
        border-radius: 100px;
        margin: 0.8rem auto 0;
        max-width: 300px;
        overflow: hidden;
        position: relative; z-index: 1;
    }
    .confidence-bar-fill {
        height: 100%;
        background: linear-gradient(90deg, #FFD54F, #FFFFFF);
        border-radius: 100px;
        animation: fillBar 1s ease-out;
    }
    @keyframes fillBar {
        from { width: 0; }
    }
    .reliability-tag {
        display: inline-block;
        padding: 0.3rem 1rem;
        border-radius: 100px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-top: 0.8rem;
        background: rgba(255,255,255,0.2);
        position: relative; z-index: 1;
    }

    /* ===== INFO BOX ===== */
    .info-box {
        padding: 1.2rem 1.5rem;
        border-radius: 14px;
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,255,255,0.1);
        border-left: 4px solid var(--primary-light);
        margin: 1rem 0;
        font-size: 0.95rem;
        color: var(--text-main);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
    }

    /* ===== BUTTONS ===== */
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #2E7D32, #4CAF50);
        color: white;
        font-size: 1.1rem;
        font-weight: 700;
        padding: 0.85rem 1.5rem;
        border-radius: 14px;
        border: none;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        letter-spacing: 0.02em;
        box-shadow: 0 4px 20px rgba(46,125,50,0.3);
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #1B5E20, #388E3C);
        transform: translateY(-3px);
        box-shadow: 0 8px 30px rgba(46,125,50,0.45);
    }
    .stButton > button:active {
        transform: translateY(-1px);
    }

    /* ===== SIDEBAR ===== */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(13,27,14,0.95), rgba(15,40,16,0.95));
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border-right: 1px solid rgba(255,255,255,0.08);
    }
    section[data-testid="stSidebar"] .stMarkdown, 
    section[data-testid="stSidebar"] .stText {
        color: var(--text-main) !important;
    }

    /* ===== SECTION TITLES ===== */
    .section-title {
        font-size: 1.3rem;
        font-weight: 700;
        color: var(--text-main);
        margin: 1.5rem 0 0.8rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .section-title .icon {
        font-size: 1.5rem;
    }

    /* ===== DISEASE CARDS (sample section) ===== */
    .disease-card {
        background: var(--bg-card);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid var(--glass-border);
        border-radius: 16px;
        padding: 1.2rem 0.8rem;
        text-align: center;
        transition: all 0.3s ease;
        height: 100%;
    }
    .disease-card:hover {
        transform: translateY(-5px);
        border-color: rgba(255,179,0,0.4);
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }
    .disease-card .card-icon {
        font-size: 2rem;
        margin-bottom: 0.5rem;
        display: block;
    }
    .disease-card .card-name {
        font-size: 0.85rem;
        font-weight: 700;
        color: var(--text-main);
        margin-bottom: 0.3rem;
    }
    .disease-card .card-desc {
        font-size: 0.72rem;
        color: var(--text-muted);
        line-height: 1.3;
    }

    /* ===== PROGRESS BARS ===== */
    .stProgress > div > div {
        background: linear-gradient(90deg, #4CAF50, #FFB300);
        border-radius: 100px;
    }
    .stProgress > div {
        background: rgba(255,255,255,0.1);
        border-radius: 100px;
        height: 10px;
    }

    /* ===== EXPANDERS ===== */
    .streamlit-expanderHeader {
        background: rgba(255,255,255,0.06);
        border-radius: 12px;
        font-weight: 600;
        color: var(--text-main);
    }

    /* ===== FILE UPLOADER ===== */
    .stFileUploader {
        border-radius: 16px;
    }
    [data-testid="stFileUploaderDropzone"] {
        background: rgba(255,255,255,0.04);
        border: 2px dashed rgba(102,187,106,0.3);
        border-radius: 16px;
        transition: all 0.3s ease;
    }
    [data-testid="stFileUploaderDropzone"]:hover {
        border-color: var(--primary-light);
        background: rgba(102,187,106,0.08);
    }

    /* ===== FOOTER ===== */
    .footer {
        text-align: center;
        padding: 2rem 1rem;
        margin-top: 2rem;
        border-top: 1px solid rgba(255,255,255,0.08);
        color: var(--text-muted);
    }
    .footer-brand {
        font-size: 1rem;
        font-weight: 600;
        color: var(--text-main);
        margin-bottom: 0.3rem;
    }
    .footer-copy {
        font-size: 0.8rem;
        opacity: 0.7;
    }
    .footer-note {
        font-size: 0.78rem;
        margin-top: 0.8rem;
        padding: 0.6rem 1rem;
        border-radius: 10px;
        background: rgba(239,83,80,0.08);
        border: 1px solid rgba(239,83,80,0.15);
        display: inline-block;
    }

    /* ===== STEP INDICATOR ===== */
    .step-indicator {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.4rem 1rem;
        border-radius: 100px;
        font-size: 0.85rem;
        font-weight: 600;
        margin: 0.3rem 0;
    }
    .step-indicator.step-1 {
        background: rgba(102,187,106,0.12);
        color: var(--success);
        border: 1px solid rgba(102,187,106,0.25);
    }
    .step-indicator.step-2 {
        background: rgba(255,179,0,0.12);
        color: var(--accent);
        border: 1px solid rgba(255,179,0,0.25);
    }

    /* ===== METRICS GRID ===== */
    .metrics-grid {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 0.8rem;
        margin: 1rem 0;
    }
    .metric-item {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px;
        padding: 0.8rem 1rem;
        text-align: center;
    }
    .metric-label {
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: var(--text-muted);
        margin-bottom: 0.2rem;
    }
    .metric-value {
        font-size: 1.1rem;
        font-weight: 700;
        color: var(--text-main);
    }

    /* ===== TEXT COLORS ===== */
    .stMarkdown, .stText, p, span, li {
        color: var(--text-main) !important;
    }
    h1, h2, h3, h4, h5, h6 {
        color: var(--text-main) !important;
    }
    .stMarkdown strong, b {
        color: var(--accent-light) !important;
    }

    /* ===== ALERTS ===== */
    .stAlert {
        border-radius: 14px;
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
    }

    /* ===== SCROLLBAR ===== */
    ::-webkit-scrollbar { width: 8px; }
    ::-webkit-scrollbar-track { background: rgba(255,255,255,0.03); }
    ::-webkit-scrollbar-thumb {
        background: rgba(102,187,106,0.3);
        border-radius: 100px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: rgba(102,187,106,0.5);
    }

    /* ===== MOBILE ===== */
    @media (max-width: 768px) {
        .main .block-container { padding: 1rem; }
        .hero-container { padding: 1.5rem 0.5rem 1rem; }
        .hero-emoji { font-size: 2.5rem; }
        .glass-card { padding: 1rem; }
        .result-box { padding: 1.2rem; }
        .metrics-grid { grid-template-columns: 1fr; }
    }
</style>
""", unsafe_allow_html=True)

# Disease information dictionary
DISEASE_INFO = {
    "Alternaria Leaf Spot": {
        "description": "ছত্রাকজনিত রোগ যা সূর্যমুখী পাতায় গাঢ় বাদামী দাগ সৃষ্টি করে। Alternaria leaf spot is a fungal disease causing dark brown lesions with concentric rings on sunflower leaves.",
        "symptoms": "Dark brown to black circular lesions with concentric rings (target-like pattern) on leaves",
        "treatment": "ফাঙ্গিসাইড প্রয়োগ করুন, সংক্রমিত পাতা সরান। Apply fungicides, remove infected leaves, practice crop rotation",
        "severity": "মাঝারি (Moderate)",
        "color": "#FFA500"
    },
    "Downy Mildew": {
        "description": "সূর্যমুখীর একটি মারাত্মক ছত্রাকজনিত রোগ। Plasmopara halstedii দ্বারা সৃষ্ট যা পাতায় হলুদ দাগ ও নিচে সাদা ছত্রাক তৈরি করে। Downy mildew is a serious fungal disease caused by Plasmopara halstedii.",
        "symptoms": "Yellowish patches on upper leaf surface with white/gray fuzzy growth on the underside",
        "treatment": "প্রতিরোধী জাত ব্যবহার করুন, বীজ চিকিৎসা করুন। Use resistant varieties, seed treatment, improve drainage",
        "severity": "উচ্চ (High)",
        "color": "#FF4500"
    },
    "Healthy": {
        "description": "কোন রোগ সনাক্ত করা হয়নি - গাছ সুস্থ। The sunflower plant appears healthy with no visible disease symptoms.",
        "symptoms": "Green, vibrant leaves without lesions, spots, or discoloration",
        "treatment": "নিয়মিত পর্যবেক্ষণ চালিয়ে যান। Maintain good agricultural practices and regular monitoring",
        "severity": "সুস্থ (Healthy)",
        "color": "#32CD32"
    },
    "Powdery Mildew": {
        "description": "পাতার উপর সাদা গুঁড়ার মতো ছত্রাক বৃদ্ধি। Powdery mildew appears as white powdery fungal growth on leaf surfaces.",
        "symptoms": "White to grayish powdery patches on upper and lower leaf surfaces",
        "treatment": "সালফার ভিত্তিক ফাঙ্গিসাইড ব্যবহার করুন। Apply sulfur-based fungicides, ensure proper spacing for air circulation",
        "severity": "মাঝারি (Moderate)",
        "color": "#DDA0DD"
    },
    "Wilted Leaves": {
        "description": "পাতা কুঁকড়ে ও শুকিয়ে যাওয়া। Wilted leaves indicate water stress or vascular disease causing leaves to droop and dry.",
        "symptoms": "Drooping, curling, and drying of leaves; loss of turgor pressure",
        "treatment": "নিয়মিত সেচ দিন, রোগ প্রতিরোধী জাত ব্যবহার করুন। Ensure proper irrigation, use disease-resistant varieties, check for root pathogens",
        "severity": "উচ্চ (High)",
        "color": "#8B4513"
    }
}

@st.cache_resource
def load_trained_model():
    """Load the ONNX model for inference"""
    try:
        LOCAL_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sunflower_model.onnx')

        if os.path.exists(LOCAL_MODEL_PATH):
            st.info("🔄 Loading model...")
            sess = ort.InferenceSession(LOCAL_MODEL_PATH)
            st.success("✅ Model loaded successfully!")
            return sess, LOCAL_MODEL_PATH

        st.error("❌ Model file not found: sunflower_model.onnx")
        return None, None

    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None, None

def preprocess_input(x):
    """MobileNetV2 preprocessing: scale [-1, 1]"""
    return (x / 127.5) - 1.0

def preprocess_image(image, target_size=(224, 224)):
    """Preprocess image for MobileNetV2 model"""
    if image.mode != 'RGB':
        image = image.convert('RGB')

    image = image.resize(target_size)
    img_array = np.array(image).astype(np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    return img_array

def is_sunflower_leaf_image(image):
    """
    Multi-feature validation to check if image is likely a plant leaf.
    Uses color analysis, texture detection, and plant-like pixel ratio.
    Rejects non-leaf images (sky, walls, paper, skin, soil, flat colors, etc.)
    """
    # Convert to RGB and resize for consistent analysis
    img_rgb = image.convert('RGB')
    img_array = np.array(img_rgb).astype(float)

    # Resize to 224x224 for consistent metrics
    if img_array.shape[:2] != (224, 224):
        pil_img = Image.fromarray(img_array.astype('uint8')).resize((224, 224))
        img_array = np.array(pil_img).astype(float)

    r, g, b = img_array[:, :, 0], img_array[:, :, 1], img_array[:, :, 2]
    avg_r, avg_g, avg_b = np.mean(r), np.mean(g), np.mean(b)

    # --- Feature 1: Green dominance (overall color balance) ---
    green_dominance = avg_g / (avg_r + avg_g + avg_b + 1) * 100

    # --- Feature 2: Texture (standard deviation of pixel values) ---
    std_r, std_g, std_b = np.std(r), np.std(g), np.std(b)
    std_avg = (std_r + std_g + std_b) / 3

    # --- Feature 3: Unique colors (flat images have very few) ---
    unique_colors = len(np.unique(img_array.astype(int).reshape(-1, 3), axis=0))

    # --- Feature 4: Green pixel ratio (pixels where green clearly dominates) ---
    green_pixels = np.sum((g > r) & (g > b) & (g > 40))
    green_ratio = green_pixels / (224 * 224) * 100

    # --- Feature 5: Brown/yellow pixel ratio (for diseased/wilted leaves) ---
    # Brown: reddish-brown tones (r > g > b)
    brown_pixels = np.sum((r > g) & (g > b) & (r > 80) & (r < 220) & (g > 50))
    brown_ratio = brown_pixels / (224 * 224) * 100
    # Yellow: yellowish tones (high r, high g, low b) — common in diseased leaves
    yellow_pixels = np.sum((r > 180) & (g > 150) & (b < 130) & (r >= g))
    yellow_ratio = yellow_pixels / (224 * 224) * 100

    # --- Feature 6: Plant-like pixel ratio (green OR brown OR yellow) ---
    plant_ratio = green_ratio + brown_ratio + yellow_ratio

    # --- Feature 7: Skin tone detection (reject human photos) ---
    # Use pixel-level skin ratio, not just average color (brown/wilted leaves
    # can have average colors similar to skin tone, but very few actual skin pixels)
    skin_pixels = np.sum((r > 150) & (r > g) & (g > b) & (r > 120) & ((r - b) > 20))
    skin_ratio = skin_pixels / (224 * 224) * 100

    # --- Validation checks ---
    reasons = []

    # Check 1: Must have texture (reject flat colors, gradients)
    if std_avg < 10:
        reasons.append(f"Too flat/uniform (texture: {std_avg:.1f}, need >10)")

    # Check 2: Must have enough color variety (reject flat/gradient images)
    if unique_colors < 3000:
        reasons.append(f"Too few unique colors ({unique_colors}, need >3000)")

    # Check 3: Must look like a plant — either clearly green-dominant
    # (green_ratio > 35%, above the ~33% random baseline in RGB)
    # OR mostly plant-colored with at least some green (for wilted/diseased)
    is_green_dominant = green_ratio > 35
    is_plant_dominant = plant_ratio > 60 and green_ratio > 10
    if not (is_green_dominant or is_plant_dominant):
        reasons.append(
            f"Not enough plant-like colors (green: {green_ratio:.1f}%, "
            f"plant: {plant_ratio:.1f}% — need green >35% OR plant >60%+green >10%)"
        )

    # Check 4: Reject skin tones (only if majority of pixels are skin-colored)
    if skin_ratio > 50:
        reasons.append(f"Appears to be human skin (skin: {skin_ratio:.1f}%)")

    # Build result
    is_valid = len(reasons) == 0
    detail = {
        'green_dominance': green_dominance,
        'std_avg': std_avg,
        'unique_colors': unique_colors,
        'green_ratio': green_ratio,
        'brown_ratio': brown_ratio,
        'yellow_ratio': yellow_ratio,
        'plant_ratio': plant_ratio,
        'skin_ratio': skin_ratio,
        'reasons': reasons
    }

    return is_valid, detail

def predict_disease(model, image):
    """Predict disease from image with confidence filtering"""
    # Preprocess image
    processed_img = preprocess_image(image)

    # Make prediction using ONNX Runtime
    input_name = model.get_inputs()[0].name
    predictions = model.run(None, {input_name: processed_img})[0]

    # Get class names (must match training order)
    class_names = ['Alternaria Leaf Spot', 'Downy Mildew', 'Healthy', 'Powdery Mildew', 'Wilted Leaves']

    # Convert to numpy
    predictions_np = np.array(predictions[0])

    # Get predicted class and confidence
    predicted_class_idx = np.argmax(predictions_np)
    predicted_class = class_names[predicted_class_idx]
    confidence = float(predictions_np[predicted_class_idx] * 100)

    # Check confidence gap
    max_confidence = np.max(predictions_np)
    second_max_confidence = np.partition(predictions_np, -2)[-2]
    confidence_gap = (max_confidence - second_max_confidence) * 100

    # Get all predictions for display
    all_predictions = {class_names[i]: float(predictions_np[i] * 100) for i in range(len(class_names))}

    return predicted_class, confidence, all_predictions, confidence_gap

def main():
    # Hero Header
    st.markdown("""
    <div class="hero-container">
        <div class="hero-emoji">🌻</div>
        <h1 class="hero-title">Sunflower Disease Detection</h1>
        <p class="hero-subtitle">AI-Powered Leaf Disease Classification</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("""
        <div style="text-align:center; padding: 1rem 0 0.5rem;">
            <div style="font-size: 3rem; display: inline-block; animation: float 3s ease-in-out infinite;">🌻</div>
            <h3 style="margin: 0.3rem 0; font-weight: 800; background: linear-gradient(135deg, #FFD54F, #66BB6A); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;">Sunflower AI</h3>
            <p style="font-size: 0.8rem; opacity: 0.6; margin: 0;">Deep Learning Disease Detection</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        st.markdown("""
        <div class="glass-card" style="margin: 0.5rem 0;">
            <h4 style="margin: 0 0 0.5rem; font-size: 0.9rem; color: #FFD54F;">📋 How to Use</h4>
            <p style="font-size: 0.82rem; margin: 0.2rem 0; opacity: 0.85;">
                <b>1.</b> Upload a sunflower leaf image<br>
                <b>2.</b> Click "Analyze Image"<br>
                <b>3.</b> View prediction results
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="glass-card" style="margin: 0.5rem 0;">
            <h4 style="margin: 0 0 0.5rem; font-size: 0.9rem; color: #66BB6A;">🦠 Detectable Diseases</h4>
            <p style="font-size: 0.82rem; margin: 0.15rem 0; opacity: 0.85;">
                🟤 Alternaria Leaf Spot<br>
                🟠 Downy Mildew<br>
                🟢 Healthy<br>
                ⚪ Powdery Mildew<br>
                🟤 Wilted Leaves
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="glass-card" style="margin: 0.5rem 0;">
            <h4 style="margin: 0 0 0.5rem; font-size: 0.9rem; color: #FFD54F;">📸 Best Practices</h4>
            <p style="font-size: 0.78rem; margin: 0.15rem 0; opacity: 0.8;">
                • Clear, well-lit photos<br>
                • Leaf surface clearly visible<br>
                • Avoid blurry images<br>
                • Close-up of affected area
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # Load model
    model, model_name = load_trained_model()
    
    if model is None:
        st.stop()

    # File uploader
    st.markdown("""
    <div class="section-title">
        <span class="icon">📤</span>
        <span>Upload Leaf Image</span>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Drop a sunflower leaf image here or click to browse",
        type=['jpg', 'jpeg', 'png'],
        help="Upload a clear image of a sunflower leaf for disease detection"
    )
    
    if uploaded_file is not None:
        # Create two columns
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("""
            <div class="section-title">
                <span class="icon">📸</span>
                <span>Uploaded Image</span>
            </div>
            """, unsafe_allow_html=True)
            image = Image.open(uploaded_file)
            st.image(image, use_container_width=True)
            
            # Image details in glass card
            st.markdown(f"""
            <div class="glass-card" style="padding: 1rem 1.2rem;">
                <div style="font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.1em; color: #A5D6A7; margin-bottom: 0.5rem;">Image Details</div>
                <div class="metrics-grid" style="margin: 0;">
                    <div class="metric-item">
                        <div class="metric-label">Format</div>
                        <div class="metric-value">{image.format}</div>
                    </div>
                    <div class="metric-item">
                        <div class="metric-label">Resolution</div>
                        <div class="metric-value">{image.size[0]}×{image.size[1]}</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="section-title">
                <span class="icon">🔬</span>
                <span>AI Analysis</span>
            </div>
            """, unsafe_allow_html=True)
            
            # Analyze button
            if st.button("🚀 Analyze Image", use_container_width=True):
                with st.spinner("🔄 Analyzing image... Please wait..."):
                    try:
                        # Step 1: Validate if it's a sunflower leaf image
                        st.markdown('<div class="step-indicator step-1">🔍 Step 1: Validating image...</div>', unsafe_allow_html=True)
                        is_leaf, detail = is_sunflower_leaf_image(image)
                        
                        if not is_leaf:
                            reasons_text = "\n".join(f"  • {r}" for r in detail['reasons'])
                            st.markdown(f"""
                            <div class="glass-card" style="border-left: 4px solid #EF5350;">
                                <h4 style="color: #EF5350; margin: 0 0 0.5rem;">🚫 Not a Leaf Image</h4>
                                <p style="margin: 0.3rem 0; font-size: 0.9rem;"><b>এটি সূর্যমুখীর পাতার ছবি বলে মনে হচ্ছে না!</b></p>
                                <p style="margin: 0.5rem 0 0.3rem; font-size: 0.85rem; opacity: 0.8;">Validation failed:</p>
                                <p style="font-size: 0.82rem; opacity: 0.85; margin: 0.3rem 0;">{reasons_text}</p>
                                <div class="metrics-grid" style="margin-top: 0.8rem;">
                                    <div class="metric-item"><div class="metric-label">Green</div><div class="metric-value">{detail['green_ratio']:.1f}%</div></div>
                                    <div class="metric-item"><div class="metric-label">Plant-like</div><div class="metric-value">{detail['plant_ratio']:.1f}%</div></div>
                                    <div class="metric-item"><div class="metric-label">Texture</div><div class="metric-value">{detail['std_avg']:.1f}</div></div>
                                    <div class="metric-item"><div class="metric-label">Colors</div><div class="metric-value">{detail['unique_colors']}</div></div>
                                </div>
                                <p style="margin-top: 0.8rem; font-size: 0.82rem; color: #FFB74D;">Please upload a clear sunflower leaf image.</p>
                            </div>
                            """, unsafe_allow_html=True)
                            st.stop()
                        
                        st.markdown(f"""
                        <div class="glass-card" style="border-left: 4px solid #66BB6A; padding: 0.8rem 1.2rem;">
                            <p style="margin: 0; color: #66BB6A; font-size: 0.88rem;">
                                ✅ Validation passed · Green: {detail['green_ratio']:.1f}% · Plant: {detail['plant_ratio']:.1f}% · Texture: {detail['std_avg']:.1f}
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Step 2: Predict disease
                        st.markdown('<div class="step-indicator step-2">🤖 Step 2: Running AI analysis...</div>', unsafe_allow_html=True)
                        predicted_class, confidence, all_predictions, confidence_gap = predict_disease(model, image)
                        
                        # Step 3: Check confidence levels
                        CONFIDENCE_THRESHOLD = 50
                        CONFIDENCE_GAP_THRESHOLD = 20
                        is_reliable = confidence >= CONFIDENCE_THRESHOLD and confidence_gap >= CONFIDENCE_GAP_THRESHOLD
                        
                        if not is_reliable:
                            st.markdown(f"""
                            <div class="glass-card" style="border-left: 4px solid #FFB74D;">
                                <h4 style="color: #FFB74D; margin: 0 0 0.3rem; font-size: 0.95rem;">⚠️ Low Confidence Detection</h4>
                                <p style="font-size: 0.82rem; margin: 0.2rem 0; opacity: 0.85;">মডেল এই ছবি সম্পর্কে নিশ্চিত নয়।</p>
                                <p style="font-size: 0.8rem; margin: 0.3rem 0; opacity: 0.75;">
                                    Confidence: {confidence:.2f}% · Gap: {confidence_gap:.2f}%<br>
                                    Try a clearer image or consult an expert.
                                </p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Display result
                        result_class = "reliable" if is_reliable else "unreliable"
                        st.markdown(f"""
                        <div class="result-box {result_class}">
                            <div class="result-label">🎯 Detection Result</div>
                            <div class="disease-name">{predicted_class.replace('_', ' ')}</div>
                            <div class="confidence-score">Confidence: {confidence:.2f}%</div>
                            <div class="confidence-bar">
                                <div class="confidence-bar-fill" style="width: {confidence:.1f}%;"></div>
                            </div>
                            <div class="reliability-tag">
                                {"✅ High Reliability" if is_reliable else "⚠️ Low Reliability"}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Display disease information
                        if predicted_class in DISEASE_INFO:
                            info = DISEASE_INFO[predicted_class]
                            
                            with st.expander("📋 Disease Information", expanded=is_reliable):
                                if not is_reliable:
                                    st.warning("⚠️ Take this information with caution due to low confidence / কম নিশ্চয়তার কারণে সতর্কতার সাথে দেখুন")
                                st.markdown(f"**Description:** {info['description']}")
                                st.markdown(f"**Symptoms:** {info['symptoms']}")
                                st.markdown(f"**Treatment:** {info['treatment']}")
                                st.markdown(f"**Severity Level:** {info['severity']}")
                        
                        # Display all predictions
                        with st.expander("📊 All Class Probabilities", expanded=not is_reliable):
                            sorted_predictions = dict(sorted(all_predictions.items(), key=lambda x: x[1], reverse=True))
                            for class_name, prob in sorted_predictions.items():
                                st.progress(prob / 100)
                                st.write(f"**{class_name.replace('_', ' ')}**: {prob:.2f}%")
                            
                            st.info(f"Confidence gap between top 2 predictions: {confidence_gap:.2f}%")
                        
                    except Exception as e:
                        st.error(f"❌ Error during prediction: {str(e)}")
    
    else:
        # Welcome / empty state
        st.markdown("""
        <div class="glass-card" style="text-align: center; padding: 2.5rem 1.5rem;">
            <div style="font-size: 3rem; margin-bottom: 0.5rem;">📸</div>
            <h3 style="margin: 0.3rem 0; font-weight: 700;">Upload a Sunflower Leaf Image to Begin</h3>
            <p style="opacity: 0.7; font-size: 0.9rem; margin: 0.5rem 0;">👆 Drop your image above or click to browse · JPG, JPEG, PNG supported</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="section-title">
            <span class="icon">📚</span>
            <span>Detectable Disease Categories</span>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.markdown("""
            <div class="disease-card">
                <span class="card-icon">🟤</span>
                <div class="card-name">Alternaria Leaf Spot</div>
                <div class="card-desc">Dark brown concentric ring lesions</div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown("""
            <div class="disease-card">
                <span class="card-icon">🟠</span>
                <div class="card-name">Downy Mildew</div>
                <div class="card-desc">Yellow patches, white fuzzy growth</div>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown("""
            <div class="disease-card">
                <span class="card-icon">🟢</span>
                <div class="card-name">Healthy</div>
                <div class="card-desc">Green, vibrant, no lesions</div>
            </div>
            """, unsafe_allow_html=True)
        with col4:
            st.markdown("""
            <div class="disease-card">
                <span class="card-icon">⚪</span>
                <div class="card-name">Powdery Mildew</div>
                <div class="card-desc">White powdery patches on leaves</div>
            </div>
            """, unsafe_allow_html=True)
        with col5:
            st.markdown("""
            <div class="disease-card">
                <span class="card-icon">🟤</span>
                <div class="card-name">Wilted Leaves</div>
                <div class="card-desc">Drooping, curling, dry leaves</div>
            </div>
            """, unsafe_allow_html=True)

    # Footer
    st.markdown("""
    <div class="footer">
        <div class="footer-brand">🌻 Sunflower Disease Detection System</div>
        <div class="footer-copy">Built with Streamlit & TensorFlow · © 2026</div>
        <div class="footer-note">
            ⚠️ <b>Important Note:</b> এটি একটি AI-based diagnostic tool। গুরুত্বপূর্ণ সিদ্ধান্তের জন্য কৃষি বিশেষজ্ঞ বা উদ্ভিদ রোগবিদের সাথে পরামর্শ করুন।
        </div>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
