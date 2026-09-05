---
title: Sunflower Disease Detection
emoji: 🌻
colorFrom: green
colorTo: yellow
sdk: streamlit
sdk_version: "1.28.0"
app_file: app.py
pinned: false
---

# 🌻 Sunflower Disease Detection Web Application

AI-powered web application for detecting diseases in sunflower leaves using deep learning (MobileNetV2).

## 🚀 Features

- **Real-time Disease Detection**: Upload sunflower leaf images and get instant predictions
- **5 Disease Classes**:
  - Alternaria Leaf Spot
  - Downy Mildew
  - Healthy
  - Powdery Mildew
  - Wilted Leaves
- **Smart Image Validation**: Automatically rejects non-leaf images (sky, walls, skin, etc.)
- **Modern UI**: Glassmorphism design with animated gradients
- **Detailed Results**:
  - Prediction confidence scores
  - Disease information and symptoms
  - Treatment recommendations
  - Probability distribution for all classes

## 📋 Requirements

- Python 3.8 or higher
- TensorFlow
- Streamlit
- Other dependencies (see requirements.txt)

## 🔧 Installation

1. **Clone the repository**
```bash
git clone https://github.com/0xzahed/sunflower-web-app.git
cd sunflower-web-app
```

2. **Create virtual environment and install packages**
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## 🎯 Usage

1. **Make sure the trained model (.h5 file) is in the same directory**
   - The app uses: `MobilenetV2_corn_disease_full_training.h5`

2. **Run the Streamlit app**
```bash
streamlit run app.py
```

3. **Open your browser**
   - The app will automatically open at `http://localhost:8501`
   - If not, manually open the URL shown in the terminal

4. **Upload and analyze**
   - Click "Browse files" or drag & drop a sunflower leaf image
   - Click "Analyze Image" button
   - View the prediction results and disease information

## 📁 Project Structure

```
sunflower-web-app/
├── app.py                                    # Main Streamlit application
├── requirements.txt                          # Python dependencies
├── README.md                                 # This file
├── MobilenetV2_corn_disease_full_training.h5 # Trained model
├── run_app.sh                                # Setup & run script
└── deploy.sh                                 # Deployment script
```

## 🎨 Features Explained

### Image Upload
- Supports JPG, JPEG, and PNG formats
- Displays original image with details
- Shows image dimensions and format

### Disease Detection
- Uses MobileNetV2 transfer learning model
- Provides confidence score for prediction
- Shows probability distribution for all classes

### Image Validation
- Multi-feature validation (color, texture, plant-like pixels)
- Rejects non-leaf images (sky, walls, paper, skin, soil, flat colors)
- Provides detailed validation metrics

### Disease Information
- Detailed description of each disease (English + Bangla)
- Common symptoms to look for
- Recommended treatment methods

## 🛠️ Troubleshooting

### Model not loading
- Ensure the .h5 model file is in the same directory as app.py
- Check that the model file name matches the expected name

### Dependencies error
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Port already in use
```bash
streamlit run app.py --server.port 8502
```

## 🤝 Support

For issues or questions, please check:
1. All dependencies are installed correctly
2. Model file is in the correct location
3. Python version is 3.8 or higher

## 📝 License

This project is for educational and research purposes.

---

**Built with Streamlit and TensorFlow**
