# 🌽 Corn Disease Detection Web Application

AI-powered web application for detecting diseases in corn leaves using InceptionV3 deep learning model.

## 🚀 Features

- **Real-time Disease Detection**: Upload corn leaf images and get instant predictions
- **5 Disease Classes**: 
  - Common Rust
  - Corn Leaf Blight
  - Gray Leaf Spot
  - Healthy
  - Maize Chlorotic Mottle Virus
- **Beautiful UI**: Modern, responsive design with gradient effects
- **Detailed Results**: 
  - Prediction confidence scores
  - Disease information and symptoms
  - Treatment recommendations
  - Probability distribution for all classes

## 📋 Requirements

- Python 3.8 or higher
- TensorFlow 2.15.0
- Streamlit 1.28.0
- Other dependencies (see requirements.txt)

## 🔧 Installation

1. **Clone or navigate to the project directory**
```bash
cd /media/panda/Data1/leaf_ditection
```

2. **Install required packages**
```bash
pip install -r requirements.txt
```

## 🎯 Usage

1. **Make sure your trained model (.h5 file) is in the same directory**
   - The app will automatically look for:
     - `best_inceptionv3_corn_full_training.h5`
     - `inceptionv3_corn_disease_full_training.h5`
     - `lightning_studio_inceptionv3_corn_disease_full_training.h5`

2. **Run the Streamlit app**
```bash
streamlit run app.py
```

3. **Open your browser**
   - The app will automatically open at `http://localhost:8501`
   - If not, manually open the URL shown in the terminal

4. **Upload and analyze**
   - Click "Browse files" or drag & drop a corn leaf image
   - Click "Analyze Image" button
   - View the prediction results and disease information

## 📁 Project Structure

```
leaf_ditection/
│
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── InceptionV3_Corn.ipynb         # Model training notebook
└── *.h5                           # Trained model files
```

## 🎨 Features Explained

### Image Upload
- Supports JPG, JPEG, and PNG formats
- Displays original image with details
- Shows image dimensions and format

### Disease Detection
- Uses InceptionV3 model trained on corn disease dataset
- Provides confidence score for prediction
- Shows probability distribution for all classes

### Disease Information
- Detailed description of each disease
- Common symptoms to look for
- Recommended treatment methods

## 🛠️ Troubleshooting

### Model not loading
- Ensure your .h5 model file is in the same directory as app.py
- Check that the model file name matches one of the expected names

### Dependencies error
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Port already in use
```bash
streamlit run app.py --server.port 8502
```

## 📊 Model Information

- **Architecture**: InceptionV3 (Transfer Learning)
- **Input Size**: 299 x 299 pixels
- **Training Dataset**: Augmented corn leaf disease dataset
- **Classes**: 5 disease categories

## 🤝 Support

For issues or questions, please check:
1. All dependencies are installed correctly
2. Model file is in the correct location
3. Python version is 3.8 or higher

## 📝 License

This project is for educational and research purposes.

---

**Built with ❤️ using Streamlit and TensorFlow**
# leaf-detection
