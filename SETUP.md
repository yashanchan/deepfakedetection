# Deepfake Detection System - Setup Guide

This guide will help you set up the complete deepfake detection system with the trained model.

## 📋 Prerequisites

- Python 3.8 or higher
- Node.js 16 or higher (for frontend)
- CUDA-compatible GPU (optional, for faster inference)

## 🚀 Quick Setup

### 1. Download the Trained Model

**Important**: You need to download the trained model file from your Colab training session.

1. Open your `Unified-Deepfake-Training.ipynb` notebook in Google Colab
2. After training completes, download the `best_unified_model.pth` file
3. Place it in: `deepdetect-guard/backend/models/best_unified_model.pth`

### 2. Backend Setup

```bash
# Navigate to backend directory
cd deepdetect-guard/backend

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start the backend server
python main.py
```

The backend will start on `http://localhost:8000`

### 3. Frontend Setup

```bash
# Navigate to frontend directory
cd deepdetect-guard/frontend

# Install dependencies
npm install

# Start the development server
npm run dev
```

The frontend will start on `http://localhost:5173`

## 🔧 Configuration

### Model Configuration

The model is configured in `backend/models/deepfake_model.py`:

- **Image input size**: 224x224 pixels
- **Video frames**: 20 frames per video
- **Face detection**: MediaPipe with 0.5 confidence threshold
- **Confidence thresholds**: 
  - Real: < 0.4
  - Uncertain: 0.4 - 0.6
  - Fake: > 0.6

### Backend Configuration

Key settings in `backend/main.py`:

- **File size limits**: 10MB for images, 100MB for videos
- **Supported formats**: PNG, JPEG (images); MP4, AVI, MOV, MKV (videos)
- **Device**: Automatically detects CUDA/CPU

## 🧪 Testing the Setup

### 1. Test Backend Health

```bash
curl http://localhost:8000/health
```

Expected response:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cuda:0",
  "cuda_available": true,
  "model_path": "models/best_unified_model.pth"
}
```

### 2. Test Model Inference

```bash
# Test with an image
curl -X POST "http://localhost:8000/analyze" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@path/to/your/image.jpg"
```

Expected response:
```json
{
  "prediction": "REAL",
  "confidence": 0.85,
  "probability_fake": 0.15,
  "media_type": "image",
  "file_name": "image.jpg"
}
```

### 3. Test Frontend

1. Open `http://localhost:5173` in your browser
2. Upload an image or video file
3. Wait for the analysis to complete
4. View the prediction results

## 📁 Project Structure

```
deepdetect-guard/
├── backend/
│   ├── models/
│   │   ├── deepfake_model.py      # Model architecture
│   │   ├── best_unified_model.pth # Trained weights (you add this)
│   │   └── README.md              # Model setup instructions
│   ├── main.py                    # FastAPI server
│   └── requirements.txt           # Python dependencies
├── frontend/
│   ├── src/                       # React frontend source
│   ├── public/                    # Static assets
│   ├── node_modules/              # Node.js dependencies
│   ├── package.json               # Node.js dependencies
│   └── vite.config.ts             # Vite configuration
└── SETUP.md                       # This file
```

## 🐛 Troubleshooting

### Common Issues

1. **Model not found error**
   ```
   Model file not found at: models/best_unified_model.pth
   ```
   **Solution**: Ensure you've downloaded and placed the model file correctly.

2. **CUDA out of memory**
   ```
   RuntimeError: CUDA out of memory
   ```
   **Solution**: The model will automatically fall back to CPU if CUDA memory is insufficient.

3. **No face detected**
   ```
   No face detected in the media
   ```
   **Solution**: Ensure your image/video contains clearly visible faces.

4. **Import errors**
   ```
   ModuleNotFoundError: No module named 'models'
   ```
   **Solution**: Make sure you're running the backend from the correct directory.

### Performance Optimization

1. **GPU Acceleration**: Install CUDA-compatible PyTorch for faster inference
2. **Batch Processing**: For multiple files, consider implementing batch processing
3. **Model Quantization**: For deployment, consider model quantization to reduce memory usage

## 🔒 Security Considerations

- **File Upload Limits**: Configure appropriate file size limits for your use case
- **Input Validation**: The system validates file types and sizes
- **Temporary Files**: Uploaded files are automatically cleaned up after processing

## 📊 Model Performance

The trained model provides:
- **Unified Architecture**: Handles both images and videos
- **Face Detection**: Uses MediaPipe for robust face detection
- **Confidence Scoring**: Provides uncertainty ranges for predictions
- **Real-time Processing**: Optimized for web deployment

## 🚀 Deployment

For production deployment:

1. **Environment Variables**: Set up proper environment configuration
2. **CORS Settings**: Update CORS origins for your domain
3. **File Storage**: Consider using cloud storage for uploaded files
4. **Load Balancing**: Use multiple backend instances for high traffic
5. **Monitoring**: Implement logging and monitoring for production use

## 📞 Support

If you encounter issues:

1. Check the troubleshooting section above
2. Verify all dependencies are correctly installed
3. Ensure the model file is properly placed
4. Check the backend logs for detailed error messages
