# Models Directory

This directory contains the trained deepfake detection model.

## Required Files

### `best_unified_model.pth`
This is the trained ResNet-LSTM-Transformer model weights file that should be placed in this directory.

**To obtain this file:**
1. Run the training notebook (`Unified-Deepfake-Training.ipynb`) in Google Colab
2. Download the `best_unified_model.pth` file from your Colab environment
3. Place it in this directory: `backend/models/best_unified_model.pth`

## Model Architecture

The model uses a unified architecture that can handle both images and videos:

- **Images**: ResNet50 → Classifier
- **Videos**: ResNet50 → LSTM → Transformer → Classifier

## File Structure

```
backend/models/
├── README.md                    # This file
├── deepfake_model.py           # Model architecture and inference code
└── best_unified_model.pth      # Trained model weights (you need to add this)
```

## Setup Instructions

1. **Download the trained model:**
   - From your Colab notebook, download `best_unified_model.pth`
   - Place it in this directory

2. **Verify the file:**
   - The file should be approximately 100-200 MB in size
   - The filename must be exactly `best_unified_model.pth`

3. **Test the setup:**
   - Start the backend server: `python main.py`
   - Check the health endpoint: `GET /health`
   - Verify `model_loaded` is `true`

## Troubleshooting

- **Model not found error**: Ensure the file is named exactly `best_unified_model.pth`
- **CUDA errors**: The model will automatically use CPU if CUDA is not available
- **Import errors**: Make sure all dependencies are installed via `pip install -r requirements.txt`
