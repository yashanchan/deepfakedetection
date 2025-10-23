# Deepfake Detection Backend

FastAPI backend for deepfake detection using PyTorch ResNet-LSTM-Transformer model.

## Setup

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Add your model:**
   - Place your trained model weights in `models/deepfake_detector.pth`
   - Update the model loading logic in `main.py` (search for TODO comments)

3. **Implement your model:**
   - Replace the `DeepfakeModel` class placeholder with your actual ResNet-LSTM-Transformer implementation
   - Update preprocessing functions to match your model's requirements
   - Implement the prediction logic with your trained weights

## Running the Server

**Development:**
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

**Production:**
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

## API Endpoints

### `POST /analyze`
Upload and analyze media for deepfake detection.

**Request:**
- Method: `POST`
- Content-Type: `multipart/form-data`
- Body: `file` (image or video file)

**Response:**
```json
{
  "prediction": "REAL" | "FAKE",
  "confidence": 0.85,
  "probability_fake": 0.75
}
```

### `GET /health`
Check server health and model status.

## Model Integration

Replace the placeholder code in `DeepfakeModel` class:

```python
def __init__(self, model_path: str = None):
    # Load your ResNet-LSTM-Transformer model
    from your_model import YourDeepfakeModel
    
    self.model = YourDeepfakeModel()
    if model_path and Path(model_path).exists():
        self.model.load_state_dict(
            torch.load(model_path, map_location=self.device)
        )
    self.model.to(self.device)
    self.model.eval()
```

## Frontend Configuration

Add backend URL to frontend `.env`:
```
VITE_API_URL=http://localhost:8000
```

For production, update CORS origins in `main.py`.

## Notes

- Supports images: PNG, JPG, JPEG (max 10MB)
- Supports videos: MP4, AVI, MOV (max 100MB)
- Uses MediaPipe for face detection
- GPU acceleration automatically enabled if available
