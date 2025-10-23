"""
FastAPI Backend for Deepfake Detection
This server handles media uploads and runs inference using a PyTorch model.
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import torch
import cv2
import numpy as np
from pathlib import Path
import tempfile
import logging
from typing import Dict, Any
import mediapipe as mp

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Deepfake Detection API", version="1.0.0")

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update with your frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model
model = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# MediaPipe face detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)


class DeepfakeModel:
    """
    Placeholder for your ResNet-LSTM-Transformer model.
    Replace this with your actual model implementation.
    """
    
    def __init__(self, model_path: str = None):
        """
        Initialize your model here.
        
        Args:
            model_path: Path to your trained model weights
        """
        self.device = device
        logger.info(f"Using device: {self.device}")
        
        # TODO: Load your pre-trained model here
        # Example:
        # self.model = YourModelClass()
        # if model_path:
        #     self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        # self.model.to(self.device)
        # self.model.eval()
        
        logger.info("Model initialized (placeholder)")
    
    def preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """
        Preprocess image for model input.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Preprocessed tensor ready for model
        """
        # TODO: Implement your preprocessing pipeline
        # Example preprocessing:
        # - Resize to model input size (e.g., 224x224)
        # - Normalize with ImageNet stats
        # - Convert to tensor
        
        # Placeholder preprocessing
        image = cv2.resize(image, (224, 224))
        image = image.astype(np.float32) / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
        return image.to(self.device)
    
    def preprocess_video(self, video_path: str, max_frames: int = 32) -> torch.Tensor:
        """
        Extract and preprocess frames from video.
        
        Args:
            video_path: Path to video file
            max_frames: Maximum number of frames to extract
            
        Returns:
            Preprocessed tensor of video frames
        """
        cap = cv2.VideoCapture(video_path)
        frames = []
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Sample frames evenly throughout video
        frame_indices = np.linspace(0, frame_count - 1, max_frames, dtype=int)
        
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
        
        cap.release()
        
        if not frames:
            raise ValueError("No frames extracted from video")
        
        # TODO: Implement your video preprocessing
        # Process frames according to your model requirements
        
        # Placeholder: just return preprocessed first frame
        return self.preprocess_image(frames[0])
    
    def predict(self, input_tensor: torch.Tensor) -> Dict[str, Any]:
        """
        Run inference on preprocessed input.
        
        Args:
            input_tensor: Preprocessed input tensor
            
        Returns:
            Dictionary with prediction and confidence
        """
        # TODO: Replace with your actual model inference
        # Example:
        # with torch.no_grad():
        #     output = self.model(input_tensor)
        #     probability_fake = torch.sigmoid(output).item()
        
        # Placeholder: return mock prediction
        # Replace this with actual model output
        probability_fake = 0.75  # Mock value
        
        # Determine verdict based on probability
        if probability_fake < 0.4:
            verdict = "REAL"
        elif probability_fake > 0.6:
            verdict = "FAKE"
        else:
            # Edge case: classify based on which threshold is closer
            verdict = "FAKE" if probability_fake >= 0.5 else "REAL"
        
        return {
            "prediction": verdict,
            "confidence": probability_fake if verdict == "FAKE" else (1 - probability_fake),
            "probability_fake": probability_fake
        }


def detect_faces(image: np.ndarray) -> bool:
    """
    Detect if image contains faces using MediaPipe.
    
    Args:
        image: Input image as numpy array
        
    Returns:
        True if faces detected, False otherwise
    """
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_detection.process(image_rgb)
    return results.detections is not None and len(results.detections) > 0


@app.on_event("startup")
async def startup_event():
    """Initialize model on startup."""
    global model
    
    # TODO: Provide path to your trained model weights
    model_path = "models/deepfake_detector.pth"  # Update this path
    
    try:
        model = DeepfakeModel(model_path)
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        # Initialize with placeholder anyway
        model = DeepfakeModel()


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "online",
        "message": "Deepfake Detection API",
        "device": str(device)
    }


@app.post("/analyze")
async def analyze_media(file: UploadFile = File(...)):
    """
    Analyze uploaded image or video for deepfake detection.
    
    Args:
        file: Uploaded media file (image or video)
        
    Returns:
        JSON with prediction result and confidence
    """
    if not model:
        raise HTTPException(status_code=503, detail="Model not initialized")
    
    # Validate file type
    allowed_image_types = ["image/png", "image/jpeg", "image/jpg"]
    allowed_video_types = ["video/mp4", "video/avi", "video/mov"]
    
    if file.content_type not in allowed_image_types + allowed_video_types:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file.content_type}"
        )
    
    # Validate file size
    max_size = 100 * 1024 * 1024  # 100MB for videos
    if file.content_type in allowed_image_types:
        max_size = 10 * 1024 * 1024  # 10MB for images
    
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp_file:
            content = await file.read()
            
            if len(content) > max_size:
                raise HTTPException(
                    status_code=400,
                    detail=f"File too large. Max size: {max_size / (1024*1024):.0f}MB"
                )
            
            tmp_file.write(content)
            tmp_path = tmp_file.name
        
        # Process based on file type
        if file.content_type in allowed_image_types:
            # Read and validate image
            image = cv2.imread(tmp_path)
            if image is None:
                raise HTTPException(status_code=400, detail="Invalid image file")
            
            # Check for faces
            if not detect_faces(image):
                logger.warning("No faces detected in image")
                # You can choose to reject or continue with analysis
            
            # Preprocess and predict
            input_tensor = model.preprocess_image(image)
            result = model.predict(input_tensor)
            
        else:  # Video
            # Preprocess video and predict
            input_tensor = model.preprocess_video(tmp_path)
            result = model.predict(input_tensor)
        
        # Clean up temporary file
        Path(tmp_path).unlink()
        
        logger.info(f"Analysis complete: {result}")
        return JSONResponse(content=result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Analysis error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@app.get("/health")
async def health_check():
    """Detailed health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": str(device),
        "cuda_available": torch.cuda.is_available()
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
