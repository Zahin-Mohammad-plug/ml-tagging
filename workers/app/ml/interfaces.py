"""ML interfaces and model abstractions"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from pathlib import Path

class VisionEncoder(ABC):
    """Abstract base class for vision encoders (CLIP, etc.)"""
    
    @abstractmethod
    def encode_image(self, image_path: str) -> np.ndarray:
        """Encode a single image to embedding vector"""
        pass
    
    @abstractmethod
    def encode_images_batch(self, image_paths: List[str]) -> np.ndarray:
        """Encode multiple images in a batch"""
        pass
    
    @abstractmethod
    def encode_text(self, text: str) -> np.ndarray:
        """Encode text to embedding vector"""
        pass
    
    @abstractmethod
    def compute_similarity(self, image_embedding: np.ndarray, text_embedding: np.ndarray) -> float:
        """Compute cosine similarity between image and text embeddings"""
        pass
    
    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """Embedding dimension size"""
        pass

class ASREngine(ABC):
    """Abstract base class for Automatic Speech Recognition"""
    
    @abstractmethod
    def transcribe_audio(self, audio_path: str, language: Optional[str] = None) -> Dict[str, Any]:
        """
        Transcribe audio file to text
        
        Returns:
            {
                "text": str,
                "segments": List[Dict], # with timestamps
                "language": str,
                "confidence": float
            }
        """
        pass
    
    @abstractmethod
    def transcribe_audio_chunk(
        self, 
        audio_data: np.ndarray, 
        sample_rate: int,
        language: Optional[str] = None
    ) -> Dict[str, Any]:
        """Transcribe audio data chunk"""
        pass

class OCREngine(ABC):
    """Abstract base class for Optical Character Recognition"""
    
    @abstractmethod
    def extract_text(self, image_path: str, languages: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Extract text from image
        
        Returns:
            {
                "text": str,
                "boxes": List[Dict], # bounding boxes with confidence
                "confidence": float
            }
        """
        pass
    
    @abstractmethod
    def extract_text_batch(self, image_paths: List[str], languages: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Extract text from multiple images"""
        pass

class CLIPVisionEncoder(VisionEncoder):
    """CLIP model for visual content understanding"""
    
    def __init__(self, model_name: str = "clip-vit-base-patch32", device: str = "auto"):
        self.model_name = model_name
        self.device = device
        self.model = None
        self.processor = None
        self._load_model()
    
    def _load_model(self):
        """Load the CLIP model"""
        import torch
        from transformers import CLIPModel, CLIPProcessor
        
        # Determine device
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load base CLIP model
        # TODO: Replace with domain-specific fine-tuned version
        self.model = CLIPModel.from_pretrained(f"openai/{self.model_name}")
        self.processor = CLIPProcessor.from_pretrained(f"openai/{self.model_name}")
        
        self.model.to(self.device)
        self.model.eval()
    
    def encode_image(self, image_path: str) -> np.ndarray:
        """Encode single image"""
        from PIL import Image
        import torch
        
        image = Image.open(image_path).convert('RGB')
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)
            # Normalize embeddings
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
        return image_features.cpu().numpy().flatten()
    
    def encode_images_batch(self, image_paths: List[str]) -> np.ndarray:
        """Encode multiple images in batch"""
        from PIL import Image
        import torch
        
        images = [Image.open(path).convert('RGB') for path in image_paths]
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)
            # Normalize embeddings
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
        return image_features.cpu().numpy()
    
    def encode_text(self, text: str) -> np.ndarray:
        """Encode text to embedding"""
        import torch
        
        inputs = self.processor(text=text, return_tensors="pt", padding=True, truncation=True).to(self.device)
        
        with torch.no_grad():
            text_features = self.model.get_text_features(**inputs)
            # Normalize embeddings
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
        return text_features.cpu().numpy().flatten()
    
    def compute_similarity(self, image_embedding: np.ndarray, text_embedding: np.ndarray) -> float:
        """Compute cosine similarity"""
        return float(np.dot(image_embedding, text_embedding))
    
    @property
    def embedding_dim(self) -> int:
        """CLIP embedding dimension"""
        return 512  # Standard CLIP dimension

class WhisperASR(ASREngine):
    """Whisper-based ASR implementation"""
    
    def __init__(self, model_name: str = "small", device: str = "auto"):
        self.model_name = model_name
        self.device = device
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load Whisper model"""
        import whisper
        import torch
        
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            
        self.model = whisper.load_model(self.model_name, device=self.device)
    
    def transcribe_audio(self, audio_path: str, language: Optional[str] = None) -> Dict[str, Any]:
        """Transcribe audio file"""
        
        # Whisper transcription options
        options = {
            "task": "transcribe",
            "language": language,
            "word_timestamps": True,
            "condition_on_previous_text": False
        }
        
        result = self.model.transcribe(audio_path, **options)
        
        # Format segments with timestamps
        segments = []
        for segment in result.get("segments", []):
            segments.append({
                "start": segment["start"],
                "end": segment["end"],
                "text": segment["text"].strip(),
                "confidence": getattr(segment, "avg_logprob", 0.0)
            })
        
        return {
            "text": result["text"].strip(),
            "segments": segments,
            "language": result.get("language", "unknown"),
            "confidence": np.mean([s["confidence"] for s in segments]) if segments else 0.0
        }
    
    def transcribe_audio_chunk(
        self, 
        audio_data: np.ndarray, 
        sample_rate: int,
        language: Optional[str] = None
    ) -> Dict[str, Any]:
        """Transcribe audio data chunk"""
        import tempfile
        import soundfile as sf
        
        # Save audio data to temporary file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            sf.write(tmp_file.name, audio_data, sample_rate)
            
            # Transcribe the temporary file
            result = self.transcribe_audio(tmp_file.name, language)
            
            # Clean up
            Path(tmp_file.name).unlink()
            
        return result

class PaddleOCR_Engine(OCREngine):
    """PaddleOCR implementation"""
    
    def __init__(self, languages: List[str] = ["en"]):
        self.languages = languages
        self.ocr = None
        self._load_model()
    
    def _load_model(self):
        """Load PaddleOCR model"""
        from paddleocr import PaddleOCR
        
        # Initialize PaddleOCR
        self.ocr = PaddleOCR(
            use_angle_cls=True, 
            lang=self.languages[0] if self.languages else "en",
            show_log=False
        )
    
    def extract_text(self, image_path: str, languages: Optional[List[str]] = None) -> Dict[str, Any]:
        """Extract text from image"""
        
        result = self.ocr.ocr(image_path, cls=True)
        
        if not result or not result[0]:
            return {"text": "", "boxes": [], "confidence": 0.0}
        
        # Parse results
        text_parts = []
        boxes = []
        confidences = []
        
        for line in result[0]:
            bbox, (text, confidence) = line
            
            if confidence > 0.5:  # Filter low-confidence text
                text_parts.append(text)
                boxes.append({
                    "bbox": bbox,
                    "text": text,
                    "confidence": confidence
                })
                confidences.append(confidence)
        
        return {
            "text": " ".join(text_parts),
            "boxes": boxes,
            "confidence": np.mean(confidences) if confidences else 0.0
        }
    
    def extract_text_batch(self, image_paths: List[str], languages: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Extract text from multiple images"""
        return [self.extract_text(path, languages) for path in image_paths]

class EasyOCR_Engine(OCREngine):
    """EasyOCR implementation as alternative"""
    
    def __init__(self, languages: List[str] = ["en"]):
        self.languages = languages
        self.reader = None
        self._load_model()
    
    def _load_model(self):
        """Load EasyOCR model"""
        import easyocr
        
        self.reader = easyocr.Reader(self.languages, gpu=True)
    
    def extract_text(self, image_path: str, languages: Optional[List[str]] = None) -> Dict[str, Any]:
        """Extract text from image"""
        
        results = self.reader.readtext(image_path)
        
        text_parts = []
        boxes = []
        confidences = []
        
        for bbox, text, confidence in results:
            if confidence > 0.5:  # Filter low-confidence text
                text_parts.append(text)
                boxes.append({
                    "bbox": bbox,
                    "text": text,
                    "confidence": confidence
                })
                confidences.append(confidence)
        
        return {
            "text": " ".join(text_parts),
            "boxes": boxes,
            "confidence": np.mean(confidences) if confidences else 0.0
        }
    
    def extract_text_batch(self, image_paths: List[str], languages: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Extract text from multiple images"""
        return [self.extract_text(path, languages) for path in image_paths]

# Model factory functions
def create_vision_encoder(model_type: str = "clip", **kwargs) -> VisionEncoder:
    """Create vision encoder instance"""
    
    if model_type == "clip":
        return CLIPVisionEncoder(**kwargs)
    else:
        raise ValueError(f"Unknown vision encoder type: {model_type}")

def create_asr_engine(engine_type: str = "whisper", **kwargs) -> ASREngine:
    """Create ASR engine instance"""
    
    if engine_type == "whisper":
        return WhisperASR(**kwargs)
    else:
        raise ValueError(f"Unknown ASR engine type: {engine_type}")

def create_ocr_engine(engine_type: str = "paddleocr", **kwargs) -> OCREngine:
    """Create OCR engine instance"""
    
    if engine_type == "paddleocr":
        return PaddleOCR_Engine(**kwargs)
    elif engine_type == "easyocr":
        return EasyOCR_Engine(**kwargs)
    else:
        raise ValueError(f"Unknown OCR engine type: {engine_type}")