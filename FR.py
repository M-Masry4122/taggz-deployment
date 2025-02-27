import cv2
import numpy as np
import os
import torch
import time
from torchvision import transforms
from PIL import Image
from scipy.spatial.distance import cosine
from insightface.app import FaceAnalysis
import warnings
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple, Optional, Dict
import logging
from dataclasses import dataclass
from pathlib import Path
import pickle

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class FaceMatch:
    """Data class to store face match results"""
    image_path: str
    confidence: float
    bbox: Optional[Tuple[int, int, int, int]] = None
    face_area: Optional[int] = None

class EnhancedFaceRecognition:
    def __init__(
        self,
        quality_threshold: float = 60.0,
        similarity_threshold: float = 0.45,
        min_face_size: int = 50,
        cache_dir: Optional[str] = None
    ):
        """Initialize the face recognition system with configurable parameters."""
        self.quality_threshold = quality_threshold
        self.similarity_threshold = similarity_threshold
        self.min_face_size = min_face_size
        self.cache_dir = Path(cache_dir) if cache_dir else None
        
        # Initialize face analyzer with both CPU support
        self.face_analyzer = FaceAnalysis(
            name='buffalo_l',
            providers=['CPUExecutionProvider']
        )
        self.face_analyzer.prepare(ctx_id=0)
        
        # Create cache directory if specified
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            
        # Track problematic images
        self.poor_quality_images = []
        self.no_face_images = []

    def _align_face(self, image: np.ndarray) -> np.ndarray:
        """Align face to correct for rotation using detected landmarks."""
        try:
            faces = self.face_analyzer.get(image)
            if not faces:
                logger.warning("No face detected during alignment")
                return image
            
            face = faces[0]  # Use the first detected face
            kps = face.kps  # Get facial keypoints
            
            # Calculate angle based on eye positions
            left_eye, right_eye = kps[0], kps[1]
            dY = right_eye[1] - left_eye[1]
            dX = right_eye[0] - left_eye[0]
            angle = np.degrees(np.arctan2(dY, dX))
            
            # Perform rotation
            h, w = image.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            aligned_image = cv2.warpAffine(image, M, (w, h), 
                                         flags=cv2.INTER_CUBIC,
                                         borderMode=cv2.BORDER_REPLICATE)
            
            return aligned_image
            
        except Exception as e:
            logger.error(f"Error during face alignment: {str(e)}")
            return image

    def _load_and_preprocess_image(self, image_path: str) -> Optional[np.ndarray]:
        """Load and preprocess image with error handling and validation."""
        try:
            image = cv2.imread(str(image_path))
            if image is None:
                logger.error(f"Failed to load image: {image_path}")
                return None
                
            # Convert to RGB and check dimensions
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            if image.shape[0] < self.min_face_size or image.shape[1] < self.min_face_size:
                logger.warning(f"Image too small: {image_path}")
                self.poor_quality_images.append(image_path)
                print(f"Image too small: {os.path.basename(image_path)}")
                return None
                
            return image
            
        except Exception as e:
            logger.error(f"Error processing image {image_path}: {str(e)}")
            self.poor_quality_images.append(image_path)
            print(f"Error processing image: {os.path.basename(image_path)}")
            return None

    def compute_face_quality(self, image: np.ndarray, image_path: Optional[str] = None) -> float:
        """Compute image quality score and log poor quality images."""
        try:
            # Check blur using Laplacian variance
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            variance = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            # Check brightness
            brightness = np.mean(gray)
            is_brightness_ok = 40 <= brightness <= 250
            
            # Check contrast
            contrast = np.std(gray)
            is_contrast_ok = contrast >= 20
            
            # Get overall quality score
            quality_score = variance
            
            # Log poor quality image if path is provided
            if image_path and (quality_score <= self.quality_threshold or 
                              not is_brightness_ok or 
                              not is_contrast_ok):
                if image_path not in self.poor_quality_images:
                    self.poor_quality_images.append(image_path)
                
                # Print specific quality issues
                if quality_score <= self.quality_threshold:
                    print(f"Blurry image: {os.path.basename(image_path)}")
                if not is_brightness_ok:
                    print(f"Poor brightness image: {os.path.basename(image_path)}")
                if not is_contrast_ok:
                    print(f"Low contrast image: {os.path.basename(image_path)}")
            
            return quality_score
            
        except Exception as e:
            logger.error(f"Error in quality check: {str(e)}")
            if image_path and image_path not in self.poor_quality_images:
                self.poor_quality_images.append(image_path)
                print(f"Error in quality check: {os.path.basename(image_path)}")
            return 0.0

    def extract_faces(self, image_path: str) -> List:
        """Extract faces from an image and log if no faces are detected."""
        image = self._load_and_preprocess_image(image_path)
        if image is None:
            return []
        
        # Run quality check
        quality_score = self.compute_face_quality(image, image_path)
        if quality_score <= self.quality_threshold:
            logger.warning(f"Low quality image: {image_path}")
        
        # Align face before feature extraction
        aligned_image = self._align_face(image)
        
        # Detect faces
        faces = self.face_analyzer.get(aligned_image)
        
        # Log if no faces are detected
        if not faces:
            if image_path not in self.no_face_images:
                self.no_face_images.append(image_path)
                print(f"No face detected: {os.path.basename(image_path)}")
        
        return faces

    def _process_single_image(self, image_path: str) -> Optional[Dict]:
        """Process a single image and return its face embeddings and metadata."""
        # Use extract_faces method to ensure logging of no faces
        faces = self.extract_faces(image_path)
        if not faces:
            return None
        
        # Get embeddings and metadata
        return {
            'path': image_path,
            'embeddings': [face.normed_embedding for face in faces],
            'bboxes': [face.bbox for face in faces],
            'areas': [(box[2]-box[0])*(box[3]-box[1]) for box in [face.bbox for face in faces]]
        }
    
    def rebuild_face_database(self, image_folder):
        """Rebuild the face database and clear cache."""
        # Clear the tracking lists
        self.poor_quality_images = []
        self.no_face_images = []
        
        cache_file = self.cache_dir / 'face_database.pkl' if self.cache_dir else None
        
        if cache_file and cache_file.exists():
            logger.info(f"Deleting cache file: {cache_file}")
            cache_file.unlink()
        
        return self.load_face_database(image_folder)

    def load_face_database(self, image_folder: str, cache: bool = True) -> Dict:
        """Build or load a database of face embeddings from an image folder."""
        # Clear tracking lists if not rebuilding (rebuild already clears them)
        if not (hasattr(self, 'poor_quality_images') and self.poor_quality_images):
            self.poor_quality_images = []
            self.no_face_images = []
            
        cache_file = self.cache_dir / 'face_database.pkl' if self.cache_dir else None
        
        # Try loading from cache
        if cache and cache_file and cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                logger.error(f"Error loading cache: {str(e)}")
        
        # Process images in parallel
        image_paths = [os.path.join(image_folder, f) for f in os.listdir(image_folder)
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        with ThreadPoolExecutor() as executor:
            results = list(executor.map(self._process_single_image, image_paths))
        
        # Filter None results and build database
        database = {result['path']: {
            'embeddings': result['embeddings'],
            'bboxes': result['bboxes'],
            'areas': result['areas']
        } for result in results if result is not None}
        
        # Save to cache if enabled
        if cache and cache_file:
            try:
                with open(cache_file, 'wb') as f:
                    pickle.dump(database, f)
            except Exception as e:
                logger.error(f"Error saving cache: {str(e)}")
        
        # Print summary of problematic images
        if self.poor_quality_images:
            logger.info(f"Identified {len(self.poor_quality_images)} poor quality images")
        if self.no_face_images:
            logger.info(f"Identified {len(self.no_face_images)} images with no faces detected")
        
        return database

    def find_matching_faces(
        self,
        query_image_path: str,
        face_database: Dict,
        top_k: int = 500
    ) -> List[FaceMatch]:
        """Find faces matching the query image in the database."""
        query_data = self._process_single_image(query_image_path)
        if not query_data:
            logger.error("No valid face found in query image")
            return []
        
        query_embedding = query_data['embeddings'][0]  # Use first face as query
        matches = []
        
        for image_path, data in face_database.items():
            for idx, embedding in enumerate(data['embeddings']):
                similarity = 1 - cosine(query_embedding, embedding)
                if similarity > self.similarity_threshold:
                    matches.append(FaceMatch(
                        image_path=image_path,
                        confidence=similarity,
                        bbox=data['bboxes'][idx],
                        face_area=data['areas'][idx]
                    ))
        
        # Sort by confidence and return top-k results
        matches.sort(key=lambda x: x.confidence, reverse=True)
        return matches[:top_k]

# Example usage
if __name__ == "__main__":
    start_time = time.time()
    # Initialize face recognition system
    face_recognition = EnhancedFaceRecognition(
        quality_threshold=10.0,
        similarity_threshold=0.37,
        min_face_size=10,
        cache_dir="./cache"
    )
    
    # Build or load face database
    database = face_recognition.load_face_database(
        image_folder="./dataset",
        cache=True
    )
    
    # Find matches for a query image
    matches = face_recognition.find_matching_faces(
        query_image_path="query.jpg",
        face_database=database,
        top_k=500
    )
    end_time = time.time()
    print(f"Time taken: {end_time - start_time:.2f} seconds")
    
    # Print results
    for idx, match in enumerate(matches, 1):
        logger.info(f"Match {idx}:")
        logger.info(f"  Image: {match.image_path}")
        logger.info(f"  Confidence: {match.confidence:.2%}")
        logger.info(f"  Face Area: {match.face_area}")