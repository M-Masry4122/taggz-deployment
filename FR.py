import cv2
import numpy as np
import os
import time
from scipy.spatial.distance import cosine
from insightface.app import FaceAnalysis
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple, Optional, Dict, Union
import logging
from dataclasses import dataclass
from pathlib import Path
import pickle
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
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
        cache_dir: Optional[str] = None,
    ):
        """Initialize the face recognition system with configurable parameters.

        Args:
            quality_threshold: Minimum image quality score to consider (higher is better)
            similarity_threshold: Minimum similarity score for face matching (higher is stricter)
            min_face_size: Minimum size in pixels for a valid face
            cache_dir: Directory to store cached face embeddings
        """
        self.quality_threshold = quality_threshold
        self.similarity_threshold = similarity_threshold
        self.min_face_size = min_face_size
        self.cache_dir = Path(cache_dir) if cache_dir else None

        # Initialize face analyzer with both CPU support
        self.face_analyzer = FaceAnalysis(
            name="buffalo_l", providers=["CPUExecutionProvider"]
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
                logger.debug("No face detected during alignment")
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
            aligned_image = cv2.warpAffine(
                image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
            )

            return aligned_image

        except Exception as e:
            logger.error(f"Error during face alignment: {str(e)}")
            return image

    def _load_and_preprocess_image(
        self, image_path: Union[str, Path]
    ) -> Optional[np.ndarray]:
        """Load and preprocess image with error handling and validation.

        Args:
            image_path: Path to the image file

        Returns:
            Preprocessed image array or None if loading failed
        """
        try:
            image_path = str(image_path)  # Convert Path to string if needed

            # Check if file exists before attempting to read
            if not os.path.exists(image_path):
                logger.error(f"Image file does not exist: {image_path}")
                return None

            image = cv2.imread(image_path)
            if image is None:
                logger.error(f"Failed to load image: {image_path}")
                return None

            # Convert to RGB and check dimensions
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            if (
                image.shape[0] < self.min_face_size
                or image.shape[1] < self.min_face_size
            ):
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

    def compute_face_quality(
        self, image: np.ndarray, image_path: Optional[str] = None
    ) -> float:
        """Compute image quality score and log poor quality images.

        Args:
            image: The image array to analyze
            image_path: Optional path to the image for logging purposes

        Returns:
            Quality score (higher is better)
        """
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
            if image_path and (
                quality_score <= self.quality_threshold
                or not is_brightness_ok
                or not is_contrast_ok
            ):
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

    def extract_faces(self, image_path: Union[str, Path]) -> List:
        """Extract faces from an image and log if no faces are detected.

        Args:
            image_path: Path to the image file

        Returns:
            List of detected faces with embeddings
        """
        image = self._load_and_preprocess_image(image_path)
        if image is None:
            return []

        # Run quality check
        quality_score = self.compute_face_quality(image, str(image_path))
        if quality_score <= self.quality_threshold:
            logger.warning(f"Low quality image: {image_path}")

        # Align face before feature extraction
        aligned_image = self._align_face(image)

        # Detect faces
        faces = self.face_analyzer.get(aligned_image)

        # Log if no faces are detected
        if not faces:
            image_path_str = str(image_path)
            if image_path_str not in self.no_face_images:
                self.no_face_images.append(image_path_str)
                print(f"No face detected: {os.path.basename(image_path_str)}")

        return faces

    def _process_single_image(self, image_path: Union[str, Path]) -> Optional[Dict]:
        """Process a single image and return its face embeddings and metadata.

        Args:
            image_path: Path to the image file

        Returns:
            Dictionary with image path, embeddings, bounding boxes and face areas
            or None if processing failed
        """
        # Use extract_faces method to ensure logging of no faces
        faces = self.extract_faces(image_path)
        if not faces:
            return None

        # Get embeddings and metadata
        return {
            "path": str(image_path),
            "embeddings": [face.normed_embedding for face in faces],
            "bboxes": [face.bbox for face in faces],
            "areas": [
                (box[2] - box[0]) * (box[3] - box[1])
                for box in [face.bbox for face in faces]
            ],
        }

    def get_cache_path(self, image_folder: Union[str, Path]) -> Optional[Path]:
        """Get the appropriate cache file path for an image folder.

        Args:
            image_folder: Path to the folder containing images

        Returns:
            Path object for the cache file or None if caching is disabled
        """
        if not self.cache_dir:
            return None

        # Derive event name from the image folder path
        event_name = os.path.basename(os.path.normpath(str(image_folder)))

        # Construct cache file path specific to the event
        cache_file = Path(self.cache_dir) / f"{event_name}/{event_name}.pkl"

        # Ensure the event-specific directory exists
        cache_file.parent.mkdir(parents=True, exist_ok=True)

        return cache_file

    def rebuild_face_database(self, image_folder: Union[str, Path]) -> Dict:
        """
        Rebuild the face database with improved cache handling.
        Ensures that the old cache is only deleted after successful database creation.

        Args:
            image_folder: Path to the folder containing images

        Returns:
            Dictionary mapping image paths to face data
        """
        # Clear the tracking lists
        self.poor_quality_images = []
        self.no_face_images = []

        # Get cache path
        cache_file = self.get_cache_path(image_folder)

        # Create a temporary cache file name
        if cache_file:
            temp_cache_file = cache_file.with_suffix(".pkl.tmp")

            # Load new database first
            new_database = self.load_face_database(image_folder, cache=False)

            # Save to temporary file
            try:
                with open(temp_cache_file, "wb") as f:
                    pickle.dump(new_database, f)

                # If successful, remove old cache and rename temp file
                if cache_file.exists():
                    cache_file.unlink()
                temp_cache_file.rename(cache_file)
                logger.info(
                    f"Successfully updated cache for {os.path.basename(os.path.normpath(str(image_folder)))}"
                )
            except Exception as e:
                logger.error(f"Error updating cache: {str(e)}")
                # Clean up temporary file if it exists
                if temp_cache_file.exists():
                    temp_cache_file.unlink()
        else:
            # If no cache directory is specified, just load the database
            new_database = self.load_face_database(image_folder, cache=False)

        return new_database

    def load_face_database(
        self, image_folder: Union[str, Path], cache: bool = True
    ) -> Dict:
        """Build or load a database of face embeddings from an image folder.

        Args:
            image_folder: Path to the folder containing images
            cache: Whether to use caching

        Returns:
            Dictionary mapping image paths to face data
        """
        # Clear tracking lists if not rebuilding
        if not (hasattr(self, "poor_quality_images") and self.poor_quality_images):
            self.poor_quality_images = []
            self.no_face_images = []

        # Get cache path
        cache_file = self.get_cache_path(image_folder)

        # Try loading from cache
        if cache and cache_file and cache_file.exists():
            try:
                with open(cache_file, "rb") as f:
                    logger.info(f"Loading face database from cache: {cache_file}")
                    return pickle.load(f)
            except Exception as e:
                logger.error(f"Error loading cache: {str(e)}")

        # Process images in parallel
        image_paths = [
            os.path.join(str(image_folder), f)
            for f in os.listdir(str(image_folder))
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]

        logger.info(f"Processing {len(image_paths)} images in {image_folder}")

        # Use ThreadPoolExecutor with progress bar
        with ThreadPoolExecutor() as executor:
            # Submit all tasks and get futures
            future_to_path = {
                executor.submit(self._process_single_image, path): path
                for path in image_paths
            }

            # Process results with progress bar
            results = []
            for future in tqdm(future_to_path, desc="Processing images", unit="image"):
                try:
                    result = future.result()
                    if result is not None:
                        results.append(result)
                except Exception as e:
                    path = future_to_path[future]
                    logger.error(f"Error processing {path}: {str(e)}")

        # Build database
        database = {
            result["path"]: {
                "embeddings": result["embeddings"],
                "bboxes": result["bboxes"],
                "areas": result["areas"],
            }
            for result in results
            if result is not None
        }

        # Save to cache if enabled
        if cache and cache_file:
            try:
                with open(cache_file, "wb") as f:
                    pickle.dump(database, f)
                logger.info(f"Saved face database to cache: {cache_file}")
            except Exception as e:
                logger.error(f"Error saving cache: {str(e)}")

        # Print summary of problematic images
        if self.poor_quality_images:
            logger.info(
                f"Identified {len(self.poor_quality_images)} poor quality images"
            )
        if self.no_face_images:
            logger.info(
                f"Identified {len(self.no_face_images)} images with no faces detected"
            )

        # Send notification when database is successfully rebuilt
        if not cache or not cache_file or not cache_file.exists():
            try:
                import requests

                event_id = getattr(self, "event_id", None)
                if event_id:
                    url = f"https://admin.taggz.app/api/send-notification/{event_id}"
                    payload = {"no_uploaded": len(database)}
                    response = requests.post(url, json=payload)
                    if response.status_code == 200:
                        logger.info(
                            f"Successfully sent notification for {len(database)} uploaded images"
                        )
                    else:
                        logger.error(
                            f"Failed to send notification: {response.status_code}"
                        )
            except Exception as e:
                logger.error(f"Error sending notification: {str(e)}")
        else:
            logger.info("Using cached face database, no notification sent")

        logger.info(f"Face database created with {len(database)} valid images")
        return database

    def find_matching_faces(
        self,
        query_image_path: Union[str, Path],
        face_database: Dict,
        top_k: int = 500,
        min_area: Optional[int] = None,
    ) -> List[FaceMatch]:
        """Find faces matching the query image in the database.

        Args:
            query_image_path: Path to the query image
            face_database: Database of face embeddings
            top_k: Maximum number of matches to return
            min_area: Minimum face area to consider (filter small faces)

        Returns:
            List of FaceMatch objects sorted by confidence
        """
        query_data = self._process_single_image(query_image_path)
        if not query_data:
            logger.error("No valid face found in query image")
            return []

        query_embedding = query_data["embeddings"][0]  # Use first face as query
        matches = []

        # Use tqdm for progress tracking
        for image_path, data in tqdm(
            face_database.items(), desc="Matching faces", unit="image"
        ):
            for idx, embedding in enumerate(data["embeddings"]):
                face_area = data["areas"][idx]

                # Skip if face is too small
                if min_area and face_area < min_area:
                    continue

                similarity = 1 - cosine(query_embedding, embedding)
                if similarity > self.similarity_threshold:
                    matches.append(
                        FaceMatch(
                            image_path=image_path,
                            confidence=similarity,
                            bbox=data["bboxes"][idx],
                            face_area=face_area,
                        )
                    )

        # Sort by confidence and return top-k results
        matches.sort(key=lambda x: x.confidence, reverse=True)
        logger.info(
            f"Found {len(matches)} matches with confidence > {self.similarity_threshold}"
        )
        return matches

    def export_match_results(
        self, matches: List[FaceMatch], output_path: Union[str, Path]
    ) -> None:
        """Export match results to a file.

        Args:
            matches: List of FaceMatch objects
            output_path: Path to save the results
        """
        output_path = Path(output_path)

        # Ensure directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Write results to file
        with open(output_path, "w") as f:
            f.write("Image Path,Confidence,Face Area\n")
            for match in matches:
                f.write(
                    f"{match.image_path},{match.confidence:.4f},{match.face_area}\n"
                )

        logger.info(f"Exported {len(matches)} matches to {output_path}")

    def visualize_matches(
        self,
        query_image_path: Union[str, Path],
        matches: List[FaceMatch],
        output_dir: Union[str, Path],
        max_matches: int = 10,
    ) -> None:
        """Visualize matches by creating a grid of matched faces.

        Args:
            query_image_path: Path to the query image
            matches: List of FaceMatch objects
            output_dir: Directory to save visualization
            max_matches: Maximum number of matches to visualize
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Load query image
        query_img = cv2.imread(str(query_image_path))
        query_img = cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB)

        # Create a grid of matches
        rows = min(4, (len(matches) + 2) // 3)
        cols = min(3, len(matches))

        # Resize query image
        height, width = 200, 200
        query_img_resized = cv2.resize(query_img, (width, height))

        # Create grid
        grid = np.ones((height * rows, width * cols, 3), dtype=np.uint8) * 255

        # Add query image at top
        grid[:height, :width] = query_img_resized

        # Add matches
        for i, match in enumerate(matches[:max_matches]):
            if i >= rows * cols - 1:  # Save space for query image
                break

            # Load match image
            match_img = cv2.imread(match.image_path)
            if match_img is None:
                continue

            match_img = cv2.cvtColor(match_img, cv2.COLOR_BGR2RGB)

            # Extract face using bbox if available
            if match.bbox is not None:
                x1, y1, x2, y2 = match.bbox
                # Add some margin
                margin = int((x2 - x1) * 0.2)
                x1 = max(0, x1 - margin)
                y1 = max(0, y1 - margin)
                x2 = min(match_img.shape[1], x2 + margin)
                y2 = min(match_img.shape[0], y2 + margin)
                face_img = match_img[y1:y2, x1:x2]
            else:
                face_img = match_img

            # Resize to fit grid
            face_img_resized = cv2.resize(face_img, (width, height))

            # Calculate position in grid
            row = (i + 1) // cols
            col = (i + 1) % cols

            # Place in grid
            grid[row * height : (row + 1) * height, col * width : (col + 1) * width] = (
                face_img_resized
            )

            # Add confidence text
            confidence_text = f"{match.confidence:.2f}"
            cv2.putText(
                grid,
                confidence_text,
                (col * width + 10, row * height + 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )

        # Save grid
        query_name = os.path.splitext(os.path.basename(query_image_path))[0]
        output_path = output_dir / f"{query_name}_matches.jpg"

        # Convert back to BGR for saving
        grid_bgr = cv2.cvtColor(grid, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(output_path), grid_bgr)

        logger.info(f"Visualization saved to {output_path}")


# Example usage
if __name__ == "__main__":
    start_time = time.time()

    # Initialize face recognition system
    face_recognition = EnhancedFaceRecognition(
        quality_threshold=10.0,
        similarity_threshold=0.37,
        min_face_size=10,
        cache_dir="./cache",
    )

    # Build or load face database
    database = face_recognition.load_face_database(image_folder="./dataset", cache=True)

    # Find matches for a query image
    matches = face_recognition.find_matching_faces(
        query_image_path="query.jpg",
        face_database=database,
        top_k=500,
    )

    # Export results
    face_recognition.export_match_results(
        matches=matches, output_path="./results/matches.csv"
    )

    # Visualize matches
    face_recognition.visualize_matches(
        query_image_path="query.jpg",
        matches=matches,
        output_dir="./results/visualizations",
        max_matches=10,
    )

    end_time = time.time()
    print(f"Time taken: {end_time - start_time:.2f} seconds")

    # Print results
    for idx, match in enumerate(matches, 1):  # Show only top 10 matches
        logger.info(f"Match {idx}:")
        logger.info(f"  Image: {match.image_path}")
        logger.info(f"  Confidence: {match.confidence:.2%}")
        logger.info(f"  Face Area: {match.face_area}")
