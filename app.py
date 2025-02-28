from flask import Flask, request, jsonify, send_from_directory
import os
from werkzeug.utils import secure_filename
from pathlib import Path
import logging
import time
import numpy as np
from FR import EnhancedFaceRecognition

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configuration
app.config["UPLOAD_FOLDER"] = "uploads"
app.config["PROFILE_PICTURES_FOLDER"] = os.path.join(
    app.config["UPLOAD_FOLDER"], "profile_pictures"
)
app.config["EVENTS_FOLDER"] = os.path.join(app.config["UPLOAD_FOLDER"], "events")
app.config["CACHE_FOLDER"] = os.path.join(app.config["UPLOAD_FOLDER"], "cache")
app.config["ALLOWED_EXTENSIONS"] = {"png", "jpg", "jpeg"}
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB max upload size

# Initialize the face recognition system
face_recognition = EnhancedFaceRecognition(
    quality_threshold=10.0,
    similarity_threshold=0.37,
    min_face_size=10,
    cache_dir=app.config["CACHE_FOLDER"],
)

# Ensure all required directories exist
for folder in [
    app.config["UPLOAD_FOLDER"],
    app.config["PROFILE_PICTURES_FOLDER"],
    app.config["EVENTS_FOLDER"],
    app.config["CACHE_FOLDER"],
]:
    os.makedirs(folder, exist_ok=True)


def allowed_file(filename):
    """Check if the file extension is allowed"""
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in app.config["ALLOWED_EXTENSIONS"]
    )


# Helper function to convert numpy arrays to lists for JSON serialization
def numpy_to_json_serializable(obj):
    """Convert numpy arrays to lists for JSON serialization"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, (list, tuple)):
        return [numpy_to_json_serializable(item) for item in obj]
    elif isinstance(obj, dict):
        return {k: numpy_to_json_serializable(v) for k, v in obj.items()}
    return obj


@app.route("/build_database", methods=["POST"])
def build_database():
    """
    Build or rebuild the face database for a specific event
    Input: Event path (full path to event directory)
    Output: Status message
    """
    try:
        data = request.json
        if not data or "event_path" not in data:
            return jsonify({"error": "Event path is required"}), 400

        event_path = data["event_path"]

        # Validate that the path exists
        if not os.path.exists(event_path) or not os.path.isdir(event_path):
            return (
                jsonify([]),
                404,
            )

        # Build the database (this will also delete the old one if it exists)
        start_time = time.time()
        face_recognition.rebuild_face_database(event_path)
        end_time = time.time()

        # Get lists of problematic images
        poor_quality_images = [
            os.path.basename(img) for img in face_recognition.poor_quality_images
        ]
        no_face_images = [
            os.path.basename(img) for img in face_recognition.no_face_images
        ]

        return jsonify(
            {
                "status": "success",
                "message": f"Face database for event at {event_path} built successfully",
                "time_taken": f"{end_time - start_time:.2f} seconds",
                "poor_quality_images": poor_quality_images,
                "no_face_images": no_face_images,
            }
        )

    except Exception as e:
        logger.error(f"Error building database: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route("/compare_faces", methods=["POST"])
def compare_faces():
    """
    Compare a profile picture with photos from an event
    Input: Profile picture path (full path) and event path (full path)
    Output: List of paths of matched photos
    """
    try:
        data = request.json
        if not data or "profile_picture_path" not in data or "event_path" not in data:
            return (
                jsonify({"error": "Profile picture path and event path are required"}),
                400,
            )

        profile_picture_path = data["profile_picture_path"]
        event_path = data["event_path"]

        # Validate paths
        if not os.path.exists(profile_picture_path) or not os.path.isfile(
            profile_picture_path
        ):
            return jsonify({"error": "Profile picture not found or is not a file"}), 404

        if not os.path.exists(event_path) or not os.path.isdir(event_path):
            return (
                jsonify({"error": f"Event path does not exist or is not a directory"}),
                404,
            )

        # Load the face database for the event
        start_time = time.time()
        database = face_recognition.load_face_database(event_path, cache=True)

        # Check profile picture for faces
        faces = face_recognition.extract_faces(profile_picture_path)
        if not faces:
            return (
                jsonify(
                    {
                        "status": "error",
                        "error": "No face detected in profile picture",
                        "profile_picture": os.path.basename(profile_picture_path),
                    }
                ),
                400,
            )

        # Find matching faces
        matches = face_recognition.find_matching_faces(
            query_image_path=profile_picture_path,
            face_database=database,
            top_k=int(data.get("max_results", 500)),
        )
        end_time = time.time()

        # Format the results
        results = []
        for match in matches:
            match_dict = {
                "image_path": match.image_path,  # This will be a full path
                "confidence": float(match.confidence),
            }
            results.append(match_dict)

        # Get lists of problematic images
        poor_quality_images = [
            os.path.basename(img) for img in face_recognition.poor_quality_images
        ]
        no_face_images = [
            os.path.basename(img) for img in face_recognition.no_face_images
        ]

        return jsonify(
            {
                "status": "success",
                "matches": results,
                "match_count": len(results),
                "time_taken": f"{end_time - start_time:.2f} seconds",
                "poor_quality_images": poor_quality_images,
                "no_face_images": no_face_images,
            }
        )

    except Exception as e:
        logger.error(f"Error comparing faces: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route("/upload_profile_picture", methods=["POST"])
def upload_profile_picture():
    """
    Upload a profile picture for a user
    Input: user_id, profile picture file, and optional destination_path
    Output: Path to the uploaded profile picture
    """
    try:
        if "user_id" not in request.form:
            return jsonify({"error": "User ID is required"}), 400

        if "file" not in request.files:
            return jsonify({"error": "No file part"}), 400

        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "No selected file"}), 400

        if not allowed_file(file.filename):
            return (
                jsonify(
                    {
                        "error": f'File type not allowed. Use {", ".join(app.config["ALLOWED_EXTENSIONS"])}'
                    }
                ),
                400,
            )

        user_id = secure_filename(request.form["user_id"])

        # Use custom destination path if provided, otherwise use default
        destination_path = request.form.get(
            "destination_path", app.config["PROFILE_PICTURES_FOLDER"]
        )

        # Ensure the directory exists
        os.makedirs(destination_path, exist_ok=True)

        # Save the file with user_id as filename
        filename = f"{user_id}.jpg"
        file_path = os.path.join(destination_path, filename)
        file.save(file_path)

        # Check for faces and quality
        faces = face_recognition.extract_faces(file_path)
        status = "success"
        message = "Profile picture uploaded successfully"

        if not faces:
            status = "warning"
            message = "Profile picture uploaded, but no face was detected"
        elif file_path in face_recognition.poor_quality_images:
            status = "warning"
            message = "Profile picture uploaded, but image quality is poor"

        return jsonify({"status": status, "message": message, "file_path": file_path})

    except Exception as e:
        logger.error(f"Error uploading profile picture: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route("/upload_event_photos", methods=["POST"])
def upload_event_photos():
    """
    Upload photos for an event
    Input: event_id and photo files
    Output: Status message
    """
    try:
        if "event_id" not in request.form:
            return jsonify({"error": "Event ID is required"}), 400

        if "files" not in request.files:
            return jsonify({"error": "No file part"}), 400

        files = request.files.getlist("files")
        if not files or files[0].filename == "":
            return jsonify({"error": "No selected files"}), 400

        event_id = secure_filename(request.form["event_id"])
        event_path = os.path.join(app.config["EVENTS_FOLDER"], event_id)

        # Create event directory if it doesn't exist
        os.makedirs(event_path, exist_ok=True)

        # Clear any existing quality tracking
        face_recognition.poor_quality_images = []
        face_recognition.no_face_images = []

        # Save all files
        saved_files = []
        for file in files:
            if allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file_path = os.path.join(event_path, filename)
                file.save(file_path)
                saved_files.append(file_path)

                # Check image quality and face detection
                face_recognition.extract_faces(file_path)
            else:
                logger.warning(
                    f"Skipping file with unsupported extension: {file.filename}"
                )

        # Get lists of problematic images
        poor_quality_images = [
            os.path.basename(img) for img in face_recognition.poor_quality_images
        ]
        no_face_images = [
            os.path.basename(img) for img in face_recognition.no_face_images
        ]

        return jsonify(
            {
                "status": "success",
                "message": f"{len(saved_files)} photos uploaded successfully to event {event_id}",
                "saved_files": saved_files,
                "poor_quality_images": poor_quality_images,
                "no_face_images": no_face_images,
            }
        )

    except Exception as e:
        logger.error(f"Error uploading event photos: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route("/get_photo", methods=["POST"])
def get_photo():
    """
    Serve a photo file
    Input: Full file path
    Output: The requested image file
    """
    try:
        data = request.json
        if not data or "file_path" not in data:
            return jsonify({"error": "File path is required"}), 400

        file_path = data["file_path"]

        if not os.path.exists(file_path) or not os.path.isfile(file_path):
            return jsonify({"error": "File not found or is not a file"}), 404

        directory = os.path.dirname(file_path)
        filename = os.path.basename(file_path)

        return send_from_directory(directory, filename)

    except Exception as e:
        logger.error(f"Error serving photo: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route("/quality_issues", methods=["POST"])
def quality_issues():
    """
    Get a list of images with quality issues for a specific event
    Input: Event path (full path)
    Output: Lists of problematic images
    """
    try:
        data = request.json
        if not data or "event_path" not in data:
            return jsonify({"error": "Event path is required"}), 400

        event_path = data["event_path"]

        if not os.path.exists(event_path) or not os.path.isdir(event_path):
            return (
                jsonify({"error": f"Event path does not exist or is not a directory"}),
                404,
            )

        # Load the database to run quality checks
        database = face_recognition.load_face_database(event_path, cache=True)

        # Get lists of problematic images
        poor_quality_images = [
            img
            for img in face_recognition.poor_quality_images
            if img.startswith(event_path)
        ]
        no_face_images = [
            img for img in face_recognition.no_face_images if img.startswith(event_path)
        ]

        return jsonify(
            {
                "event_path": event_path,
                "poor_quality_images": poor_quality_images,
                "no_face_images": no_face_images,
                "total_problematic": len(poor_quality_images) + len(no_face_images),
            }
        )

    except Exception as e:
        logger.error(f"Error getting quality issues: {str(e)}")
        return jsonify({"error": str(e)}), 500


# Optional: Add a health check endpoint
@app.route("/health", methods=["GET"])
def health_check():
    return jsonify(
        {
            "status": "healthy",
            "api_version": "1.0",
            "system_info": {
                "profile_pictures": len(
                    os.listdir(app.config["PROFILE_PICTURES_FOLDER"])
                ),
                "events": len(os.listdir(app.config["EVENTS_FOLDER"])),
            },
        }
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5500, debug=True)
