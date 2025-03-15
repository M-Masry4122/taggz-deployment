from flask import Flask, request, jsonify, send_from_directory
import os
from werkzeug.utils import secure_filename
from pathlib import Path
import logging
import time
import numpy as np
from FR import EnhancedFaceRecognition
import threading
from queue import Queue

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configuration
# app.config["UPLOAD_FOLDER"] = "uploads/"  # For local development
app.config["UPLOAD_FOLDER"] = "/var/www/html/public_html/storage/app/public/"  # For production
app.config["PROFILE_PICTURES_FOLDER"] = os.path.join(
    app.config["UPLOAD_FOLDER"], "user/"
)
app.config["EVENTS_FOLDER"] = os.path.join(app.config["UPLOAD_FOLDER"], "event_photos")
app.config["CACHE_FOLDER"] = os.path.join(app.config["UPLOAD_FOLDER"], "cache")
app.config["ALLOWED_EXTENSIONS"] = {"png", "jpg", "jpeg"}
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB max upload size

# Add a global queue for database build requests
database_build_queue = Queue()
# Track enqueued events to prevent duplicates
enqueued_events = set()
# Track current database build
current_database_build = None
# Lock for thread-safe access to enqueued_events
queue_lock = threading.Lock()

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


def background_database_builder():
    """
    Background thread to process database build requests
    Allows other requests to be processed while building database
    """
    global current_database_build
    while True:
        # Wait for a new build request
        build_request = database_build_queue.get()
        
        try:
            event_name = build_request['event_name']
            event_path = build_request['event_path']

            logger.info(f"Background database build for {event_name} started")
            current_database_build = {
                'event_name': event_name,
                'status': 'in_progress',
                'start_time': time.time()
            }

            # Perform database build
            start_time = time.time()
            face_recognition.rebuild_face_database(event_path)
            end_time = time.time()

            # Collect problematic image lists
            poor_quality_images = [
                os.path.basename(img) for img in face_recognition.poor_quality_images
            ]
            no_face_images = [
                os.path.basename(img) for img in face_recognition.no_face_images
            ]

            # Log successful build
            logger.info(
                f"Background database build for event {event_name} completed. "
                f"Time taken: {end_time - start_time:.2f} seconds"
            )

            # Store results for status check
            current_database_build = {
                'event_name': event_name,
                'status': 'completed',
                'time_taken': f"{end_time - start_time:.2f} seconds",
                'poor_quality_images': poor_quality_images,
                'no_face_images': no_face_images,
                'completed_at': time.time()
            }

            logger.info(f"Build completed: {current_database_build}")

        except Exception as e:
            logger.error(f"Error in background database build: {str(e)}")
            current_database_build = {
                'event_name': event_name,
                'status': 'failed',
                'error': str(e),
                'completed_at': time.time()
            }
        
        finally:
            # Remove event from enqueued set
            with queue_lock:
                if event_name in enqueued_events:
                    enqueued_events.remove(event_name)
            
            # Mark the task as done
            database_build_queue.task_done()

# Start the background database builder thread
database_build_thread = threading.Thread(target=background_database_builder, daemon=True)
database_build_thread.start()

@app.route("/build_database", methods=["POST"])
def build_database():
    """
    Build or rebuild the face database for a specific event
    Input: Event name
    Output: Status message
    """
    try:
        data = request.json
        if not data or "event_name" not in data:
            return jsonify({"error": "Event name is required"}), 400

        event_name = secure_filename(data["event_name"])
        event_path = os.path.join(app.config["EVENTS_FOLDER"], event_name)

        if not os.path.exists(event_path):
            return jsonify({"error": f"Event {event_name} does not exist"}), 404

        # Check if event is already in queue
        with queue_lock:
            if event_name in enqueued_events:
                return jsonify({
                    "status": "already_queued",
                    "message": f"Database build for event {event_name} is already in the queue",
                    "event_name": event_name
                }), 202

            # Check if this event is being processed right now
            if current_database_build and current_database_build.get('event_name') == event_name and current_database_build.get('status') == 'in_progress':
                return jsonify({
                    "status": "in_progress",
                    "message": f"Database build for event {event_name} is already in progress",
                    "event_name": event_name
                }), 202

            # Add to enqueued set and queue the build request
            enqueued_events.add(event_name)
            database_build_queue.put({
                'event_name': event_name, 
                'event_path': event_path
            })

        return jsonify(
            {
                "status": "queued",
                "message": f"Database build for event {event_name} has been queued",
                "event_name": event_name,
                "queue_length": database_build_queue.qsize(),
            }
        ), 202  # Accepted status code

    except Exception as e:
        logger.error(f"Error queueing database build: {str(e)}")
        return jsonify({"error": str(e)}), 500
    
@app.route("/database_build_status", methods=["GET"])
def database_build_status():
    """
    Check the status of the most recent database build
    """
    event_name = request.args.get('event_name')
    
    if event_name:
        # Check if the specific event is in the queue
        with queue_lock:
            if event_name in enqueued_events:
                return jsonify({
                    "status": "queued",
                    "event_name": event_name,
                    "position": list(enqueued_events).index(event_name) + 1,
                    "queue_length": len(enqueued_events)
                })
            
        # Check if this is the currently building event
        if current_database_build and current_database_build.get('event_name') == event_name:
            return jsonify(current_database_build)
        
        # Check for cached database
        cache_file = Path(app.config["CACHE_FOLDER"]) / f"{event_name}/{event_name}.pkl"
        if cache_file.exists():
            return jsonify({
                "status": "cached",
                "event_name": event_name,
                "last_modified": time.ctime(os.path.getmtime(cache_file))
            })
            
        return jsonify({
            "status": "not_found",
            "event_name": event_name
        })
    
    # Return current build status if no specific event is requested
    if current_database_build:
        # Calculate duration for in-progress builds
        if current_database_build.get('status') == 'in_progress':
            current_database_build['duration_so_far'] = f"{time.time() - current_database_build.get('start_time', time.time()):.2f} seconds"
        
        return jsonify({
            **current_database_build,
            "queue_length": database_build_queue.qsize(),
            "queued_events": list(enqueued_events)
        })
    else:
        return jsonify({
            "status": "no_recent_build",
            "queue_length": database_build_queue.qsize(),
            "queued_events": list(enqueued_events)
        })


@app.route("/compare_faces", methods=["POST"])
def compare_faces():
    """
    Compare a profile picture with photos from an event
    Input: Profile picture path and event name
    Output: List of paths of matched photos
    """
    try:
        data = request.json
        if not data or "profile_picture" not in data or "event_name" not in data:
            return (
                jsonify({"error": "Profile picture and event name are required"}),
                400,
            )

        profile_picture = data["profile_picture"]
        event_name = secure_filename(data["event_name"])

        # Validate profile picture path
        if not profile_picture.startswith(app.config["PROFILE_PICTURES_FOLDER"]):
            profile_picture = os.path.join(
                app.config["PROFILE_PICTURES_FOLDER"], profile_picture
            )

        if not os.path.exists(profile_picture):
            return jsonify({"error": "Profile picture not found"}), 404

        # Validate event path
        event_path = os.path.join(app.config["EVENTS_FOLDER"], event_name)
        if not os.path.exists(event_path):
            return jsonify({"error": f"Event {event_name} does not exist"}), 404

        # Check if cache exists for this event
        cache_file = Path(app.config["CACHE_FOLDER"]) / f"{event_name}/{event_name}.pkl"
        
        if not cache_file.exists():
            # Check if build is already queued
            with queue_lock:
                if event_name in enqueued_events:
                    return jsonify({
                        "status": "build_in_queue",
                        "message": f"Database build for event {event_name} is already queued",
                        "event_name": event_name,
                        "position": list(enqueued_events).index(event_name) + 1,
                        "queue_length": len(enqueued_events)
                    }), 202
                
                # Check if build is in progress
                if current_database_build and current_database_build.get('event_name') == event_name and current_database_build.get('status') == 'in_progress':
                    return jsonify({
                        "status": "build_in_progress",
                        "message": f"Database build for event {event_name} is in progress",
                        "event_name": event_name,
                        "duration_so_far": f"{time.time() - current_database_build.get('start_time', time.time()):.2f} seconds"
                    }), 202
                
                # Enqueue cache generation
                enqueued_events.add(event_name)
                database_build_queue.put({
                    'event_name': event_name,
                    'event_path': event_path
                })
            
            return jsonify({
                "status": "queued",
                "message": f"Database build for event {event_name} has been queued",
                "event_name": event_name,
                "queue_length": database_build_queue.qsize()
            }), 202  # Accepted status code

        # If cache exists, proceed with normal face comparison
        start_time = time.time()
        database = face_recognition.load_face_database(event_path, cache=True)

        # Check profile picture for faces
        faces = face_recognition.extract_faces(profile_picture)
        if not faces:
            return (
                jsonify(
                    {
                        "status": "error",
                        "error": "No face detected in profile picture",
                        "profile_picture": os.path.basename(profile_picture),
                    }
                ),
                400,
            )

        # Find matching faces
        matches = face_recognition.find_matching_faces(
            query_image_path=profile_picture,
            face_database=database,
            top_k=int(data.get("max_results", 500)),
        )
        end_time = time.time()

        # Format the results, converting numpy arrays to regular Python types for JSON serialization
        results = []
        for match in matches:
            match_dict = {
                "image_path": match.image_path,
                "confidence": float(match.confidence),
            }
            results.append(match_dict)

        # Print results in separate lines for logging
        for result in results:
            logger.debug(f"Match: {result}")

        logger.info(
            f"match_count: {len(results)}, time_taken: {end_time - start_time:.2f} seconds"
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
                "matches": results,
                "match_count": len(results),
                "time_taken": f"{end_time - start_time:.2f} seconds",
                "poor_quality_images": poor_quality_images,
                "no_face_images": no_face_images,
                "cache_age": time.ctime(os.path.getmtime(cache_file))
            }
        )

    except Exception as e:
        logger.error(f"Error comparing faces: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route("/upload_profile_picture", methods=["POST"])
def upload_profile_picture():
    """
    Upload a profile picture for a user
    Input: user_id and profile picture file
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

        # Save the file with user_id as filename
        filename = f"{user_id}.jpg"
        file_path = os.path.join(app.config["PROFILE_PICTURES_FOLDER"], filename)
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

        return jsonify({
            "status": status, 
            "message": message, 
            "file_path": file_path,
            "face_count": len(faces) if faces else 0
        })

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
        skipped_files = []
        face_counts = {}
        
        for file in files:
            if allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file_path = os.path.join(event_path, filename)
                file.save(file_path)
                saved_files.append(filename)

                # Check image quality and face detection
                faces = face_recognition.extract_faces(file_path)
                face_counts[filename] = len(faces) if faces else 0
            else:
                logger.warning(f"Skipping file with unsupported extension: {file.filename}")
                skipped_files.append(file.filename)

        # Get lists of problematic images
        poor_quality_images = [
            os.path.basename(img) for img in face_recognition.poor_quality_images
        ]
        no_face_images = [
            os.path.basename(img) for img in face_recognition.no_face_images
        ]
        
        # Automatically queue a database build if files were added
        if saved_files and request.form.get("auto_build", "true").lower() != "false":
            with queue_lock:
                if event_id not in enqueued_events:
                    enqueued_events.add(event_id)
                    database_build_queue.put({
                        'event_name': event_id,
                        'event_path': event_path
                    })
                    build_status = "queued"
                else:
                    build_status = "already_queued"
        else:
            build_status = "not_requested"

        return jsonify(
            {
                "status": "success",
                "message": f"{len(saved_files)} photos uploaded successfully to event {event_id}",
                "saved_files": saved_files,
                "skipped_files": skipped_files,
                "face_counts": face_counts,
                "poor_quality_images": poor_quality_images,
                "no_face_images": no_face_images,
                "database_build": build_status
            }
        )

    except Exception as e:
        logger.error(f"Error uploading event photos: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route("/get_photo/<path:filename>")
def get_photo(filename):
    """
    Serve a photo file
    Input: File path
    Output: The requested image file
    """
    try:
        # Determine if the file is a profile picture or an event photo
        if filename.startswith("profile_pictures/") or filename.startswith("user/"):
            # Handle both legacy paths and new paths
            if filename.startswith("profile_pictures/"):
                filename = filename.replace("profile_pictures/", "user/")
            directory = os.path.dirname(os.path.join(app.config["UPLOAD_FOLDER"], filename))
            filename = os.path.basename(filename)
        elif filename.startswith("events/") or filename.startswith("event_photos/"):
            # Handle both legacy paths and new paths
            if filename.startswith("events/"):
                filename = filename.replace("events/", "event_photos/")
            parts = filename.split("/", 2)
            if len(parts) < 3:
                return jsonify({"error": "Invalid file path"}), 400
            directory = os.path.join(app.config["UPLOAD_FOLDER"], parts[0], parts[1])
            filename = parts[2]
        else:
            return jsonify({"error": "Invalid file path"}), 400

        if not os.path.exists(os.path.join(directory, filename)):
            return jsonify({"error": "File not found"}), 404

        return send_from_directory(directory, filename)

    except Exception as e:
        logger.error(f"Error serving photo: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route("/quality_issues/<event_name>", methods=["GET"])
def quality_issues(event_name):
    """
    Get a list of images with quality issues for a specific event
    Input: Event name
    Output: Lists of problematic images
    """
    try:
        event_name = secure_filename(event_name)
        event_path = os.path.join(app.config["EVENTS_FOLDER"], event_name)

        if not os.path.exists(event_path):
            return jsonify({"error": f"Event {event_name} does not exist"}), 404

        # Check if cache exists
        cache_file = Path(app.config["CACHE_FOLDER"]) / f"{event_name}/{event_name}.pkl"
        if not cache_file.exists():
            # Try to load the database to run quality checks
            try:
                database = face_recognition.load_face_database(event_path, cache=False)
            except Exception as e:
                logger.warning(f"Error loading database for quality check: {str(e)}")
                # Handle missing database by scanning all images in event directory
                face_recognition.poor_quality_images = []
                face_recognition.no_face_images = []
                for img_file in os.listdir(event_path):
                    if allowed_file(img_file):
                        img_path = os.path.join(event_path, img_file)
                        face_recognition.extract_faces(img_path)
        else:
            # Load the database from cache
            database = face_recognition.load_face_database(event_path, cache=True)

        # Get lists of problematic images
        poor_quality_images = [
            os.path.basename(img)
            for img in face_recognition.poor_quality_images
            if event_name in img
        ]
        no_face_images = [
            os.path.basename(img)
            for img in face_recognition.no_face_images
            if event_name in img
        ]
        
        # Get total event images count
        total_images = len([f for f in os.listdir(event_path) if allowed_file(f)])

        return jsonify(
            {
                "event": event_name,
                "poor_quality_images": poor_quality_images,
                "no_face_images": no_face_images,
                "total_problematic": len(poor_quality_images) + len(no_face_images),
                "total_images": total_images,
                "quality_percentage": round(100 * (1 - (len(poor_quality_images) + len(no_face_images)) / max(1, total_images)), 2)
            }
        )

    except Exception as e:
        logger.error(f"Error getting quality issues: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/list_events", methods=["GET"])
def list_events():
    """
    List all available events
    Output: List of event names and metadata
    """
    try:
        events = []
        events_dir = app.config["EVENTS_FOLDER"]
        
        for event_name in os.listdir(events_dir):
            event_path = os.path.join(events_dir, event_name)
            if os.path.isdir(event_path):
                # Count images in the event
                image_count = len([f for f in os.listdir(event_path) if allowed_file(f)])
                
                # Check if database is built
                cache_file = Path(app.config["CACHE_FOLDER"]) / f"{event_name}/{event_name}.pkl"
                database_built = cache_file.exists()
                
                # Check if event is in queue
                with queue_lock:
                    is_queued = event_name in enqueued_events
                    is_current = (current_database_build and 
                                 current_database_build.get('event_name') == event_name and 
                                 current_database_build.get('status') == 'in_progress')
                
                events.append({
                    "name": event_name,
                    "image_count": image_count,
                    "database_built": database_built,
                    "in_queue": is_queued,
                    "in_progress": is_current,
                    "created": time.ctime(os.path.getctime(event_path)),
                    "modified": time.ctime(os.path.getmtime(event_path)),
                    "size_mb": round(sum(os.path.getsize(os.path.join(event_path, f)) 
                                     for f in os.listdir(event_path) if os.path.isfile(os.path.join(event_path, f))) / (1024 * 1024), 2)
                })
        
        return jsonify({
            "events": events,
            "total_count": len(events)
        })
        
    except Exception as e:
        logger.error(f"Error listing events: {str(e)}")
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
                "cache_size_mb": round(sum(
                    os.path.getsize(os.path.join(dirpath, filename))
                    for dirpath, _, filenames in os.walk(app.config["CACHE_FOLDER"])
                    for filename in filenames
                ) / (1024 * 1024), 2),
                "queue_size": database_build_queue.qsize(),
                "current_build": current_database_build.get('event_name') if current_database_build else None
            },
        }
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5500, debug=True)