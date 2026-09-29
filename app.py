from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import uuid

from optical_flow import process_video

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
OUTPUT_FOLDER = os.path.join(BASE_DIR, "outputs")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["OUTPUT_FOLDER"] = OUTPUT_FOLDER


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "success": True,
        "message": "Farneback Optical Flow Backend is running"
    })


@app.route("/process", methods=["POST"])
def process():
    if "video" not in request.files:
        return jsonify({
            "success": False,
            "error": "No video uploaded"
        }), 400

    video = request.files["video"]

    if video.filename == "":
        return jsonify({
            "success": False,
            "error": "No video selected"
        }), 400

    noise_type = request.form.get("noise_type", "Gaussian")
    filter_type = request.form.get("filter_type", "Gaussian")

    try:
        kernel_size = int(request.form.get("kernel_size", 5))
    except ValueError:
        return jsonify({
            "success": False,
            "error": "Invalid kernel size"
        }), 400

    file_id = str(uuid.uuid4())

    extension = os.path.splitext(video.filename)[1]

    if not extension:
        extension = ".mp4"

    input_filename = file_id + extension
    output_filename = file_id + "_optical_flow.mp4"

    input_path = os.path.join(
        UPLOAD_FOLDER,
        input_filename
    )

    output_path = os.path.join(
        OUTPUT_FOLDER,
        output_filename
    )

    video.save(input_path)

    try:
        statistics = process_video(
            input_path,
            output_path,
            noise_type,
            filter_type,
            kernel_size
        )

        return jsonify({
            "success": True,
            "message": "Video processed successfully",
            "output_video": output_filename,
            "statistics": statistics
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

    finally:
        if os.path.exists(input_path):
            os.remove(input_path)


@app.route("/output/<filename>", methods=["GET"])
def output_video(filename):
    return send_from_directory(
        OUTPUT_FOLDER,
        filename
    )


if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )