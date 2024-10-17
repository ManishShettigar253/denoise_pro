from flask import Flask, request, jsonify, render_template
import os
import cv2
import numpy as np
import subprocess

app = Flask(__name__)

# Set upload and denoised folder paths
UPLOAD_FOLDER = 'static/uploads'
DENNOISED_FOLDER = 'static/denoised'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DENNOISED_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def extract_audio(video_path, audio_output_path):
    """Extract audio from video using ffmpeg"""
    command = f"ffmpeg -i \"{video_path}\" -q:a 0 -map a \"{audio_output_path}\""
    subprocess.run(command, shell=True, check=True)

def analyze_video(video_path, frame_sample_rate=10):
    """Analyze video for noise levels"""
    cap = cv2.VideoCapture(video_path)
    gaussian_noise_total = 0
    salt_pepper_noise_total = 0
    total_frames = 0

    if not cap.isOpened():
        raise Exception("Could not open video for analysis.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Only analyze every nth frame
        if total_frames % frame_sample_rate == 0:
            total_frames += 1

            # Gaussian noise analysis (calculating variance)
            gaussian_noise_total += np.var(frame)

            # Salt-and-pepper noise analysis (counting noisy pixels)
            noisy_pixels = np.sum((frame == 0) | (frame == 255))  # Count white (255) and black (0) pixels
            salt_pepper_noise_total += noisy_pixels / (frame.size) * 100  # Percentage of noisy pixels

        total_frames += 1  # Increment total frames for sampled frames

    cap.release()

    # Calculate average noise levels
    if total_frames > 0:
        average_gaussian_noise = gaussian_noise_total / total_frames
        average_salt_pepper_noise = salt_pepper_noise_total / total_frames
    else:
        average_gaussian_noise = 0
        average_salt_pepper_noise = 0

    # Convert Gaussian noise from variance to a percentage (0-100 scale)
    gaussian_noise_percentage = min(average_gaussian_noise / 255 * 100, 100)  # Assuming pixel values are from 0-255

    return {
        'gaussian_noise': gaussian_noise_percentage,
        'salt_pepper_noise': average_salt_pepper_noise,
    }


def denoise_video(video_path):
    """Denoise video while preserving audio"""
    original_file_name = os.path.splitext(os.path.basename(video_path))[0]
    output_path = os.path.join(DENNOISED_FOLDER, f"{original_file_name}_denoised_temp.mp4")
    audio_path = os.path.join(DENNOISED_FOLDER, 'audio.mp3')  # Temp audio file path

    try:
        # Extract audio from the original video
        extract_audio(video_path, audio_path)

        # Open the video using OpenCV
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise Exception("Could not open video.")

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))

        frame_count = 0  # Counter to sample every nth frame

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Apply denoising only on every nth frame
            if frame_count % 5 == 0:  # Change this number for more or less frequent processing
                denoised_frame = cv2.fastNlMeansDenoisingColored(frame, None, 10, 10, 7, 21)
                out.write(denoised_frame)

            frame_count += 1

        cap.release()
        out.release()

        # Combine the denoised video with the original audio
        final_output_path = os.path.join(DENNOISED_FOLDER, f"{original_file_name}_denoised.mp4")
        command = f"ffmpeg -i \"{output_path}\" -i \"{audio_path}\" -c:v copy -c:a aac \"{final_output_path}\""
        subprocess.run(command, shell=True, check=True)

        # Remove the temporary audio file and temp video file
        os.remove(audio_path)
        os.remove(output_path)

        return final_output_path

    except Exception as e:
        print(f"Error: {e}")
        return None


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'video' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file:
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(video_path)

        # Analyze video
        noise_data = analyze_video(video_path)
        return jsonify(noise_data), 200

@app.route('/denoise', methods=['POST'])
def denoise():
    video_name = request.form['video_name']
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], video_name)

    denoised_video_path = denoise_video(video_path)
    if denoised_video_path is None:
        return jsonify({'error': 'Failed to denoise video.'}), 500

    return jsonify(original=video_path, denoised=denoised_video_path), 200

if __name__ == '__main__':
    app.run(debug=True)
