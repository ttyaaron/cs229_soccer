import os
import numpy as np
import librosa
from moviepy.editor import VideoFileClip
import matplotlib.pyplot as plt
import cv2

def save_frame_as_image(video, frame_number, output_path):
    """
    Save a specific frame from the video as an image file.

    Args:
        video (VideoFileClip): Loaded video file.
        frame_number (int): Frame number to save as an image.
        output_path (str): Path to save the image file.
    """
    # Calculate the time in seconds for the frame
    time_in_seconds = frame_number / video.fps

    # Extract the frame at the specified time
    frame = video.get_frame(time_in_seconds)

    # Convert to a format suitable for saving with OpenCV (BGR format)
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    
    # Save the frame as a PNG file
    cv2.imwrite(output_path, frame_bgr)

# Step 1: Extract Audio from Video using MoviePy
def extract_audio_from_video(video_path):
    video = VideoFileClip(video_path)
    audio = video.audio
    audio_path = video_path.replace(".mp4", ".wav")
    audio.write_audiofile(audio_path)
    return audio_path, video


# Step 2: Compute the Short-Time Fourier Transform (STFT)
def compute_stft(audio_path):
    y, sr = librosa.load(audio_path)
    stft_data = librosa.stft(y)
    stft_db = librosa.amplitude_to_db(np.abs(stft_data))
    return stft_db, sr


# Step 3: Detect ball contact by analyzing low-frequency volume spikes
def detect_ball_contact(stft_db, sr, low_freq_range=(64, 1500)):
    freqs = librosa.fft_frequencies(sr=sr)
    low_freq_indices = np.where((freqs >= low_freq_range[0]) & (freqs <= low_freq_range[1]))[0]
    low_freq_magnitudes = np.sum(stft_db[low_freq_indices, :], axis=0)
    low_freq_magnitudes_dif = low_freq_magnitudes[1:-2] - low_freq_magnitudes[0:-3]
    offset = 4
    ball_contact_stft_frame = np.argmax(low_freq_magnitudes_dif) + offset
    ball_contact_time = librosa.frames_to_time(ball_contact_stft_frame, sr=sr)
    return ball_contact_time


# Step 4: Convert audio time to video frame
def time_to_frame(ball_contact_time, video_fps):
    ball_contact_frame = int(ball_contact_time * video_fps)
    return ball_contact_frame

# Function for testing: visualize the contact frame when each contact frame is extracted from audio
def visualize_contact_frame(video_path, contact_frame):
    video = VideoFileClip(video_path)
    frame_rate = video.fps
    time_in_seconds = contact_frame / frame_rate
    frame_image = video.get_frame(time_in_seconds)
    plt.imshow(frame_image)
    plt.axis('off')
    plt.title(f'Contact Frame {contact_frame}')
    plt.show()

# Main function to process multiple kick videos and save contact frames
def process_kick_videos(num_kicks, batch_number):
    contact_frames = []  # Array to store ball contact frames

    # Define the output directory for this batch
    output_dir = f"..\\output\\contact_frames_{batch_number}"
    os.makedirs(output_dir, exist_ok=True)

    for i in range(1, num_kicks + 1):
        video_path = f"../dataset/Session 1/Kick {i}.mp4"
        print(f"Processing {video_path}...")

        # Step 1: Extract audio from video
        audio_path, video = extract_audio_from_video(video_path)

        # Step 2: Perform STFT on extracted audio
        stft_db, sr = compute_stft(audio_path)

        # Step 3: Detect the ball contact time based on low-frequency volume spike
        ball_contact_time = detect_ball_contact(stft_db, sr)

        # Step 4: Convert ball contact time to video frame
        ball_contact_frame = time_to_frame(ball_contact_time, video.fps)
        print(f"Ball contact detected in {video_path} at frame {ball_contact_frame}.")

        # Save the frame number in the array
        contact_frames.append(ball_contact_frame)

        # Save the contact frame as a PNG file in the batch-specific directory
        output_image_path = os.path.join(output_dir, f"contact_frame_{i}.png")
        # Assume 5 frames ago the ball hasn't been kicked.
        save_frame_as_image(video, ball_contact_frame - 5, output_image_path)
        print(f"Contact frame saved as {output_image_path}.")

    # Save the contact frames as a NumPy array
    contact_frames_array = np.array(contact_frames)
    np.save(os.path.join(output_dir, "contact_frames.npy"), contact_frames_array)
    print(f"Contact frames saved to '{output_dir}\\contact_frames.npy'.")


# Example usage
if __name__ == "__main__":
    num_kicks = 20  # Adjust this to the number of kick videos you have
    batch_number = 1  # Set your batch number here
    process_kick_videos(num_kicks, batch_number)
