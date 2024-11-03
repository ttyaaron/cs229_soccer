import numpy as np
import librosa
from moviepy.editor import VideoFileClip
import matplotlib.pyplot as plt

def save_frame_as_image(video, frame_number, output_path):
    # Set the video to the specified frame
    video.reader.seek(frame_number)
    frame = video.reader.read_frame()
    # Save the frame as an image
    cv2.imwrite(output_path, frame)

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
    # Get frequency bins for the STFT
    freqs = librosa.fft_frequencies(sr=sr)

    # Find indices corresponding to the low-frequency range (64-1500 Hz)
    low_freq_indices = np.where((freqs >= low_freq_range[0]) & (freqs <= low_freq_range[1]))[0]

    # Sum the magnitude of the selected low-frequency range across all time frames
    low_freq_magnitudes = np.sum(stft_db[low_freq_indices, :], axis=0)

    # Find where the largest change in magnitude happens to track the shot
    low_freq_magnitudes_dif = low_freq_magnitudes[1:-2] - low_freq_magnitudes[0:-3]

    # Manually tuned offset of frames
    offset = 4

    # Detect the time (in STFT frames) where the maximum low-frequency magnitude occurs
    ball_contact_stft_frame = np.argmax(low_freq_magnitudes_dif) + offset

    # Convert the STFT frame to time (in seconds)
    ball_contact_time = librosa.frames_to_time(ball_contact_stft_frame, sr=sr)

    return ball_contact_time


# Step 4: Convert audio time to video frame
def time_to_frame(ball_contact_time, video_fps):
    ball_contact_frame = int(ball_contact_time * video_fps)
    return ball_contact_frame

# Function for testing: visualize the contact frame when each contact frame is extracted from audio
def visualize_contact_frame(video_path, contact_frame):
    """
    Visualize the specified contact frame from the video using MoviePy.

    Args:
        video_path (str): Path to the video file.
        contact_frame (int): The index of the contact frame to visualize.
    """
    # Load the video file
    video = VideoFileClip(video_path)

    # Calculate the time in seconds corresponding to the contact frame
    frame_rate = video.fps
    time_in_seconds = contact_frame / frame_rate

    # Get the specific frame as an image
    frame_image = video.get_frame(time_in_seconds)

    # Display the frame using Matplotlib
    plt.imshow(frame_image)
    plt.axis('off')  # Turn off axis labels
    plt.title(f'Contact Frame {contact_frame}')

    plt.show()

# Main function to process multiple kick videos and save contact frames
def process_kick_videos(num_kicks):
    contact_frames = []  # Array to store ball contact frames

    for i in range(1, num_kicks + 1):
        video_path = "../dataset/Session 1/Kick " + str(i) + ".mp4"
        # video_path = "/Users/nolanjetter/Desktop/Soccer ML videos/Raw Kicks/Session 1/kick " + str(i) + ".mp4"
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

        # Visualize the result for TESTING
        visualize_contact_frame(video_path, ball_contact_frame)

        # Save the frame number in the array
        contact_frames.append(ball_contact_frame)

    # Save the contact frames as a NumPy array
    contact_frames_array = np.array(contact_frames)
    np.save("contact_frames.npy", contact_frames_array)
    print(f"Contact frames saved to 'contact_frames.npy'.")


# Example usage
if __name__ == "__main__":
    num_kicks = 20  # Adjust this to the number of kick videos you have
    process_kick_videos(num_kicks)