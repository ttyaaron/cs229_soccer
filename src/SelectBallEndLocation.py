import cv2
import numpy as np
import os


def process_videos(session_number, output_directory, base_path, batch_name="Session 1"):
    """
    Process videos in a given session, allowing the user to mark locations with clicks
    and navigate through frames. Clicking once on a video ends the processing for that kick.

    Args:
        session_number (int): The session number to process.
        output_directory (str): The directory to save the output NumPy arrays.
        base_path (str): The base path containing the video files.
        batch_name (str): Name of the batch for output identification.
    """
    # Ensure the output directory exists
    os.makedirs(output_directory, exist_ok=True)

    current_kick = 1

    def mouse_callback(event, x, y, flags, param):
        """Mouse callback to capture click location and frame number."""
        nonlocal click_data, current_frame, end_processing
        if event == cv2.EVENT_LBUTTONDOWN:
            print(f"Clicked at: ({x}, {y}), Frame: {current_frame}")
            click_data.append((current_frame, x, y))
            end_processing = True  # Signal to skip the rest of the kick

    while True:
        # Construct video path
        video_path = os.path.join(base_path, f"Session {session_number}", f"Kick {current_kick}.mp4")
        if not os.path.exists(video_path):
            print(f"No more videos found in session {session_number}. Processing complete.")
            break

        print(f"Processing video: {video_path}")
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print(f"Failed to open video: {video_path}. Skipping to next kick.")
            current_kick += 1
            continue

        cv2.namedWindow("Video")
        click_data = []  # Reset for the current kick
        end_processing = False  # Reset the flag
        current_frame = 0
        cv2.setMouseCallback("Video", mouse_callback)

        while True:
            ret, frame = cap.read()
            if not ret:
                print("End of video reached.")
                break

            cv2.imshow("Video", frame)

            # Handle key inputs
            key = cv2.waitKey(0) & 0xFF
            if end_processing:
                # Break loop if a click was registered
                print("Click registered, moving to next video.")
                break
            if key == ord('m'):  # Skip 5 frames forward
                current_frame += 5
                cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
            elif key == ord('n'):  # Skip 1 frame forward
                current_frame += 1
                cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)

        cap.release()
        cv2.destroyAllWindows()

        # Save click data for the current kick (if any)
        if click_data:
            output_file = os.path.join(output_directory, f"{batch_name}_Session{session_number}_Kick{current_kick}_net_hits.npy")
            np.save(output_file, np.array(click_data, dtype=np.int32))
            print(f"Saved click data to: {output_file}")

        current_kick += 1


if __name__ == "__main__":
    # Define paths
    session_number = 3
    base_path = "/Users/nolanjetter/Documents/GitHub/Soccer ML Project Main/dataset"
    output_directory = f"/Users/nolanjetter/Documents/GitHub/Soccer ML Project Main/output/Batch {session_number}/Net Hits"

    # Process videos
    process_videos(session_number, output_directory, base_path)