import os
import json
import numpy as np
import cv2

# Directories for keypoints
side_keypoints_dir = "/output/Multi-View/Session 1/side view/pose_side_views"
back_keypoints_dir = "/output/Multi-View/Session 1/back view/pose_back_views"

# Define connections for COCO keypoints
COCO_SKELETON = [
    (0, 1), (1, 2), (2, 3), (3, 4),       # Nose to left arm
    (0, 5), (5, 6), (6, 7), (7, 8),       # Nose to right arm
    (5, 11), (11, 12), (12, 13),          # Right hip to right leg
    (6, 14), (14, 15), (15, 16),          # Left hip to left leg
    (5, 6), (11, 14),                     # Torso connections
]

# Function to load keypoints from JSON
def load_keypoints(keypoints_path):
    with open(keypoints_path, "r") as f:
        keypoints_list = json.load(f)
    if len(keypoints_list) > 0:
        kp = np.array(keypoints_list[0]["keypoints"])  # x, y positions
        confidence = np.array(keypoints_list[0]["confidence"])  # Confidence scores
        keypoints = np.hstack((kp, confidence.reshape(-1, 1)))  # Combine x, y, and confidence
        return keypoints
    return None

# Function to overlay keypoints and skeleton on an image
def overlay_keypoints(image, keypoints, skeleton):
    for x, y, c in keypoints:
        if c > 0.1:  # Confidence threshold
            cv2.circle(image, (int(x), int(y)), 5, (0, 255, 0), -1)  # Draw keypoint
    for connection in skeleton:
        pt1, pt2 = connection
        if keypoints[pt1][2] > 0.1 and keypoints[pt2][2] > 0.1:  # Check confidence
            x1, y1 = keypoints[pt1][:2]
            x2, y2 = keypoints[pt2][:2]
            cv2.line(image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)  # Draw skeleton line
    return image

# Function to extract a specific frame from a video
def get_frame(video_path, frame_num):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)  # Set the frame to read
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise ValueError(f"Could not read frame {frame_num} from {video_path}")
    return frame

# Visualize pose for ball contact frames
def visualize_contact_frames(video_num, side_contact_frame, back_contact_frame):
    # Paths for video files
    side_video_path = f"/Users/nolanjetter/Documents/GitHub/Soccer ML Project Main/dataset/Multi-View/Session 1/side view/Kick {video_num}.mp4"
    back_video_path = f"/Users/nolanjetter/Documents/GitHub/Soccer ML Project Main/dataset/Multi-View/Session 1/back view/Kick {video_num}.mp4"

    # Paths for keypoints
    side_keypoints_path = f"/Users/nolanjetter/Documents/GitHub/Soccer ML Project Main/output/Multi-View/Session 1/side view/pose_side_views"
    back_keypoints_path = f"/Users/nolanjetter/Documents/GitHub/Soccer ML Project Main/output/Multi-View/Session 1/back view/pose_back_views"

    # Starting frames
    side_frame_idx = side_contact_frame[video_num - 1]
    back_frame_idx = back_contact_frame[video_num - 1]

    # Open video files
    side_cap = cv2.VideoCapture(side_video_path)
    back_cap = cv2.VideoCapture(back_video_path)

    # Set to start at contact frames
    side_cap.set(cv2.CAP_PROP_POS_FRAMES, side_frame_idx)
    back_cap.set(cv2.CAP_PROP_POS_FRAMES, back_frame_idx)

    while True:
        # Read side and back frames
        ret1, side_frame = side_cap.read()
        ret2, back_frame = back_cap.read()

        # Break if either video ends
        if not ret1 or not ret2:
            print("End of video reached.")
            break

        # Load keypoints for current frames
        side_keypoints_path_frame = f"{side_keypoints_path}/keypoints_video{video_num}_frame{side_frame_idx}.json"
        back_keypoints_path_frame = f"{back_keypoints_path}/keypoints_video{video_num}_frame{back_frame_idx}.json"

        side_keypoints = load_keypoints(side_keypoints_path_frame)
        back_keypoints = load_keypoints(back_keypoints_path_frame)

        if side_keypoints is None or back_keypoints is None:
            print(f"Keypoints missing for video {video_num}, frame {side_frame_idx} or {back_frame_idx}")
            break

        # Overlay keypoints on frames
        side_frame_with_pose = overlay_keypoints(side_frame, side_keypoints, COCO_SKELETON)
        back_frame_with_pose = overlay_keypoints(back_frame, back_keypoints, COCO_SKELETON)

        # Combine frames side by side
        combined_frame = np.hstack((side_frame_with_pose, back_frame_with_pose))

        # Display the combined frame
        cv2.imshow(f"Video {video_num}: Side and Back Views", combined_frame)

        # Wait for key press
        key = cv2.waitKey(0)

        # Exit on 'q'
        if key == ord('q'):
            break

        # Increment frame indices
        side_frame_idx += 1
        back_frame_idx += 1

    # Release video captures and close windows
    side_cap.release()
    back_cap.release()
    cv2.destroyAllWindows()


# Example usage
video_number = 3
side_contact_dir = "/Users/nolanjetter/Documents/GitHub/Soccer ML Project Main/output/Multi-View/Session 1/side view/contact frames.npy"
back_contact_dir = "/Users/nolanjetter/Documents/GitHub/Soccer ML Project Main/output/Multi-View/Session 1/back view/contact frames.npy"
side_contact_frame = np.load(side_contact_dir, allow_pickle=True)
back_contact_frame = np.load(back_contact_dir, allow_pickle=True)

visualize_contact_frames(video_number, side_contact_frame, back_contact_frame)