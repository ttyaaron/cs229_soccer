import json
import os
import numpy as np

# Constants
FILENAME = 'contact_frames.npy'
LEFT_FOOT_INDICES = [11, 22, 23, 24]
RIGHT_FOOT_INDICES = [14, 19, 20, 21]
THRESHOLD = 0.5  # Threshold for detecting stability in x/y coordinates


# Data Loading
def load_contact_frames(filename=FILENAME):
    """Load the array representing frames of ball contact."""
    try:
        contact_frames = np.load(filename)
        print(f"Successfully loaded {filename}. Shape: {contact_frames.shape}")
        return contact_frames
    except FileNotFoundError:
        print(f"Error: {filename} not found.")
        return None


def load_keypoints_from_json(json_file):
    """Load pose keypoints from a JSON file."""
    with open(json_file, 'r') as f:
        data = json.load(f)
        if len(data["people"]) == 0:
            return None
        return np.array(data["people"][0]["pose_keypoints_2d"]).reshape(-1, 3)


# Foot Analysis
def compute_foot_covariance(foot_data):
    """Compute covariance matrix for foot movement based on x and y coordinates."""
    foot_coords_flat = foot_data.reshape(-1, 2)  # Flatten for covariance calculation
    return np.cov(foot_coords_flat, rowvar=False)


def detect_plant_foot(master_array, contact_frame):
    """Detect the plant foot by comparing determinants of covariance matrices up to ball contact."""
    left_foot_data = master_array[:contact_frame, LEFT_FOOT_INDICES, :2]  # X, Y coordinates for left foot
    right_foot_data = master_array[:contact_frame, RIGHT_FOOT_INDICES, :2]  # X, Y coordinates for right foot

    left_foot_cov = compute_foot_covariance(left_foot_data)
    right_foot_cov = compute_foot_covariance(right_foot_data)

    left_det = np.linalg.det(left_foot_cov)
    right_det = np.linalg.det(right_foot_cov)

    plant_foot = 'left' if left_det < right_det else 'right'
    print(f"Plant foot determined: {plant_foot} (Left det: {left_det}, Right det: {right_det})")
    return plant_foot, LEFT_FOOT_INDICES if plant_foot == 'left' else RIGHT_FOOT_INDICES


def detect_plant_frame(master_array, foot_indices, contact_frame, threshold=THRESHOLD):
    """Detect the frame at which the plant foot stops moving and remains stable until ball contact."""
    stable_frame = None
    for i in range(contact_frame):
        foot_coords = master_array[i, foot_indices, :2]

        # Check if all x/y coordinates are within the threshold for stability
        if np.all(np.abs(np.diff(foot_coords, axis=0)) < threshold):
            if stable_frame is None:
                stable_frame = i  # Mark this as the first stable frame
        else:
            stable_frame = None  # Reset if any movement exceeds the threshold

    if stable_frame is not None:
        print(f"Plant foot stable starting at frame {stable_frame}")
    else:
        print("No stable frame detected before contact")

    return stable_frame


# Main Processing
def main_func(kick_number):
    # Load contact frames
    contact_frames = load_contact_frames()
    if contact_frames is None:
        return

    # Use contact_frames[kick_number - 1] to get the correct frame for the specified kick
    contact_frame = int(contact_frames[kick_number - 1])  # Frame number of ball contact
    master_array = []
    for i in range(contact_frame):  # Load frames strictly up to ball contact
        json_file = os.path.join(os.path.dirname(__file__),
                                 f'../output/pose_estimation_results_1/Kick_{kick_number}_0000000000{str(i).zfill(2)}_keypoints.json')
        pose_keypoints = load_keypoints_from_json(json_file)
        if pose_keypoints is not None:
            master_array.append(pose_keypoints)

    master_array = np.array(master_array)

    # Detect the plant foot
    plant_foot, foot_indices = detect_plant_foot(master_array, contact_frame)

    # Detect the frame of planting
    plant_frame = detect_plant_frame(master_array, foot_indices, contact_frame)
    print(f"Detected plant frame: {plant_frame}")


if __name__ == "__main__":
    kick_number = 9  # Example kick number
    main_func(kick_number)