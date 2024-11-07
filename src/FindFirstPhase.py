import json
import numpy as np
import os

FILENAME = 'contact_frames.npy'


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


def find_plant_foot(ball_location, pose_keypoints_array):
    left_foot_joints = [11, 22, 23, 24]
    right_foot_joints = [14, 19, 20, 21]

    # Move keypoints relative to the soccer ball so it’s at (0, 0)
    keypoints_relative_to_ball = pose_keypoints_array.copy()
    keypoints_relative_to_ball[:, :, 0] -= ball_location[0]
    keypoints_relative_to_ball[:, :, 1] -= ball_location[1]

    # Gather averaged valid positions (confidence > 0) for left and right foot joints per frame
    left_foot_positions = []
    right_foot_positions = []

    for frame in range(keypoints_relative_to_ball.shape[0]):
        valid_left_x = []
        valid_left_y = []
        valid_right_x = []
        valid_right_y = []

        for joint in left_foot_joints:
            if keypoints_relative_to_ball[frame, joint, 2] > 0:  # Check confidence
                valid_left_x.append(keypoints_relative_to_ball[frame, joint, 0])
                valid_left_y.append(keypoints_relative_to_ball[frame, joint, 1])

        for joint in right_foot_joints:
            if keypoints_relative_to_ball[frame, joint, 2] > 0:  # Check confidence
                valid_right_x.append(keypoints_relative_to_ball[frame, joint, 0])
                valid_right_y.append(keypoints_relative_to_ball[frame, joint, 1])

        # Calculate mean x and y for each foot if there are valid points
        if valid_left_x and valid_left_y:
            avg_left_x = np.mean(valid_left_x)
            avg_left_y = np.mean(valid_left_y)
            left_foot_positions.append([avg_left_x, avg_left_y])

        if valid_right_x and valid_right_y:
            avg_right_x = np.mean(valid_right_x)
            avg_right_y = np.mean(valid_right_y)
            right_foot_positions.append([avg_right_x, avg_right_y])

    # Convert lists to arrays for mean and covariance calculations
    left_foot_positions = np.array(left_foot_positions)
    right_foot_positions = np.array(right_foot_positions)

    # Calculate mean positions
    left_foot_mean = np.mean(left_foot_positions, axis=0)
    right_foot_mean = np.mean(right_foot_positions, axis=0)

    # Center positions around the mean and calculate covariance
    left_foot_centered = left_foot_positions - left_foot_mean
    right_foot_centered = right_foot_positions - right_foot_mean

    left_foot_covariance = np.cov(left_foot_centered, rowvar=False)
    right_foot_covariance = np.cov(right_foot_centered, rowvar=False)

    # Determine which foot has the smaller covariance determinant
    left_det = np.linalg.det(left_foot_covariance)
    print("left det:", left_det)
    right_det = np.linalg.det(right_foot_covariance)
    print("right det:", right_det)

    return ["left", left_foot_positions] if left_det < right_det else ["right", right_foot_positions]


if __name__ == "__main__":
    kick_number = 9
    print("Kick Number:", kick_number)
    contact_frame = load_contact_frames()[kick_number - 1]
    pose_keypoints_array = []

    # Load pose keypoints up to the contact frame
    for i in range(contact_frame):
        json_file = os.path.join(os.path.dirname(__file__),
                                 f'../output/pose_estimation_results_1/Kick_{kick_number}_0000000000{str(i).zfill(2)}_keypoints.json')
        pose_keypoints = load_keypoints_from_json(json_file)
        if pose_keypoints is not None:
            pose_keypoints_array.append(pose_keypoints)

    pose_keypoints_array = np.array(pose_keypoints_array)
    print("Pose Keypoints Shape:", pose_keypoints_array.shape)

    ball_location = [0, 0]
    plant_foot, foot_positions_per_frame = find_plant_foot(ball_location, pose_keypoints_array)

    # need to add in functionality to find the frame at which the foot is planted.

    print("Plant Foot:", plant_foot)
