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

    # - the function takes as input the contact frame, ball location, and the pose_keypoints.

    # - move the keypoints relative to the soccer ball: the soccer ball should be located at (0,0) and everything should
    #   be translated relative to this reference.
    keypoints_relative_to_ball = pose_keypoints_array.copy()
    keypoints_relative_to_ball[:, :, 0] -= ball_location[0]  # Adjust x-coordinates
    keypoints_relative_to_ball[:, :, 1] -= ball_location[1]  # Adjust y-coordinates

    # - find the mean of the position of the left joints and right joints for all of the frames up until ball contact.
    #   The mean should have an x+y component and we should have a mean for the left foot and right foot.
    left_foot_positions = keypoints_relative_to_ball[:, left_foot_joints, :2]  # Exclude confidence values
    right_foot_positions = keypoints_relative_to_ball[:, right_foot_joints, :2]

    left_foot_mean = np.mean(left_foot_positions, axis=(0, 1))  # Mean x and y for left foot
    print(left_foot_mean.shape)
    right_foot_mean = np.mean(right_foot_positions, axis=(0, 1))  # Mean x and y for right foot

    # - center the pose_keypoints relative to the mean, and then calculate the covariance for the left and right foot.
    left_foot_centered = left_foot_positions - left_foot_mean  # Center left foot positions
    right_foot_centered = right_foot_positions - right_foot_mean  # Center right foot positions

    left_foot_covariance = np.cov(left_foot_centered.reshape(-1, 2), rowvar=False)
    right_foot_covariance = np.cov(right_foot_centered.reshape(-1, 2), rowvar=False)

    # - the foot with the lower covariance will be the plant foot. return either "left" or "right" depending on which
    #   has the smaller covariance
    left_det = np.linalg.det(left_foot_covariance)
    right_det = np.linalg.det(right_foot_covariance)

    return "left" if left_det < right_det else "right"


if __name__ == "__main__":
    kick_number = 9
    contact_frame = load_contact_frames()[kick_number - 1]
    print(contact_frame)
    pose_keypoints_array = []
    for i in range(contact_frame):
        json_file = os.path.join(os.path.dirname(__file__), f'../output/pose_estimation_results_1/Kick_{kick_number}_0000000000{str(i).zfill(2)}_keypoints.json')
        pose_keypoints = load_keypoints_from_json(json_file)
        if pose_keypoints is not None:
            pose_keypoints_array.append(pose_keypoints)
    pose_keypoints_array = np.array(pose_keypoints_array)
    print(pose_keypoints_array.shape)
    ball_location = [0,0]
    find_plant_foot(ball_location, pose_keypoints_array)
