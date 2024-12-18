import os
import json
import math
import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

# Directories for keypoints
side_keypoints_dir = "/output/Multi-View/Session 1/side view/pose_side_views"
back_keypoints_dir = "/output/Multi-View/Session 1/back view/pose_back_views"

# Define connections for COCO keypoints
COCO_SKELETON = [
    (5, 11), (11, 12), (7, 5), (8, 6), (3, 6), (3, 5),
    (12, 14), (13, 15), (14, 16), (3, 1),
    (5, 6), (9, 7), (8, 10), (13, 11), (12, 6)
]


def generate_angles_and_lengths(skeleton, frames_3d, proxy_lengths):
    """
    Calculate 3D angles for each connection in the skeleton.

    Parameters:
        skeleton (list of tuple): The skeletal connections defined as pairs of indices.
        frames_3d (np.ndarray): The 3D coordinates of all points for a specific frame,
                                shape (num_points, 4) with [x, y, z, confidence].
        proxy_lengths (np.ndarray): The lengths of each connection in the skeleton.

    Returns:
        dict: A dictionary mapping each connection to its 3D angles (azimuth, elevation).
    """
    angles = {}
    for i, (pt1, pt2) in enumerate(skeleton):
        if frames_3d[pt1, 3] > 0 and frames_3d[pt2, 3] > 0:  # Check confidence
            vec = frames_3d[pt2, :3] - frames_3d[pt1, :3]
            length = np.linalg.norm(vec)

            if length == 0:  # Handle zero-length vectors
                continue

            azimuth = np.arctan2(vec[1], vec[0])  # Angle in the x-y plane
            elevation = np.arcsin(vec[2] / length)  # Angle above x-y plane

            angles[(pt1, pt2)] = (azimuth, elevation)
    return angles

def extend_point(base_point, length, azimuth, elevation):
    """
    Calculate the new point in 3D space given a starting point, length, and angles.

    Parameters:
        base_point (np.ndarray): The starting 3D point as [x, y, z].
        length (float): The length of the limb to extend.
        azimuth (float): Angle in the x-y plane.
        elevation (float): Angle above the x-y plane.

    Returns:
        np.ndarray: The calculated 3D coordinates of the extended point.
    """
    x = base_point[0] + length * np.cos(elevation) * np.cos(azimuth)
    y = base_point[1] + length * np.cos(elevation) * np.sin(azimuth)
    z = base_point[2] + length * np.sin(elevation)
    return np.array([x, y, z])

def resize_pose_3d(starting_points, skeleton, angles, proxy_lengths):
    """
    Resize the 3D pose based on angles and lengths, starting from a known base point.

    Parameters:
        starting_points (np.ndarray): Initial 3D positions of the keypoints, shape (num_points, 3).
        skeleton (list of tuple): The skeletal connections defined as pairs of indices.
        angles (dict): Dictionary of angles for each connection.
        proxy_lengths (np.ndarray): Array of lengths for each connection in the skeleton.

    Returns:
        np.ndarray: The resized 3D positions of the keypoints.
    """
    resized_points = np.zeros(starting_points.shape)

    # Define the order of connections for resizing
    connection_order = [
        (11, 13), (13, 15),  # Right leg
        (11, 12), (12, 14), (14, 16),  # Left leg
        (12, 6), (6, 8), (6, 5)
    ]

    for pt1, pt2 in connection_order:
        if (pt1, pt2) in angles or (pt2, pt1) in angles:
            azimuth, elevation = angles.get((pt1, pt2), angles.get((pt2, pt1)))
            length_idx = skeleton.index((pt1, pt2)) if (pt1, pt2) in skeleton else skeleton.index((pt2, pt1))
            length = proxy_lengths[length_idx]

            resized_points[pt2] = extend_point(resized_points[pt1], length, azimuth, elevation)

    return resized_points


def create_symmetry(skeleton, lengths):
    """
    Creates a symmetrical version of the length array by handling outliers and averaging corresponding pairs.

    Parameters:
        skeleton (list of tuple): The skeletal connections defined as pairs of indices (e.g., COCO_SKELETON).
        lengths (np.ndarray): An array of maximum lengths for each connection in the skeleton.

    Returns:
        np.ndarray: A symmetrical version of the length array.
    """
    # Define pairs of corresponding limbs
    corresponding_pairs = [
        ((10, 8), (9, 7)),
        ((8, 6), (5, 7)),
        ((6, 12), (5, 11)),
        ((12, 14), (11, 13)),
        ((14, 16), (13, 15)),
        ((3, 5), (3, 6))
    ]

    # Copy lengths to modify
    symmetric_lengths = lengths.copy()

    for pair1, pair2 in corresponding_pairs:
        # Get indices for both pairs
        idx1 = skeleton.index(pair1) if pair1 in skeleton else skeleton.index(pair1[::-1])
        idx2 = skeleton.index(pair2) if pair2 in skeleton else skeleton.index(pair2[::-1])

        # Check for outliers
        if abs(lengths[idx1] - lengths[idx2]) > 1.5 * min(lengths[idx1], lengths[idx2]):
            # Replace outlier with the length of its counterpart
            if lengths[idx1] > lengths[idx2]:
                symmetric_lengths[idx1] = lengths[idx2]
            else:
                symmetric_lengths[idx2] = lengths[idx1]

        # Take the average to ensure symmetry
        avg_length = (symmetric_lengths[idx1] + symmetric_lengths[idx2]) / 2
        symmetric_lengths[idx1] = avg_length
        symmetric_lengths[idx2] = avg_length

    return symmetric_lengths


def visualize_3d_pose(points, skeleton):
    """
    Visualizes the 3D pose given keypoints and their skeletal connections.

    Parameters:
        points (np.ndarray): 3D positions of the keypoints, shape (num_keypoints, 3).
        skeleton (list of tuple): The skeletal connections defined as pairs of indices.

    Returns:
        None: Displays the 3D visualization.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot keypoints
    for i, (x, y, z) in enumerate(points):
        ax.scatter(x, y, z, color='red', s=30)
        ax.text(x, y, z, str(i), color='black', fontsize=8)

    # Plot skeletal connections
    for pt1, pt2 in skeleton:
        if pt1 < len(points) and pt2 < len(points):  # Ensure points exist
            x_vals = [points[pt1][0], points[pt2][0]]
            y_vals = [points[pt1][1], points[pt2][1]]
            z_vals = [points[pt1][2], points[pt2][2]]
            ax.plot(x_vals, y_vals, z_vals, color='blue')

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("3D Pose Visualization")
    plt.show()


def visualize_front_view(skeleton, proxy_lengths):
    """
    Visualizes the body shape from a front view by reconstructing the pose based on proxy lengths.

    Parameters:
        skeleton (list of tuple): The skeletal connections defined as pairs of indices (e.g., COCO_SKELETON).
        proxy_lengths (np.ndarray): An array of maximum lengths for each connection in the skeleton.

    Returns:
        None: Displays a pyplot visualization of the front view.
    """
    # Initialize keypoint positions
    keypoint_positions = {}

    # Start with point 11 at (0, 0)
    keypoint_positions[11] = (0, 0)

    # Place point 12 to the left by the proxy length
    idx_12_11 = skeleton.index((12, 11)) if (12, 11) in skeleton else skeleton.index((11, 12))
    keypoint_positions[12] = (-proxy_lengths[idx_12_11], 0)

    # Place point 13 directly down from point 11
    idx_11_13 = skeleton.index((11, 13)) if (11, 13) in skeleton else skeleton.index((13, 11))
    keypoint_positions[13] = (0, -proxy_lengths[idx_11_13])

    # Place point 15 directly down from point 13
    idx_13_15 = skeleton.index((13, 15)) if (13, 15) in skeleton else skeleton.index((15, 13))
    keypoint_positions[15] = (0, keypoint_positions[13][1] - proxy_lengths[idx_13_15])

    # Place point 14 directly down from point 12
    idx_12_14 = skeleton.index((12, 14)) if (12, 14) in skeleton else skeleton.index((14, 12))
    keypoint_positions[14] = (keypoint_positions[12][0], keypoint_positions[12][1] - proxy_lengths[idx_12_14])

    # Place point 16 directly down from point 14
    idx_14_16 = skeleton.index((14, 16)) if (14, 16) in skeleton else skeleton.index((16, 14))
    keypoint_positions[16] = (keypoint_positions[14][0], keypoint_positions[14][1] - proxy_lengths[idx_14_16])

    # Place point 5 directly up from point 11
    idx_11_5 = skeleton.index((11, 5)) if (11, 5) in skeleton else skeleton.index((5, 11))
    keypoint_positions[5] = (0, keypoint_positions[11][1] + proxy_lengths[idx_11_5])

    # Place point 7 to the right of point 5
    idx_5_7 = skeleton.index((5, 7)) if (5, 7) in skeleton else skeleton.index((7, 5))
    keypoint_positions[7] = (keypoint_positions[5][0] + proxy_lengths[idx_5_7], keypoint_positions[5][1])

    # Place point 9 to the right of point 7
    idx_7_9 = skeleton.index((7, 9)) if (7, 9) in skeleton else skeleton.index((9, 7))
    keypoint_positions[9] = (keypoint_positions[7][0] + proxy_lengths[idx_7_9], keypoint_positions[7][1])

    # Place point 6 directly up from point 12
    idx_12_6 = skeleton.index((12, 6)) if (12, 6) in skeleton else skeleton.index((6, 12))
    keypoint_positions[6] = (keypoint_positions[12][0], keypoint_positions[12][1] + proxy_lengths[idx_12_6])

    # Place point 8 to the left of point 6
    idx_6_8 = skeleton.index((6, 8)) if (6, 8) in skeleton else skeleton.index((8, 6))
    keypoint_positions[8] = (keypoint_positions[6][0] - proxy_lengths[idx_6_8], keypoint_positions[6][1])

    # Place point 10 to the left of point 8
    idx_8_10 = skeleton.index((8, 10)) if (8, 10) in skeleton else skeleton.index((10, 8))
    keypoint_positions[10] = (keypoint_positions[8][0] - proxy_lengths[idx_8_10], keypoint_positions[8][1])

    # Place point 3 such that it forms an isosceles triangle with points 5 and 6
    idx_3_5 = skeleton.index((5, 3)) if (5, 3) in skeleton else skeleton.index((3, 5))
    x_distance = keypoint_positions[5][0] - keypoint_positions[6][0]
    base_length = x_distance / 2
    height = math.sqrt(proxy_lengths[idx_3_5]**2 - base_length**2)
    keypoint_positions[3] = (keypoint_positions[5][0] - base_length, keypoint_positions[5][1] + height)

    # Place point 0 directly above point 3
    idx_1_3 = skeleton.index((1, 3)) if (1, 3) in skeleton else skeleton.index((3, 1))
    keypoint_positions[1] = (
        keypoint_positions[3][0],  # Same x-coordinate as point 3
        keypoint_positions[3][1] + proxy_lengths[idx_1_3]  # Extend upward
    )

    # Plot the points and connections
    fig, ax = plt.subplots()
    for pt1, pt2 in skeleton:
        if pt1 in keypoint_positions and pt2 in keypoint_positions:
            x1, y1 = keypoint_positions[pt1]
            x2, y2 = keypoint_positions[pt2]
            ax.plot([x1, x2], [y1, y2], color='blue')  # Draw connection

    for keypoint, (x, y) in keypoint_positions.items():
        ax.scatter(x, y, color='red')  # Draw keypoint
        ax.text(x, y, str(keypoint), color='black', fontsize=8)  # Label keypoint

    ax.set_aspect('equal')
    ax.set_title("Front View of Body Shape")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)
    plt.show()


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
    for idx, (x, y, c) in enumerate(keypoints):
        if c > 0.1:  # Confidence threshold
            cv2.circle(image, (int(x), int(y)), 5, (0, 255, 0), -1)  # Draw keypoint
            cv2.putText(image, str(idx), (int(x) + 5, int(y) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    for connection in skeleton:
        pt1, pt2 = connection
        if keypoints[pt1][2] > 0.1 and keypoints[pt2][2] > 0.1:  # Check confidence
            x1, y1 = keypoints[pt1][:2]
            x2, y2 = keypoints[pt2][:2]
            cv2.line(image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)  # Draw skeleton line
    return image


def compute_proxy_body_shape_2d(side_frames, back_frames, skeleton):
    """
    Computes the proxy for the "shape" of the body by calculating the maximum observed
    Euclidean distance for each skeletal connection across all frames from both side
    and back views.

    Parameters:
        side_frames (list of np.ndarray): A list of 2D pose keypoints from the side view,
            where each frame is an array of shape (num_keypoints, 3) with columns [x, y, confidence].
        back_frames (list of np.ndarray): A list of 2D pose keypoints from the back view,
            where each frame is an array of shape (num_keypoints, 3) with columns [x, y, confidence].
        skeleton (list of tuple): The skeletal connections defined as pairs of indices
            into the keypoints array (e.g., COCO_SKELETON).

    Returns:
        np.ndarray: An array of maximum lengths for each connection in the skeleton.
    """
    num_connections = len(skeleton)
    max_lengths = np.zeros(num_connections)

    # Iterate through each frame
    for side_frame, back_frame in zip(side_frames, back_frames):
        for i, (pt1, pt2) in enumerate(skeleton):
            # Calculate distance for the side view
            if side_frame[pt1, 2] > 0 and side_frame[pt2, 2] > 0:  # Ensure confidence > 0
                side_distance = np.linalg.norm(side_frame[pt1, :2] - side_frame[pt2, :2])
            else:
                side_distance = 0

            # Calculate distance for the back view
            if back_frame[pt1, 2] > 0 and back_frame[pt2, 2] > 0:  # Ensure confidence > 0
                back_distance = np.linalg.norm(back_frame[pt1, :2] - back_frame[pt2, :2])
            else:
                back_distance = 0

            # Update the maximum length for this connection
            max_lengths[i] = max(max_lengths[i], side_distance, back_distance)

    return max_lengths


# Function to trim an image to the middle 4 rows and 4 columns
def trim_image(image):
    h, w, _ = image.shape
    row_start = h // 6
    row_end = h - h // 6
    col_start = w // 6
    col_end = w - w // 6
    return image[row_start:row_end, col_start:col_end]

# Function to average points 0, 1, 2, and 4 and replace them in keypoints
def average_points(keypoints):
    avg_x = np.mean(keypoints[[0, 1, 2, 4], 0])
    avg_y = np.mean(keypoints[[0, 1, 2, 4], 1])
    avg_conf = np.mean(keypoints[[0, 1, 2, 4], 2])
    keypoints[[0, 1, 2, 4], 0] = avg_x
    keypoints[[0, 1, 2, 4], 1] = avg_y
    keypoints[[0, 1, 2, 4], 2] = avg_conf
    return keypoints

# Function to create a 3D array from two views
def create_3d_model(side_keypoints, back_keypoints, confidence_threshold):
    num_points = side_keypoints.shape[0]
    model_3d = np.zeros((num_points, 3))
    confidence_3d = np.zeros(num_points)

    for i in range(num_points):
        if side_keypoints[i, 2] > confidence_threshold and back_keypoints[i, 2] > confidence_threshold:
            model_3d[i, 0] = back_keypoints[i, 0]  # x from back view
            model_3d[i, 1] = back_keypoints[i, 1]  # y from back view
            model_3d[i, 2] = side_keypoints[i, 0]  # z from side view
            confidence_3d[i] = 100  # Set high confidence
        else:
            confidence_3d[i] = 0  # Low confidence if one or both points are missing

    return np.hstack((model_3d, confidence_3d.reshape(-1, 1)))


# Function to plot the 3D model as an animation
def animate_3d_model_with_views(frames, side_frames, back_frames, skeleton, swap_yz=True):
    fig = plt.figure(figsize=(12, 6))  # Larger figure for three plots
    ax_3d = fig.add_subplot(133, projection='3d')  # 3D plot on the right
    ax_side = fig.add_subplot(131)  # Side view frame on the left
    ax_back = fig.add_subplot(132)  # Back view frame in the middle

    current_frame = [0]  # Track frame index

    def update(frame_idx):
        # Clear axes
        ax_3d.clear()
        ax_side.clear()
        ax_back.clear()

        # Update 3D model
        model_3d = frames[frame_idx]
        for idx, (x, y, z, c) in enumerate(model_3d):
            if swap_yz:
                y, z = z, y
            if c > 0:  # Plot only confident points
                ax_3d.scatter(-x, y, -z, color='b', s=20)
                ax_3d.text(-x, y, -z, str(idx), color='r', fontsize=8)
        for connection in skeleton:
            pt1, pt2 = connection
            if model_3d[pt1, 3] > 0 and model_3d[pt2, 3] > 0:
                x1, y1, z1 = model_3d[pt1, :3]
                x2, y2, z2 = model_3d[pt2, :3]
                if swap_yz:
                    y1, z1 = z1, y1
                    y2, z2 = z2, y2
                ax_3d.plot([-x1, -x2], [y1, y2], [-z1, -z2], color='g')
        ax_3d.set_title(f"3D Model: Frame {frame_idx + 1}")
        ax_3d.set_xlabel("X")
        ax_3d.set_ylabel("Y")
        ax_3d.set_zlabel("Z")

        # Update side and back views
        ax_side.imshow(cv2.cvtColor(side_frames[frame_idx], cv2.COLOR_BGR2RGB))
        ax_side.axis('off')
        ax_side.set_title("Side View")

        ax_back.imshow(cv2.cvtColor(back_frames[frame_idx], cv2.COLOR_BGR2RGB))
        ax_back.axis('off')
        ax_back.set_title("Back View")

    def on_key(event):
        if event.key == 'right':
            current_frame[0] = (current_frame[0] + 1) % len(frames)  # Next frame
        elif event.key == 'left':
            current_frame[0] = (current_frame[0] - 1) % len(frames)  # Previous frame
        elif event.key == 'q':
            plt.close(fig)
            return
        update(current_frame[0])
        fig.canvas.draw_idle()

    # Initialize first frame
    update(current_frame[0])
    fig.canvas.mpl_connect('key_press_event', on_key)
    plt.tight_layout()
    plt.show()

# Function to preprocess frames and keypoints
def preprocess_frames(video_num, side_frame_idx, back_frame_idx, side_keypoints_path, back_keypoints_path):
    side_keypoints_path_frame = f"{side_keypoints_path}/keypoints_video{video_num}_frame{side_frame_idx}.json"
    back_keypoints_path_frame = f"{back_keypoints_path}/keypoints_video{video_num}_frame{back_frame_idx}.json"

    side_keypoints = load_keypoints(side_keypoints_path_frame)
    back_keypoints = load_keypoints(back_keypoints_path_frame)

    if side_keypoints is None or back_keypoints is None:
        return None, None

    side_keypoints = average_points(side_keypoints)
    back_keypoints = average_points(back_keypoints)

    return side_keypoints, back_keypoints

# Visualize pose for ball contact frames
def visualize_contact_frames(video_num, side_frame_idx, back_frame_idx, side_frame, back_frame, side_keypoints, back_keypoints):
    # Overlay keypoints on frames
    side_frame_with_pose = overlay_keypoints(side_frame, side_keypoints, COCO_SKELETON)
    back_frame_with_pose = overlay_keypoints(back_frame, back_keypoints, COCO_SKELETON)

    # Trim frames to middle sections
    side_frame_trimmed = trim_image(side_frame_with_pose)
    back_frame_trimmed = trim_image(back_frame_with_pose)

    # Combine frames side by side
    combined_frame = np.hstack((side_frame_trimmed, back_frame_trimmed))

    # Display the combined frame
    cv2.imshow(f"Video {video_num}: Side and Back Views", combined_frame)

# Example usage
video_number = 2
side_contact_dir = "/Users/nolanjetter/Documents/GitHub/Soccer ML Project Main/output/Multi-View/Session 1/side view/contact frames.npy"
back_contact_dir = "/Users/nolanjetter/Documents/GitHub/Soccer ML Project Main/output/Multi-View/Session 1/back view/contact frames.npy"
side_contact_frame = np.load(side_contact_dir, allow_pickle=True)
back_contact_frame = np.load(back_contact_dir, allow_pickle=True)

side_video_path = f"/Users/nolanjetter/Documents/GitHub/Soccer ML Project Main/dataset/Multi-View/Session 1/side view/Kick {video_number}.mp4"
back_video_path = f"/Users/nolanjetter/Documents/GitHub/Soccer ML Project Main/dataset/Multi-View/Session 1/back view/Kick {video_number}.mp4"

side_keypoints_path = "/Users/nolanjetter/Documents/GitHub/Soccer ML Project Main/output/Multi-View/Session 1/side view/pose_side_views"
back_keypoints_path = "/Users/nolanjetter/Documents/GitHub/Soccer ML Project Main/output/Multi-View/Session 1/back view/pose_back_views"

side_cap = cv2.VideoCapture(side_video_path)
back_cap = cv2.VideoCapture(back_video_path)

side_frame_idx = side_contact_frame[video_number - 1] - 10
back_frame_idx = back_contact_frame[video_number - 1] - 10
all_frames_3d = []

side_frames = []
side_keypoints_master = []
back_frames = []
back_keypoints_master = []

while True:
    side_cap.set(cv2.CAP_PROP_POS_FRAMES, side_frame_idx)
    back_cap.set(cv2.CAP_PROP_POS_FRAMES, back_frame_idx)

    ret1, side_frame = side_cap.read()
    ret2, back_frame = back_cap.read()

    if not ret1 or not ret2:
        print("End of video reached.")
        break

    side_keypoints, back_keypoints = preprocess_frames(video_number, side_frame_idx, back_frame_idx,
                                                       side_keypoints_path, back_keypoints_path)
    side_keypoints_master.append(side_keypoints)
    back_keypoints_master.append(back_keypoints)
    # visualize_contact_frames(video_number, side_frame_idx, back_frame_idx, side_frame, back_frame, side_keypoints, back_keypoints)
    if side_keypoints is None or back_keypoints is None:
        print(f"Keypoints missing for video {video_number}, frame {side_frame_idx} or {back_frame_idx}")
        break

    confidence_threshold = 0.1
    model_3d = create_3d_model(side_keypoints, back_keypoints, confidence_threshold)
    all_frames_3d.append(model_3d)

    # Add frames to the lists
    side_frames.append(trim_image(side_frame))
    back_frames.append(trim_image(back_frame))

    side_frame_idx += 1
    back_frame_idx += 1

proxy_shape = compute_proxy_body_shape_2d(side_keypoints_master, back_keypoints_master, COCO_SKELETON)
symmetrical_lengths = create_symmetry(COCO_SKELETON, proxy_shape)
visualize_front_view(COCO_SKELETON, symmetrical_lengths)
frames_3d = all_frames_3d[0]
print(f"frames 3d: {np.array(frames_3d).shape}")
angles = generate_angles_and_lengths(COCO_SKELETON, frames_3d, proxy_shape)
starting_points = frames_3d[:, :3]  # Initial positions of the keypoints
resized_pose = resize_pose_3d(starting_points, COCO_SKELETON, angles, proxy_shape)
visualize_3d_pose(resized_pose, COCO_SKELETON)
# animate_3d_model_with_views(all_frames_3d, side_frames, back_frames, COCO_SKELETON)

# things to do:
# trying to restructure the 3d pose skeleton such that it is based on the length of the limbs
# the script that I am using for the