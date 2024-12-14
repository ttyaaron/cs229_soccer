import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.animation import FuncAnimation
import time
import random


# Function to reshape a column into a 2D frame
def reshape_frame(frame_column, height, width):
    """
    Reshape a 1D column vector into a 2D matrix representing a frame.

    Parameters:
    - frame_column: np.ndarray, the column vector of the frame.
    - height: int, the height of the frame.
    - width: int, the width of the frame.

    Returns:
    - reshaped_frame: np.ndarray, the 2D frame.
    """
    return frame_column.reshape((height, width))


# Function to calculate motion using optical flow
def calculate_optical_flow(frame1, frame2, motion_threshold=2.0):
    """
    Calculate motion between two frames using dense optical flow.

    Parameters:
    - frame1: np.ndarray, the first frame.
    - frame2: np.ndarray, the second frame.
    - motion_threshold: float, threshold for significant motion magnitude.

    Returns:
    - motion_mask: np.ndarray, binary mask highlighting motion areas.
    """
    flow = cv2.calcOpticalFlowFarneback(frame1, frame2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    motion_mask = mag > motion_threshold
    return motion_mask.astype(np.uint8)


# Function to detect the moving soccer ball
def detect_ball(motion_mask, min_area=50, max_area=600):
    """
    Detect the moving soccer ball using the motion mask.

    Parameters:
    - motion_mask: np.ndarray, binary mask from optical flow.
    - min_area: int, minimum area of the motion blob to be considered as the ball.

    Returns:
    - ball_positions: list of tuples, detected ball positions as (x, y, w, h).
    """
    contours, _ = cv2.findContours(motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_positions = []
    for cnt in contours:
        if min_area < cv2.contourArea(cnt) < max_area:
            x, y, w, h = cv2.boundingRect(cnt)
            contour_positions.append((x, y, w, h))
    return contour_positions


# Main function to process the single numpy array
def process_sparse_matrix(sparse_matrix, frame_height, frame_width):
    """
    Process the sparse matrix to detect and track the moving soccer ball.

    Parameters:
    - sparse_matrix: np.ndarray, the combined matrix of shape (features, frames).
    - frame_height: int, the height of each frame.
    - frame_width: int, the width of each frame.
    """

    return_arr = []

    num_frames = sparse_matrix.shape[1]
    print(num_frames)
    if num_frames < 2:
        print("Need at least two frames to calculate optical flow.")
        return

    for i in range(num_frames - 1):
        # Extract and reshape consecutive frames
        frame1 = reshape_frame(sparse_matrix[:, i], frame_height, frame_width)
        frame2 = reshape_frame(sparse_matrix[:, i + 1], frame_height, frame_width)

        # Normalize frames to uint8
        frame1 = cv2.normalize(frame1, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        frame2 = cv2.normalize(frame2, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # Calculate motion using optical flow
        motion_mask = calculate_optical_flow(frame1, frame2)

        # Detect the soccer ball in the motion mask
        ball_positions = detect_ball(motion_mask)
        return_arr.append(ball_positions)

        # Visualize the results
        motion_overlay = cv2.cvtColor(frame1, cv2.COLOR_GRAY2BGR)
        for (x, y, w, h) in ball_positions:
            cv2.rectangle(motion_overlay, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Show the motion mask and ball detection
        # cv2.imshow("Motion Mask", motion_mask * 255)
        # cv2.imshow("Ball Detection", motion_overlay)
        #
        # key = cv2.waitKey(0) & 0xFF
        # if key == ord('n'):  # Press 'n' to move to the next frame
        #     continue
        # elif key == 27:  # Press ESC to quit
        #     break

    cv2.destroyAllWindows()
    return return_arr


def open_video(video_path, target_frame=0):
    """Open the video and set to the target frame."""
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
    ret, frame = cap.read()
    if not ret:
        print("Could not retrieve the specified frame.")
        cap.release()
    return cap, frame


def calculate_angle(x1, y1, x2, y2):
    """
    Calculate the angle between two points in degrees.
    """
    return np.degrees(np.arctan2(y2 - y1, x2 - x1))


import matplotlib.pyplot as plt
import random

def plot_all_trajectories(filtered_trajectories, frame_height, frame_width, best_trajectory=None):
    """
    Plot all trajectories in different colors, highlighting the best trajectory.

    Parameters:
    - filtered_trajectories: List of trajectories. Each trajectory is a list of (x, y, area, frame_idx).
    - frame_height: int, the height of the video frames (used to set plot bounds).
    - frame_width: int, the width of the video frames (used to set plot bounds).
    - best_trajectory: List of (x, y, area, frame_idx), the best trajectory to highlight (optional).

    Returns:
    None
    """
    # Generate random colors for each trajectory
    trajectory_colors = [
        (random.random(), random.random(), random.random())
        for _ in range(len(filtered_trajectories))
    ]

    # Create the plot
    plt.figure(figsize=(12, 8))
    plt.title("Trajectories")
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.xlim(0, frame_width)
    plt.ylim(frame_height, 0)  # Invert Y-axis for image-style coordinates

    # Plot each trajectory
    for traj_idx, traj in enumerate(filtered_trajectories):
        color = trajectory_colors[traj_idx]
        x_coords = [point[0] for point in traj]  # Extract x-coordinates
        y_coords = [point[1] for point in traj]  # Extract y-coordinates
        plt.plot(x_coords, y_coords, marker="o", linestyle="-", color=color, linewidth=1.5, markersize=4)

    # Highlight the best trajectory if provided
    if best_trajectory:
        x_coords = [point[0] for point in best_trajectory]
        y_coords = [point[1] for point in best_trajectory]
        plt.plot(x_coords, y_coords, marker="o", linestyle="-", color="red", linewidth=2.5, markersize=6, label="Best Trajectory")

    plt.show()


def find_trajectories(all_contours, starting_ball_location, angle_tolerance=25):
    """
    Find trajectories by associating contours across frames.

    Parameters:
    - all_contours: List of contours for each frame.
    - starting_ball_location: (x, y) tuple for the ball's starting location.
    - angle_tolerance: Maximum angle difference to consider a trajectory.

    Returns:
    - trajectories: List of trajectories. Each trajectory is a list of contours.
    """
    trajectories = []  # List to hold all trajectories

    for frame_idx, contours in enumerate(all_contours):
        if frame_idx == 0:
            # Initialize trajectories from the first frame
            for contour in contours:
                x, y, w, h = contour
                trajectories.append([(x + w / 2, y + h / 2)])  # Use center point of contour
        else:
            # For subsequent frames, match contours to existing trajectories
            new_trajectories = []
            for contour in contours:
                x, y, w, h = contour
                cx, cy = x + w / 2, y + h / 2  # Center of the current contour
                matched = False

                for trajectory in trajectories:
                    if len(trajectory) == 1:
                        # Only one point in the trajectory, compare to the starting point
                        tx, ty = trajectory[-1]
                        angle = calculate_angle(starting_ball_location[0], starting_ball_location[1], tx, ty)
                        new_angle = calculate_angle(tx, ty, cx, cy)
                    else:
                        # Use the last two points to calculate the trajectory slope
                        tx1, ty1 = trajectory[-2]
                        tx2, ty2 = trajectory[-1]
                        angle = calculate_angle(tx1, ty1, tx2, ty2)
                        new_angle = calculate_angle(tx2, ty2, cx, cy)

                    # Check if the angle difference is within tolerance
                    if abs(new_angle - angle) <= angle_tolerance:
                        trajectory.append((cx, cy))
                        matched = True
                        break

                if not matched:
                    # Start a new trajectory if no match is found
                    new_trajectories.append([(cx, cy)])

            # Add the new trajectories to the main list
            trajectories.extend(new_trajectories)

    return trajectories


def find_best_trajectory(trajectories, size_weight=1.0, distance_weight=1.0, length_weight=1.0):
    """
    Find the best trajectory based on a loss function.

    Parameters:
    - trajectories: List of trajectories. Each trajectory is a list of (x, y, area).
    - size_weight: Weight for the size difference component of the loss.
    - distance_weight: Weight for the distance component of the loss.
    - length_weight: Weight for the trajectory length component of the loss.

    Returns:
    - best_trajectory: The trajectory with the lowest loss.
    - losses: A list of losses for all trajectories.
    """
    def calculate_distance(x1, y1, x2, y2):
        return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5

    losses = []
    avg_areas = []

    print(f"trajectories: {trajectories}")

    for trajectory in trajectories:
        if len(trajectory) < 2:
            # Single-point trajectories are penalized heavily
            losses.append(float('inf'))
            continue

        size_differences = []
        distances = []
        length = len(trajectory)

        # Loop through the trajectory to calculate size differences and distances
        avg_area = 0
        for i in range(len(trajectory) - 1):
            x1, y1, area1, _ = trajectory[i]
            x2, y2, area2, _ = trajectory[i + 1]

            avg_area += area1

            # Percentage size difference
            avg_size = (area1 + area2) / 2
            size_difference = abs(area1 - area2) / avg_size
            size_differences.append(size_difference)

            # Distance between consecutive points
            distances.append(calculate_distance(x1, y1, x2, y2))
        avg_area += trajectory[-1][2]
        avg_area /= len(trajectory)
        avg_areas.append(avg_area)

        # Average size difference and distance
        avg_size_diff = sum(size_differences) / len(size_differences)
        avg_distance = sum(distances) / len(distances)

        # Loss for this trajectory
        loss = (
            size_weight * avg_size_diff +
            distance_weight * avg_distance -
            length_weight * length  # Longer trajectories reduce loss
        )
        losses.append(loss)
    print(f"length of loss vector: {len(losses)}")
    min_loss_idx = losses.index(min(losses))
    best_trajectory = trajectories[min_loss_idx]
    return best_trajectory, losses, trajectories


def display_frames_and_trajectories(video_path, all_contours, frame_height, frame_width, ball_location, angle_threshold=25, distance_threshold=50, size_threshold=0.5):
    """
    Display the video frames and trajectories being built in real-time with additional checks for size and distance.

    Parameters:
    - video_path: Path to the video file.
    - all_contours: List of contours for each frame.
    - frame_height: Height of the video frames.
    - frame_width: Width of the video frames.
    - ball_location: Starting location of the ball (x, y).
    - angle_threshold: Maximum allowable angle difference for trajectory matching (degrees).
    - distance_threshold: Maximum allowable distance for trajectory matching (pixels).
    - size_threshold: Maximum allowable size difference for trajectory matching (fraction of avg size, e.g., 0.5 = 50%).
    """
    def calculate_distance(x1, y1, x2, y2):
        return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5

    # Initialize video capture
    cap = cv2.VideoCapture(video_path)
    num_frames = len(all_contours)

    # Initialize trajectories with area tracking
    trajectories = []  # Each trajectory is a list of (x, y, area, frame_idx)

    # Loop through frames
    for frame_idx in range(num_frames):
        # Read the next video frame
        print(f"Processing frame number {frame_idx}")
        ret, frame = cap.read()
        if not ret:
            print(f"End of video or invalid frame at index {frame_idx}")
            break

        # Get the contours for the current frame
        current_contours = all_contours[frame_idx]

        # Draw contours on the frame
        for (x, y, w, h) in current_contours:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green rectangles for contours

        # Update trajectories with the current frame's contours
        new_trajectories = []
        for contour in current_contours:
            x, y, w, h = contour
            cx, cy = x + w / 2, y + h / 2  # Center of the contour
            area = w * h

            for trajectory in trajectories:
                # Extract trajectory endpoints and calculate avg area
                tx, ty, _, _ = trajectory[-1]  # Last point in trajectory
                avg_area = np.mean([point[2] for point in trajectory])

                # Calculate checks
                distance_check = calculate_distance(tx, ty, cx, cy) <= distance_threshold
                size_check = abs(area - avg_area) <= size_threshold * avg_area

                if len(trajectory) == 1:
                    # Use ball_location for the initial angle
                    angle = calculate_angle(ball_location[0], ball_location[1], tx, ty)
                    new_angle = calculate_angle(tx, ty, cx, cy)
                else:
                    tx1, ty1, _, _ = trajectory[-2]  # Second-to-last point in trajectory
                    angle = calculate_angle(tx1, ty1, tx, ty)
                    new_angle = calculate_angle(tx, ty, cx, cy)

                angle_check = abs(new_angle - angle) <= angle_threshold

                # Add the contour to the trajectory if all checks pass
                if (angle_check or distance_check) and size_check:
                    trajectory.append((cx, cy, area, frame_idx))

            # Start a new trajectory if the contour was not added to any existing trajectory
            new_trajectories.append([(cx, cy, area, frame_idx)])

        # Add the new trajectories to the main list
        trajectories.extend(new_trajectories)

    best_trajectory, loss, filtered_trajectories = find_best_trajectory(trajectories)
    plot_all_trajectories(filtered_trajectories, frame_height, frame_width, best_trajectory=best_trajectory)
    return best_trajectory, filtered_trajectories


def fill_trajectory_gaps(best_trajectory, ball_location):
    """
    Fill in gaps in the trajectory by interpolating points for missing frames.

    Parameters:
    - best_trajectory: List of (x, y, area, frame_idx) representing the best trajectory.
    - ball_location: Tuple (x, y) of the original ball location.

    Returns:
    - filled_trajectory: List of (x, y, area, frame_idx) with gaps filled.
    """
    if not best_trajectory:
        return []

    # Start with the original ball location as the first point
    filled_trajectory = [(ball_location[0], ball_location[1], best_trajectory[0][2], 0)]

    for i in range(len(best_trajectory)):
        current_point = best_trajectory[i]
        gap = current_point[3] - filled_trajectory[-1][3]
        # if gap is larger than 1, it means we are missing some points.
        if gap > 0:
            # find the slope connecting the previous and current points.
            try:
                interpolation_slope = (current_point[1] - filled_trajectory[-1][1]) / (current_point[0] - filled_trajectory[-1][0])
            except:
                interpolation_slope = 100000
            # find the x intervals to add points.
            x_interval = (current_point[0] - filled_trajectory[-1][0]) / gap
            # find the amount by which the y value changes each time
            delta_y = interpolation_slope * x_interval
            # add in the points for each gap point.
            prev_y = filled_trajectory[-1][1]
            curr_frame = filled_trajectory[-1][3]
            curr_area = filled_trajectory[-1][2]
            new_vals = []
            for j in range(gap-1):
                x = filled_trajectory[-1][0] + j * x_interval # find the new x point to add in
                y = prev_y + delta_y*(j+1)
                new_vals.append((x, y, curr_area, curr_frame+1+j))
            # Add the current point to the filled trajectory
            filled_trajectory.extend(new_vals)
        # add in the value from the current trajectory into the filled_trajectory list
        filled_trajectory.append(current_point)
    return filled_trajectory


def visualize_trajectories(video_path, filtered_trajectories, contact_frame):
    """
    Visualize multiple trajectories simultaneously by displaying the traced paths so far.

    Parameters:
    - video_path: str, path to the video file.
    - filtered_trajectories: List of trajectories. Each trajectory is a list of (x, y, area, frame_idx).
    - contact_frame: int, the frame number where the trajectory visualization should start.

    Returns:
    None
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Unable to open video file at {video_path}.")
        return

    # Generate a unique color for each trajectory
    trajectory_colors = [
        (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        for _ in range(len(filtered_trajectories))
    ]

    frame_idx = 0
    while True:
        current_frame = contact_frame + frame_idx
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
        ret, frame = cap.read()
        if not ret:
            print(f"End of video or invalid frame at {current_frame}.")
            break

        # Draw the traced trajectories up to this frame
        for traj_idx, traj in enumerate(filtered_trajectories):
            color = trajectory_colors[traj_idx]
            points = traj[:frame_idx + 1]  # Get all points up to the current frame
            if len(points) > 1:
                for i in range(len(points) - 1):
                    # Draw lines between consecutive points
                    x1, y1, _, _ = points[i]
                    x2, y2, _, _ = points[i + 1]
                    cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness=4)

        # Show the frame
        cv2.imshow("Trajectory Visualization", frame)

        # Wait for key press
        key = cv2.waitKey(0) & 0xFF
        if key == ord('n'):  # Press 'n' to move to the next frame
            frame_idx += 1
            continue
        elif key == 27:  # Press ESC to quit
            break

    cap.release()
    cv2.destroyAllWindows()


def visualize_multiple_trajectories(video_path, filtered_trajectories, best_trajectory, contact_frame):
    """
    Visualize multiple trajectories simultaneously by drawing rectangles on the video frames.

    Parameters:
    - video_path: str, path to the video file.
    - filtered_trajectories: List of trajectories. Each trajectory is a list of (x, y, area, frame_idx).
    - best_trajectory: List of (x, y, area, frame_idx), the best trajectory points.
    - contact_frame: int, the frame number where the trajectory visualization should start.

    Returns:
    None
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Unable to open video file at {video_path}.")
        return

    # Generate a unique color for each trajectory
    trajectory_colors = [
        (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        for _ in range(len(filtered_trajectories))
    ]

    frame_idx = 0
    while True:
        current_frame = contact_frame + frame_idx
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
        ret, frame = cap.read()
        if not ret:
            print(f"End of video or invalid frame at {current_frame}.")
            break

        # Draw all filtered trajectories
        for traj_idx, traj in enumerate(filtered_trajectories):
            color = trajectory_colors[traj_idx]
            try:
                (x, y, area, _) = traj[frame_idx]
                side_length = int(np.sqrt(area))  # Approximate square size from area
                top_left = (int(x - side_length / 2), int(y - side_length / 2))
                bottom_right = (int(x + side_length / 2), int(y + side_length / 2))
                cv2.rectangle(frame, top_left, bottom_right, color, 2)
            except IndexError:
                # Ignore if trajectory doesn't have a point for the current frame
                continue

        # Draw the best trajectory with a thicker outline
        try:
            (x, y, area, _) = best_trajectory[frame_idx]
            side_length = int(np.sqrt(area))  # Approximate square size from area
            top_left = (int(x - side_length / 2), int(y - side_length / 2))
            bottom_right = (int(x + side_length / 2), int(y + side_length / 2))
            cv2.rectangle(frame, top_left, bottom_right, (255, 0, 0), 4)  # Red with thicker outline
        except IndexError:
            # Ignore if best trajectory doesn't have a point for the current frame
            pass

        # Show the frame
        cv2.imshow("Trajectory Visualization", frame)

        # Wait for key press
        key = cv2.waitKey(0) & 0xFF
        if key == ord('n'):  # Press 'n' to move to the next frame
            frame_idx += 1
            continue
        elif key == 27:  # Press ESC to quit
            break

    cap.release()
    cv2.destroyAllWindows()


def visualize_best_trajectory(video_path, best_trajectory, contact_frame):
    """
    Visualize the best trajectory by displaying its traced path in blue.

    Parameters:
    - video_path: str, path to the video file.
    - best_trajectory: List of (x, y, area, frame_idx), the best trajectory points.
    - contact_frame: int, the frame number where the trajectory visualization should start.

    Returns:
    None
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Unable to open video file at {video_path}.")
        return

    frame_idx = 0
    while True:
        current_frame = contact_frame + frame_idx
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
        ret, frame = cap.read()
        if not ret:
            print(f"End of video or invalid frame at {current_frame}.")
            break

        # Draw the traced best trajectory up to this frame
        points = best_trajectory[:frame_idx + 1]  # Get all points up to the current frame
        if len(points) > 1:
            for i in range(len(points) - 1):
                # Draw lines between consecutive points
                x1, y1, _, _ = points[i]
                x2, y2, _, _ = points[i + 1]
                cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), thickness=2)  # Blue color

        # Show the frame
        cv2.imshow("Best Trajectory Visualization", frame)

        # Wait for key press
        key = cv2.waitKey(0) & 0xFF
        if key == ord('n'):  # Press 'n' to move to the next frame
            frame_idx += 1
            continue
        elif key == 27:  # Press ESC to quit
            break

    cap.release()
    cv2.destroyAllWindows()


def main(batch_number, sample_number):
    ball_location = np.load(f"/Users/nolanjetter/Documents/GitHub/Soccer ML Project Main/output/Batch {batch_number}/ball_locations.npy")[sample_number]
    video_path = f"/Users/nolanjetter/Documents/GitHub/Soccer ML Project Main/dataset/Session {batch_number}/Kick {sample_number}.mp4"
    sparse_matrix = np.load(f"/Users/nolanjetter/Documents/GitHub/Soccer ML Project Main/output/Batch {batch_number}/RPCA_Results/sparse_sample_{sample_number}.npy")  # Load your saved sparse matrix
    frame_height = 540  # Adjust based on your data
    frame_width = 960  # Adjust based on your data
    contours = process_sparse_matrix(sparse_matrix, frame_height, frame_width)
    print("contours found...")
    saved_ball_locations = []
    loss_array = [1]

    all_contours = []

    frame_counter = np.load(f"/Users/nolanjetter/Documents/GitHub/Soccer ML Project Main/output/Batch {batch_number}/contact_frames_{batch_number}/contact_frames.npy")[sample_number-1]
    first_val = frame_counter.copy()
    while True:
        try:
            if frame_counter == first_val:
                saved_ball_locations.append([ball_location[0], ball_location[1]])
                prev_ball_location = ball_location
            else:
                prev_ball_location = saved_ball_locations[-1]
            # Open the video at the specified frame
            cap, frame = open_video(video_path, target_frame=frame_counter)
            if frame is None:
                print(f"End of video or invalid frame: {frame_counter}")
                break

            # Get the contours for the current frame from the processed sparse matrix
            current_contours = contours[frame_counter]
            scaled_contours = [(2 * x, 2 * y, 2 * w, 2 * h) for (x, y, w, h) in current_contours]
            all_contours.append(scaled_contours)

            # Draw the contours
            for (x, y, w, h) in scaled_contours:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green for contours

            # if best_pair:
            #     # Draw the best circle
            #     cx, cy, r = best_pair["circle"]
            #     cv2.circle(frame, (cx, cy), r, (255, 255, 0), 5)  # Blue for the best circle
            #
            #     # Draw the best rectangle
            #     x, y, w, h = best_pair["rectangle"]
            #     cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 5)  # Green for the best rectangle

            # Show the frame with bounding boxes
        #     cv2.imshow("Ball Tracking", frame)
        #
        #     # Wait for user interaction
        #     key = cv2.waitKey(0) & 0xFF
        #     if key == 27:  # ESC key
        #         frame_counter += 1  # Move to the next frame
        #     elif key == ord('q'):  # 'q' key to quit
        #         break
            frame_counter += 1
        except Exception as e:
            print(f"Error processing frame {frame_counter}: {e}")
            break
    cap.release()
    cv2.destroyAllWindows()

    print("finding best trajectory")
    best_trajectory, filtered_trajectories = display_frames_and_trajectories(video_path, all_contours, frame_height, frame_width, ball_location)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.clear()
    ax.set_title("Best Trajectory")
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    ax.invert_yaxis()  # Flip Y-axis for image-style coordinates
    # Convert the best trajectory to a NumPy array for plotting
    best_filled_trajectory = fill_trajectory_gaps(best_trajectory, ball_location)

    # save the best filled trajectory
    # save_directory = f"/Users/nolanjetter/Documents/GitHub/Soccer ML Project Main/output/Batch 1/Ball Trajectories/sample_{sample_number}.npy"
    # np.save(save_directory, np.array(best_filled_trajectory), allow_pickle=True)

    filtered_filled_trajectories = []
    for traj in filtered_trajectories:
        filtered_filled_trajectories.append(fill_trajectory_gaps(traj, ball_location))

    visualize_trajectories(video_path, filtered_filled_trajectories, first_val)
    visualize_best_trajectory(video_path, best_filled_trajectory, first_val)
    visualize_multiple_trajectories(video_path, filtered_filled_trajectories, best_filled_trajectory, first_val)


for i in range(1, 2):
    try:
        main(1, i)
    except:
        print(f"could not complete for sample number {i}")