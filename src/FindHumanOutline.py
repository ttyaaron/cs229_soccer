import numpy as np
import cv2
import os


def smooth_x_positions(bounding_boxes, weight=0.5):
    """
    Smooth x-coordinates of bounding boxes using a recursive weighted averaging method
    that collapses inward from the first and last frames toward the middle.

    Args:
        bounding_boxes (list of tuples): List of bounding boxes as (x, y, w, h).
        weight (float): Weight for the recursive averaging (0 < weight <= 1).

    Returns:
        smoothed_bounding_boxes (list of tuples): Bounding boxes with smoothed x-coordinates.
    """
    # Initialize smoothed list with the original bounding boxes
    smoothed_boxes = bounding_boxes.copy()

    # Extract x-coordinates for smoothing
    x_positions = [box[0] if box is not None else None for box in bounding_boxes]

    # Recursive inward smoothing
    left_smoothed = x_positions.copy()
    right_smoothed = x_positions.copy()

    for i in range(1, len(x_positions)):
        # Smooth from the left
        if left_smoothed[i] is not None and left_smoothed[i - 1] is not None:
            left_smoothed[i] = (weight * left_smoothed[i - 1] + (1 - weight) * left_smoothed[i])

        # Smooth from the right
        j = len(x_positions) - 1 - i
        if right_smoothed[j] is not None and right_smoothed[j + 1] is not None:
            right_smoothed[j] = (weight * right_smoothed[j + 1] + (1 - weight) * right_smoothed[j])

    # Combine the results by averaging the left and right smoothed values
    for i in range(len(x_positions)):
        if left_smoothed[i] is not None and right_smoothed[i] is not None:
            smoothed_x = (left_smoothed[i] + right_smoothed[i]) / 2
            smoothed_boxes[i] = (smoothed_x, smoothed_boxes[i][1], smoothed_boxes[i][2], smoothed_boxes[i][3])

    return smoothed_boxes


def reshape_frame(frame_column, height, width):
    """Reshape a 1D column vector into a 2D matrix representing a frame."""
    return frame_column.reshape((height, width))


def calculate_optical_flow(frame1, frame2, motion_threshold=2.0):
    """
    Calculate motion between two frames using dense optical flow.
    """
    flow = cv2.calcOpticalFlowFarneback(frame1, frame2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    motion_mask = (mag > motion_threshold).astype(np.uint8) * 255
    return motion_mask


def convert_contour_to_box(contours):
    """
    Converts contours to bounding boxes.

    Args:
        contours (list): List of contours, where each contour is a list of points.

    Returns:
        boxes (list of tuples): List of bounding boxes as (x, y, w, h).
    """
    boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        boxes.append((x, y, w, h))
    return boxes


def adjust_bounding_boxes(bounding_boxes):
    """
    Adjust bounding boxes based on the time-adjacent frames by taking the max y
    and min y from up to 4 neighbors on either side.

    Args:
        bounding_boxes (list of tuples): Bounding boxes for all frames as (x, y, w, h).

    Returns:
        adjusted_boxes (list of tuples): Adjusted bounding boxes.
    """
    adjusted_boxes = bounding_boxes.copy()

    for t, box in enumerate(bounding_boxes):
        if box is None:
            continue  # Skip frames with no bounding boxes

        x, y, w, h = box
        y_min = y
        y_max = y + h

        # Collect y ranges from up to 4 neighbors on either side
        for offset in range(1, 5):  # Look at neighbors 1 through 4 on each side
            # Previous neighbors
            if t - offset >= 0 and bounding_boxes[t - offset] is not None:
                _, y_prev, _, h_prev = bounding_boxes[t - offset]
                y_min = min(y_min, y_prev)
                y_max = max(y_max, y_prev + h_prev)

            # Next neighbors
            if t + offset < len(bounding_boxes) and bounding_boxes[t + offset] is not None:
                _, y_next, _, h_next = bounding_boxes[t + offset]
                y_min = min(y_min, y_next)
                y_max = max(y_max, y_next + h_next)

        # Update the box
        adjusted_boxes[t] = (x, y_min, w, y_max - y_min)

    return adjusted_boxes


def combine_and_select_largest_bounding_box(bounding_boxes, proximity_threshold=10):
    """
    Combines overlapping or close bounding boxes and selects the largest one.

    Args:
        bounding_boxes (list of tuples): List of bounding boxes as (x, y, w, h).
        proximity_threshold (int): Distance threshold to consider boxes as close.

    Returns:
        tuple: The largest bounding box (x, y, w, h) after combining, or None if no boxes.
    """
    if not bounding_boxes:
        return None

    def is_close(box1, box2):
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2

        # Check if boxes overlap or are within proximity_threshold
        return not (
            x1 + w1 + proximity_threshold < x2 or
            x2 + w2 + proximity_threshold < x1 or
            y1 + h1 + proximity_threshold < y2 or
            y2 + h2 + proximity_threshold < y1
        )

    # Combine overlapping/close boxes
    combined_boxes = []
    for box in bounding_boxes:
        merged = False
        for i, combined_box in enumerate(combined_boxes):
            if is_close(box, combined_box):
                x1, y1, w1, h1 = combined_box
                x2, y2, w2, h2 = box
                new_x = min(x1, x2)
                new_y = min(y1, y2)
                new_w = max(x1 + w1, x2 + w2) - new_x
                new_h = max(y1 + h1, y2 + h2) - new_y
                combined_boxes[i] = (new_x, new_y, new_w, new_h)
                merged = True
                break
        if not merged:
            combined_boxes.append(box)

    # Find the largest bounding box
    largest_box = max(combined_boxes, key=lambda b: b[2] * b[3])  # Maximize area (w * h)

    return largest_box


def adjust_bounding_boxes(bounding_boxes):
    """
    Adjust bounding boxes based on the time-adjacent frames by taking the max y
    and min y from neighboring frames.

    Args:
        bounding_boxes (list of tuples): Bounding boxes for all frames.

    Returns:
        adjusted_boxes (list of tuples): Adjusted bounding boxes.
    """
    adjusted_boxes = bounding_boxes.copy()

    for t, box in enumerate(bounding_boxes):
        if box is None:
            continue  # Skip frames with no boxes

        x, y, w, h = box
        y_min = y
        y_max = y + h

        # Consider the previous frame's box
        if t > 0 and bounding_boxes[t - 1] is not None:
            _, y_prev, _, h_prev = bounding_boxes[t - 1]
            y_min = min(y_min, y_prev)
            y_max = max(y_max, y_prev + h_prev)

        # Consider the next frame's box
        if t < len(bounding_boxes) - 1 and bounding_boxes[t + 1] is not None:
            _, y_next, _, h_next = bounding_boxes[t + 1]
            y_min = min(y_min, y_next)
            y_max = max(y_max, y_next + h_next)

        # Update the box
        adjusted_boxes[t] = (x, y_min, w, y_max - y_min)

    return adjusted_boxes


def detect_motion_with_bounding_boxes(file_path, frame_height, frame_width, output_dir, motion_threshold=2.0, weight=0.5, visualize=False):
    """
    Detect motion using optical flow, combine bounding boxes, select the largest box,
    and adjust bounding boxes using temporal smoothing. Visualizes the results.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    # Load sparse matrix
    sparse_matrix = np.load(file_path)
    total_pixels, num_frames = sparse_matrix.shape

    if total_pixels != frame_height * frame_width:
        raise ValueError(f"Invalid dimensions: Expected {frame_height * frame_width} pixels, got {total_pixels}.")

    # Initialize list to store the largest bounding box for each frame
    largest_boxes = []

    prev_frame = cv2.normalize(sparse_matrix[:, 0].reshape((frame_width, frame_height)), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    for i in range(1, num_frames):
        sparse_frame = sparse_matrix[:, i].reshape((frame_width, frame_height))
        curr_frame = cv2.normalize(sparse_frame, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # Calculate motion
        motion_mask = calculate_optical_flow(prev_frame, curr_frame, motion_threshold)

        # Find contours of motion
        contours, _ = cv2.findContours(motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Convert contours to bounding boxes
        boxes = convert_contour_to_box(contours)

        # Combine and select the largest bounding box
        largest_box = combine_and_select_largest_bounding_box(boxes)
        largest_boxes.append(largest_box)

        prev_frame = curr_frame

    # Adjust bounding boxes using time-adjacent frames
    adjusted_bounding_boxes = adjust_bounding_boxes(largest_boxes)

    # Smooth x-coordinates recursively
    smoothed_bounding_boxes = smooth_x_positions(adjusted_bounding_boxes, weight=weight)

    # Visualize smoothed bounding boxes
    for i, box in enumerate(smoothed_bounding_boxes):
        if box is None:
            continue  # Skip frames with no bounding boxes

        x, y, w, h = box
        sparse_frame = sparse_matrix[:, i].reshape((frame_width, frame_height))
        frame = cv2.normalize(sparse_frame, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        if visualize:
            # Draw the smoothed bounding box
            cv2.rectangle(frame_bgr, (int(x), int(y)), (int(x + w), int(y + h)), (255, 0, 0), 2)  # Blue box
            cv2.imshow("Motion Detection with Smoothed Bounding Boxes", frame_bgr)
            # Press 'ESC' to quit, any other key to continue
            key = cv2.waitKey(0) & 0xFF
            if key == 27:  # ESC key
                break
        else:
            cropped_image = frame[int(y):int(y + h), int(x):int(x + w)]
            # Save the cropped image
            img_name = f"sample_{sample_number}_frame_{i}.png"
            img_path = os.path.join(output_dir, img_name)
            cv2.imwrite(img_path, cropped_image)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Define paths
    session_number = 3
    batch_path = f"/Users/nolanjetter/Documents/GitHub/Soccer ML Project Main/output/Batch {session_number}/RPCA_Results"

    sample_number = 1
    while True:
        print(f"processing sample number {sample_number}...")
        # Construct the file path for the current sample
        file_path = os.path.join(batch_path, f"sparse_sample_{sample_number}.npy")
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}, stopping visualization.")
            break

        output_base_dir = f"/Users/nolanjetter/Documents/GitHub/Soccer ML Project Main/output/Batch {session_number}/Player Images"
        detect_motion_with_bounding_boxes(file_path, frame_height=540, frame_width=960, output_dir=output_base_dir, motion_threshold=2.0, weight=0.5)
        sample_number += 1