import cv2
import numpy as np
import time


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
def detect_ball(motion_mask, min_area=50):
    """
    Detect the moving soccer ball using the motion mask.

    Parameters:
    - motion_mask: np.ndarray, binary mask from optical flow.
    - min_area: int, minimum area of the motion blob to be considered as the ball.

    Returns:
    - ball_positions: list of tuples, detected ball positions as (x, y, w, h).
    """
    contours, _ = cv2.findContours(motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    ball_positions = []
    for cnt in contours:
        if cv2.contourArea(cnt) > min_area:
            x, y, w, h = cv2.boundingRect(cnt)
            ball_positions.append((x, y, w, h))
    return ball_positions


# Main function to process the single numpy array
def process_sparse_matrix(sparse_matrix, frame_height, frame_width):
    """
    Process the sparse matrix to detect and track the moving soccer ball.

    Parameters:
    - sparse_matrix: np.ndarray, the combined matrix of shape (features, frames).
    - frame_height: int, the height of each frame.
    - frame_width: int, the width of each frame.
    """
    num_frames = sparse_matrix.shape[1]
    print(num_frames)
    if num_frames < 2:
        print("Need at least two frames to calculate optical flow.")
        return

    for i in range(num_frames - 1):
        # Extract and reshape consecutive frames
        print(f"Processing frame {i} and frame {i + 1}")
        frame1 = reshape_frame(sparse_matrix[:, i], frame_height, frame_width)
        frame2 = reshape_frame(sparse_matrix[:, i + 1], frame_height, frame_width)

        # Normalize frames to uint8
        frame1 = cv2.normalize(frame1, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        print(f"Reshaped frame shape: {frame1.shape}")
        frame2 = cv2.normalize(frame2, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # Calculate motion using optical flow
        motion_mask = calculate_optical_flow(frame1, frame2)

        # Detect the soccer ball in the motion mask
        ball_positions = detect_ball(motion_mask)

        # Visualize the results
        motion_overlay = cv2.cvtColor(frame1, cv2.COLOR_GRAY2BGR)
        for (x, y, w, h) in ball_positions:
            cv2.rectangle(motion_overlay, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Show the motion mask and ball detection
        cv2.imshow("Motion Mask", motion_mask * 255)
        cv2.imshow("Ball Detection", motion_overlay)

        # Break on ESC key
        if cv2.waitKey(0) & 0xFF == 27:
            continue

    cv2.destroyAllWindows()


# Example usage
sparse_matrix = np.load("/Users/nolanjetter/Documents/GitHub/Soccer ML Project Main/output/Batch 1/RPCA_Results/sparse_sample_1.npy")  # Load your saved sparse matrix
frame_height = 540  # Adjust based on your data
frame_width = 960  # Adjust based on your data
process_sparse_matrix(sparse_matrix, frame_height, frame_width)
print(f"Sparse matrix shape: {sparse_matrix.shape}")