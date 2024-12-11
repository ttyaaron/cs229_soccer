import numpy as np
import cv2
from matplotlib import pyplot as plt
import os


def rpca(M, lamb=None, tol=1e-7, max_iter=1000):
    """
    Robust Principal Component Analysis (RPCA) using Inexact Augmented Lagrange Multiplier (IALM).

    Args:
        M (numpy.ndarray): Input data matrix (e.g., video frames as columns).
        lamb (float): Regularization parameter (default: 1/sqrt(max(M.shape))).
        tol (float): Convergence tolerance for residual norm.
        max_iter (int): Maximum number of iterations.

    Returns:
        L (numpy.ndarray): Low-rank matrix.
        S (numpy.ndarray): Sparse matrix.
    """
    m, n = M.shape
    if lamb is None:
        lamb = 1.0 / np.sqrt(max(m, n))

    # Initialize variables
    L = np.zeros((m, n))
    S = np.zeros((m, n))
    Y = np.zeros((m, n))  # Dual variable
    mu = 1.25 / np.linalg.norm(M, ord=2)  # Step size
    mu_bar = mu * 1e7
    rho = 1.5  # Scaling factor for mu

    # Iterative optimization
    for i in range(max_iter):
        # Update L using Singular Value Thresholding (SVT)
        U, sigma, VT = np.linalg.svd(M - S + (1 / mu) * Y, full_matrices=False)
        sigma_thresh = np.maximum(sigma - (1 / mu), 0)  # Soft-thresholding
        L = U @ np.diag(sigma_thresh) @ VT

        # Update S using Elementwise Thresholding
        S = np.maximum(M - L + (1 / mu) * Y - lamb / mu, 0) + \
            np.minimum(M - L + (1 / mu) * Y + lamb / mu, 0)

        # Update Y (dual variable)
        Z = M - L - S
        Y += mu * Z

        # Convergence check
        norm_Z = np.linalg.norm(Z, ord='fro')
        if norm_Z / np.linalg.norm(M, ord='fro') < tol:
            print(f"Converged after {i + 1} iterations.")
            break

        # Adjust mu
        mu = min(mu * rho, mu_bar)

    return L, S


def visualize_rpca_results(S, frame_shape):
    """
    Visualize the low-rank and sparse components.

    Args:
        L (numpy.ndarray): Low-rank matrix (background).
        S (numpy.ndarray): Sparse matrix (foreground).
        frame_shape (tuple): Original frame dimensions (height, width).
    """
    num_frames = S.shape[1]
    for i in range(num_frames):
        # Reshape columns back to original frame size
        sparse_frame = S[:, i].reshape(frame_shape)

        plt.title("Sparse (Foreground)")
        plt.imshow(sparse_frame, cmap='gray')
        plt.show()


def load_video_frames(video_path, max_frames=100, use_color=False, downscale_factor=2):
    """
    Load video frames into a matrix where each column represents a frame.

    Args:
        video_path (str): Path to the video file.
        max_frames (int): Maximum number of frames to load.
        use_color (bool): Whether to include RGB channels.
        downscale_factor (int): Factor by which to downscale frames.

    Returns:
        numpy.ndarray: Data matrix (flattened frames as columns).
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_count = 0
    frame_dimensions = []
    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        # Downscale frame
        frame = cv2.resize(frame, (frame.shape[1] // downscale_factor, frame.shape[0] // downscale_factor))
        # Convert to grayscale or keep RGB
        if not use_color:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if len(frames) == 0:
            frame_dimensions = frame.shape
        # Flatten the frame

        flattened_frame = frame.flatten()
        # Normalize pixel values
        normalized_frame = flattened_frame / 255.0
        frames.append(normalized_frame)
        frame_count += 1
    cap.release()
    # Stack frames into a data matrix M
    return np.array(frames).T, frame_dimensions


def find_circles(gray_image, dp=1, min_dist=20, param1=20, param2=15, min_radius=0, max_radius=0):
    """
    Detect circles in a grayscale image using Hough Circle Transform.

    Parameters:
    - gray_image: np.ndarray, the grayscale image where circles need to be detected.
    - dp: float, the inverse ratio of the accumulator resolution to the image resolution.
    - min_dist: int, the minimum distance between detected circles' centers.
    - param1: int, higher threshold for Canny edge detection.
    - param2: int, accumulator threshold for circle detection.
    - min_radius: int, minimum radius of the circles.
    - max_radius: int, maximum radius of the circles.

    Returns:
    - circles: np.ndarray or None, detected circles as (x, y, radius). None if no circles are found.
    """
    # Detect circles using HoughCircles
    circles = cv2.HoughCircles(
        gray_image,
        cv2.HOUGH_GRADIENT,
        dp=dp,
        minDist=min_dist,
        param1=param1,
        param2=param2,
        minRadius=min_radius,
        maxRadius=max_radius
    )

    if circles is not None:
        # Convert (x, y, radius) values to integers
        circles = np.uint16(np.around(circles))

    return circles


if __name__ == "__main__":
    # Example usage
    # batch_number = 3  # Replace with user input or configuration
    # video_number = 25
    # keep_iterating = True
    # while keep_iterating:
    #     try:
    #         # Extract the matrix to be decomposed
    #         parent_dir = os.path.dirname(__file__)
    #         child_dir = f"../dataset/Session {batch_number}/Kick {video_number}.mp4"
    #         file_path = os.path.join(parent_dir, child_dir)
    #         M, frame_shape = load_video_frames(file_path)
    #
    #         # Apply RPCA
    #         L, S = rpca(M)
    #         print(f"Done with RPCA. kick number {video_number}")
    #
    #         # Save sparse representation
    #         save_directory = f"/Users/nolanjetter/Documents/GitHub/Soccer ML Project Main/output/Batch {batch_number}/RPCA_Results"
    #         os.makedirs(save_directory, exist_ok=True)  # Ensure the directory exists
    #         save_path = os.path.join(save_directory, f"sparse_sample_{video_number}")
    #         np.save(save_path, S, allow_pickle=True)
    #         video_number += 1
    #     except:
    #         keep_iterating = False
    #         video_number = 1

    # visualize the sparse matrix

    M, frame_shape = load_video_frames("/Users/nolanjetter/Documents/GitHub/Soccer ML Project Main/dataset/Session 2/Kick 1.mp4")
    S_matrix = np.load("/Users/nolanjetter/Documents/GitHub/Soccer ML Project Main/output/Batch 2/RPCA_Results/sparse_sample_1.npy", allow_pickle=True)
    visualize_rpca_results(S_matrix, frame_shape)
