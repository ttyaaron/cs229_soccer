# This document provides functions in handy, 
import numpy as np

def pose25_normalization(pose25_array, reference_point_idx, reference_frame_idx = 0, norm_hyperparam=1.0):
    '''
    Normalize the input pose25_array based on the reference_point and the reference_frame.
    x = (x - x_ref)*zoom_ratio, where zoom_ratio is norm_hyperparam / height_of_reference pose.
    pose25_array comes in shape (n, 25, 3).
    '''
    pose25_array = pose25_array.reshape(-1,25,3)
    # confidence_flag = False
    # while confidence_flag == False:
    #     # Get the reference frame pose
    #     reference_frame_pose = pose25_array[reference_frame_idx, :, :]
    #     if reference_frame_pose[reference_point_idx, 2] == 0:
    #         reference_frame_idx += 1
    #         continue
    #     else:
    #         confidence_flag = True
    reference_frame_pose = pose25_array[reference_frame_idx, :, :]


    # Calculate the height of the reference frame
    height_of_reference = np.max(reference_frame_pose[:, 1]) - np.min(reference_frame_pose[:, 1])
    print("height of reference:", height_of_reference)

    # Avoid division by zero in case height_of_reference is zero
    if height_of_reference == 0:
        raise ValueError("The height of the reference frame is zero, normalization cannot be performed.")

    # Calculate the zoom ratio
    zoom_ratio = norm_hyperparam / height_of_reference
    
    # Get the reference point
    reference_point = pose25_array[reference_frame_idx, reference_point_idx, :]

    # Create a copy of the reference frame pose for the output
    output_pose = np.copy(pose25_array)

    # Normalize x and y coordinates based on the reference point and zoom ratio
    output_pose[:,:, 0] -= reference_point[0]  # Normalize x
    output_pose[:,:, 0] *= zoom_ratio

    output_pose[:,:, 1] -= reference_point[1]  # Normalize y
    output_pose[:,:, 1] *= zoom_ratio

    return output_pose

def get_target_joint_xy_series_from_Pose25(normalized_pose25_array, target_joint_idx):
    """
    Extract the time series of a specific joint from the normalized_pose25_array.
    
    Parameters:
    normalized_pose25_array: numpy array of shape (n, 25, 3) 
    target_joint_idx: Index of the joint to extract (1-based index).
    
    Returns:
    output_array: Array of shape (n, 2) containing the x and y coordinates of the target joint.
    """
    output_array = []
    # Extract both x and y coordinates of the target joint
    for idx in range(normalized_pose25_array.shape[0]):
        if normalized_pose25_array[idx, target_joint_idx, 2] == 0: # When the confidence is 0
            continue
        output_array.append(normalized_pose25_array[idx, target_joint_idx, 0:2])  # x and y coordinates
    #output_array = output_array.reshape(-1,2)
    return output_array

def target_joint_xy_polyfit(target_joint_xy_series, order=4):
    """
    Fit a polynomial of a given order to the x and y coordinates of the target joint over time.
    
    Parameters:
    target_joint_xy_series: numpy array of shape (n, 2), where n is the total number of time frames.
    order: The order of the polynomial to fit (default is 4).
    
    Returns:
    (x_fit_param, y_fit_param, total_frame_num): A tuple containing the polynomial coefficients for x(t), y(t), and the total number of frames.
    """

    # Get the total number of frames
    total_frame_num = target_joint_xy_series.shape[0]  # Correct shape access with [0]
    print("total_frame_num is ", total_frame_num)
    # print("target_joint_xy_series is ", target_joint_xy_series)
    
    # Create a time array (t) corresponding to the number of frames
    t = np.arange(total_frame_num)
    # print("t is ", t)
    
    # Extract x and y coordinates from the input series
    x = target_joint_xy_series[:, 0]
    y = target_joint_xy_series[:, 1]

    # print("x is ", x)
    # print("y is ", y)
    # print("Normalized t:",t / total_frame_num)
    
    # Use polyfit to fit a polynomial of the specified order to the x and y coordinates
    x_fit_param = np.polyfit(t / total_frame_num, x, order)  # Normalize t by total_frame_num
    y_fit_param = np.polyfit(t / total_frame_num, y, order)
    
    return (x_fit_param, y_fit_param, total_frame_num)  # Output polynomial parameters and total frame number