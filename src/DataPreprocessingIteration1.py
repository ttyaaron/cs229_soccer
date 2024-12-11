import numpy as np
import os
from FindBallLocation import FindBallLocation
from VideoSoundAnalysis import process_kick_videos
from FindFirstPhase import find_foot_plant_information
from PlotPoints import load_keypoints_from_json, plot_keypoints
from MotionTrajExtract import pose25_normalization, get_target_joint_xy_series_from_Pose25, target_joint_xy_polyfit

import matplotlib.pyplot as plt

# OpenPose body keypoint connections for drawing skeletons
POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),  # Head, neck, left arm
    (1, 5), (5, 6), (6, 7),  # Right arm
    (1, 8), (8, 9), (8, 12),  # Torso
    (9, 10), (10, 11),  # Left leg
    (12, 13), (13, 14),  # Right leg
    (11, 24), (11, 22), (22, 23),  # Left foot
    (14, 21), (14, 19), (19, 20),  # Right foot
    (0, 15), (0, 16), (15, 17), (16, 18)  # Eyes and ears
]


# Definition of the polynomial function (2D polynomial evaluation)
def polynomial(coefficients, t):
    """Evaluate a polynomial with given coefficients at points t."""
    highest_deg = len(coefficients) - 1
    return sum(coef * (t ** (highest_deg - i)) for i, coef in enumerate(coefficients))

def draw_pose(ax, keypoints):
    """Draw the pose skeleton on the given axis."""
    # Create a copy of the keypoints to avoid modifying the original
    keypoints = np.array(keypoints)  # Ensure keypoints is a NumPy array

    # Draw connections
    for (start, end) in POSE_CONNECTIONS:
        if keypoints[start, 2] > 0 and keypoints[end, 2] > 0:  # Confidence check
            ax.plot([keypoints[start, 0], keypoints[end, 0]], 
                    [keypoints[start, 1], keypoints[end, 1]], 
                    'b-', linewidth=2)
    
    # Draw keypoints
    for point in keypoints:
        if point[2] > 0:  # Confidence check
            ax.scatter(point[0], point[1], color='green', s=30)  # Draw circles for keypoints

def animate_pose_with_trajectory(normalized_pose_array, x_fit_param, y_fit_param, num_frames):
    """Animate Pose25 along with the fitted trajectory."""
    # Prepare the figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))
    
    t_seeds = np.linspace(0, 1, num_frames)  # Create a normalized time parameter
    x_trajectory = polynomial(x_fit_param, t_seeds)
    y_trajectory = polynomial(y_fit_param, t_seeds)

    for i in range(num_frames):
        ax.clear()  # Clear the current plot
        # Draw the pose keypoints for the current frame
        draw_pose(ax, normalized_pose_array[i])

        # Plot the trajectory on the same axis
        ax.plot(x_trajectory, y_trajectory, 'r-', label='Fitted Trajectory')  # Plot trajectory
        ax.set_xlim(ax.get_xlim()[0], ax.get_xlim()[1])  # Set x-axis limits (adjust as necessary)
        ax.set_ylim(ax.get_ylim()[0], ax.get_ylim()[1])  # Set y-axis limits (adjust as necessary)
        ax.set_title(f'Frame {i + 1}/{num_frames}')
        ax.legend(loc='upper right')

        ax.invert_yaxis()
        ax.set_aspect('equal', adjustable='box')  # Ensure equal scaling on both axes
        
        plt.pause(0.05)  # Pause for a short time to create animation effect

    plt.show()  # Show the completed plot

def animate(ax, pose_keypoints, i):
    """Animation function that updates the plot for each frame."""
    plot_keypoints(ax, pose_keypoints[i])  # Use pose_keypoints[i] for the current frame

def visualize_pose_with_poly(pose_keypoints, ax, x_fit_param, y_fit_param):
    """Visualizes the pose along with the fitted polynomial trajectory."""
    t_seeds = np.arange(0, 1, 0.001)  # Use np.arange to create an array of t values
    x = polynomial(x_fit_param, t_seeds)
    y = polynomial(y_fit_param, t_seeds)
    
    for i in range(pose_keypoints.shape[0]):
        # animate(ax, pose_keypoints, i)  # Animate the pose keypoints at frame i
        ax.plot(x, y, 'g-', label='Fitted Trajectory')  # Plot the trajectory on the same plot
        plt.pause(0.05)  # Pause to visualize the frame for 50 ms
    plt.show()  # Show the completed plot after animation

# ---------------------------------------------------------------
# Using the video sound analysis module, generate a list corresponding to the frame of impact.
num_kicks = 20  # Adjust this to the number of kick videos you have
batch_number = 1  # Set your batch number here
output_dir = f"..\\output\\contact_frames_{batch_number}"
os.makedirs(output_dir, exist_ok=True)

# Path to the contact frames file
contact_frames_file = os.path.join(output_dir, "contact_frames.npy")

# Check if the contact_frames.npy file already exists
if os.path.exists(contact_frames_file):
    # Load the contact frames array from the existing file
    contact_frames_array = np.load(contact_frames_file)
    print("Loaded existing contact_frames_array from:", contact_frames_file)
else:
    # Generate contact frames and save the output
    contact_frames_array = process_kick_videos(num_kicks, batch_number)
    np.save(contact_frames_file, contact_frames_array)
    print("Generated and saved new contact_frames_array to:", contact_frames_file)
# ---------------------------------------------------------------

# ---------------------------------------------------------------
# Using the Soccer Ball Location module, find the location of the soccer ball
ball_location = np.load(f"/Users/nolanjetter/Documents/GitHub/Soccer ML Project Main/output/Batch {batch_number}/ball_locations.npy", allow_pickle=True)
print(ball_location.shape)
for i in range(num_kicks, num_kicks+1):
    return_val = FindBallLocation(i, batch_number)
    print(f"return value: {return_val}")
    ball_location = np.append(ball_location, np.array(return_val).reshape(1, -1), axis=0)
print(ball_location.shape)
np.save(f"/Users/nolanjetter/Documents/GitHub/Soccer ML Project Main/output/Batch {batch_number}/ball_locations.npy",
        ball_location, allow_pickle=True)
# ---------------------------------------------------------------

# ---------------------------------------------------------------
# Find which foot is the plant foot and the frame the foot is planted.
foot_side_array = []
plant_frame_array = []
for i in range(num_kicks):
    foot_side, plant_frame = find_foot_plant_information(i + 1)
    print("plant_frame: ", plant_frame)
    
    if plant_frame == -1:
        plant_frame = 0


    foot_side_array.append(foot_side)
    plant_frame_array.append(plant_frame)
    # plant_foot_info = [find_foot_plant_information(i) for i in range(num_kicks)]
# ---------------------------------------------------------------

print("plant_frame_array: ", plant_frame_array) 
print("foot_side_array: ", foot_side_array) 

# The next part aims to extract trajectories of the contact foot
# during the time from 'plant foot frame' to 'contact frame'
# using polynomial fit parameterization.

# Import plant foot frame and contact frame (already available, called contact_frames_array)
plant_foot_frame_idx = plant_frame_array
print("Testing plant_foot_frame, Desired: 20, actual: %d \n" % len(plant_foot_frame_idx))


# Access the second column properly from the plant_foot_info numpy array
contact_frame_idx = contact_frames_array

print("Testing contact_frame, Desired: 20, actual: %d \n" % len(contact_frame_idx))
print("contact_frame_idx: ", contact_frame_idx) 

# Import plant foot recognition data
plant_foot = foot_side_array

# Load pose estimation data
pose_keypoints_array = []
x_fit_array = []
y_fit_array = []
frame_num_array = []

# Prepare figure and axis for visualization
fig, ax = plt.subplots()

for kick in range(num_kicks):
    plant_foot_frame = plant_foot_frame_idx[kick]
    print("plant_foot_frame is", plant_foot_frame)
    contact_frame = contact_frame_idx[kick]
    pose_keypoints_array = []

    for i in range(plant_foot_frame, contact_frame + 1):  # Include the contact frame as well
        json_file = os.path.join(
            os.path.dirname(__file__),
            f'../output/pose_estimation_results_{batch_number}/Kick_{kick + 1}_0000000000{str(i).zfill(2)}_keypoints.json'
        )
        pose_keypoints = load_keypoints_from_json(json_file)
        
        if pose_keypoints is not None:
            pose_keypoints_array.append(pose_keypoints)

    # Convert to a NumPy array before normalization
    pose_keypoints_array_np = np.array(pose_keypoints_array)
    pose_keypoints_array_np = pose_keypoints_array_np.reshape((-1,25,3))

    # Determine joint indices based on foot recognition
    if plant_foot[kick] == "left":
        reference_joint_idx = 11  # left_foot_index: left_foot_joints = [11, 22, 23, 24]
        target_joint_idx = 14
    else:
        reference_joint_idx = 14  # right_foot_index: right_foot_joints = [14, 19, 20, 21]
        target_joint_idx = 11

    # Normalize pose keypoints
    normalized_pose25_array = pose25_normalization(pose_keypoints_array_np, reference_joint_idx)
    
    # Extract x, y series of the target joint
    xy_series = get_target_joint_xy_series_from_Pose25(normalized_pose25_array, target_joint_idx)
    
    print("xy_series is", xy_series)

    # Fit polynomial to the x and y coordinates
    x_fit_param, y_fit_param, total_frame_num = target_joint_xy_polyfit(np.array(xy_series), order = 4)  # Order is 4
    
    # Store fitted parameters and total number of frames
    x_fit_array.append(x_fit_param)
    y_fit_array.append(y_fit_param)
    frame_num_array.append(total_frame_num)

    # Call the visualization function with the pose keypoints and fitted parameters
    animate_pose_with_trajectory(normalized_pose25_array, x_fit_param, y_fit_param, normalized_pose25_array.shape[0])

x_fit_array = np.array(x_fit_array)  
y_fit_array = np.array(y_fit_array)  

# Check if shapes are consistent before saving
print("x_fit_array shape:", x_fit_array.shape)
print("y_fit_array shape:", y_fit_array.shape)
print("frame_num_array:", frame_num_array)


np.save(os.path.join(output_dir, "contact_foot_trajs.npy"), 
        {'x_fit': x_fit_array, 'y_fit': y_fit_array, 'frame_nums': frame_num_array})

# ---------------------------------------------------------------
"""""
Next preprocessing steps:
1. gather the pose estimation keypoints corresponding to the frame of plant foot and frame of contact.
2. translate the points such that the ball is located at 0,0
3. normalize the data such that points lie from (-1) to 1.
4. gather labels from the annotations 
5. organize everything in a numpy array and write to local.
6. train some models :)
    - we can try using both the contact frame and the plant frame, or just the plant frame and see if
      there is a difference in performance. 
"""""

