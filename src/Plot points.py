import json
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
import os

ax = None
limits = None
master_array = None


# load in an array representing the frames of ball contact.
filename = 'contact_frames.npy'
try:
    contact_frames = np.load(filename)
    print(f"Successfully loaded {filename}.")
except FileNotFoundError:
    print(f"Error: {filename} not found.")

paused = False
print(contact_frames.shape)

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


def on_click(event):
    global paused
    paused = not paused


def frame_generator(num_frames):
    """Generate frame indices, pausing when the global 'paused' flag is set."""
    i = 0
    while i < num_frames:
        if not paused:  # Only yield when not paused
            yield i
            i += 1
        else:
            yield None  # Yield None to prevent frame progression when paused


def load_keypoints_from_json(json_file):
    """Load pose keypoints from a JSON file."""
    with open(json_file, 'r') as f:
        data = json.load(f)
        if len(data["people"]) == 0:
            return None
        return data["people"][0]["pose_keypoints_2d"]


def adjust_keypoints_to_fixed_foot(pose_keypoints, desired_foot_location, foot_index):
    """
    Adjust all keypoints so the plant foot stays at the desired location.

    :param pose_keypoints: A 2D array of shape (25, 3) representing the [x, y, confidence] of keypoints
    :param desired_foot_location: A list or array representing the [x, y] location for the plant foot
    :param foot_index: The list of indeces of the plant foot in the pose_keypoints array
    :return: A 2D array with adjusted keypoints so the plant foot is fixed at the desired location
    """
    keypoints = np.array(pose_keypoints).reshape(-1, 3)  # Reshape to (25, 3)

    # Average the location of the foot across the different points used
    avg_x = 0
    avg_y = 0
    avg_count = 0
    for i in range(len(foot_index)):
        foot_x, foot_y = keypoints[foot_index[i], 0], keypoints[foot_index[i], 1]
        if not (foot_x == 0 and foot_y == 0):
            avg_count += 1
            avg_x += foot_x
            avg_y += foot_y
    if avg_count == 0:
        return keypoints
    foot_x = avg_x / avg_count
    foot_y = avg_y / avg_count

    # Calculate the translation needed to move the foot to the desired location
    translation_x = desired_foot_location[0] - foot_x
    translation_y = desired_foot_location[1] - foot_y

    # Apply this translation to all the keypoints (only to x and y, not the confidence scores)
    for i in range(len(keypoints)):
        keypoints[i, 0] += translation_x  # Adjust x coordinate
        keypoints[i, 1] += translation_y  # Adjust y coordinate

    return keypoints


def calculate_limits(keypoints_array):
    """Calculate global min/max x and y values for the entire sequence of keypoints."""
    all_x = []
    all_y = []

    # Collect all x and y values across all frames
    for keypoints_list in keypoints_array:
        for keypoints in keypoints_list:
            keypoints = np.array(keypoints).reshape(-1, 3)
            xs, ys = keypoints[:, 0], keypoints[:, 1]
            all_x.extend(xs)
            all_y.extend(ys)

    return min(all_x), max(all_x), min(all_y), max(all_y)


# plot the time-series locations of specific joint.
def plot_joint_trajectories_over_time(master_array, joint_indices, joint_names=None):
    """
    Plot the x and y coordinates of specific joints across time.

    :param master_array: A 3D numpy array of shape (frames, 25, 3), where 25 is the number of joints
                         and each joint has [x, y, confidence].
    :param joint_indices: A list of integers representing the indices of the joints to analyze.
    :param joint_names: Optional list of names for each joint, to use in the plot legend.
                        If None, will default to "Joint {index}".
    """
    num_frames = master_array.shape[0]
    time_steps = range(num_frames)  # Assuming time steps correspond to frame numbers

    if joint_names is None:
        joint_names = [f"Joint {i}" for i in joint_indices]

    # Create subplots for x and y coordinates
    fig, ax = plt.subplots(2, 1, figsize=(10, 8))

    # Plot for each joint index
    for i, joint_index in enumerate(joint_indices):
        joint_x = master_array[:, joint_index, 0]  # X coordinates over time
        joint_y = master_array[:, joint_index, 1]  # Y coordinates over time

        # Plot x and y coordinates for this joint
        ax[0].plot(time_steps, joint_x, label=f"{joint_names[i]} X")
        ax[1].plot(time_steps, joint_y, label=f"{joint_names[i]} Y")

    # Set titles and labels for the plots
    ax[0].set_title("Joint X Coordinates Over Time")
    ax[0].set_xlabel("Frame")
    ax[0].set_ylabel("X Coordinate")

    ax[1].set_title("Joint Y Coordinates Over Time")
    ax[1].set_xlabel("Frame")
    ax[1].set_ylabel("Y Coordinate")

    # Show the legend for each subplot
    ax[0].legend()
    ax[1].legend()

    # Display the plot
    plt.tight_layout()
    plt.show()


def animate(i):
    """Animation function that updates the plot for each frame."""
    plot_keypoints(ax, master_array[i])


# functions to plot both the animation of the pose skeleton and the change of joint locations over time
def plot_keypoints(ax, pose_keypoints):
    """Plot the 2D pose keypoints on the given axis."""
    keypoints = np.array(pose_keypoints).reshape(-1, 3)  # Reshape to (25, 3)

    # Extract x, y, confidence
    xs, ys, confidences = keypoints[:, 0], keypoints[:, 1], keypoints[:, 2]

    ax.clear()  # Clear the axis to redraw each frame
    ax.set_xlim(limits[0] - 100, limits[1] + 100)
    ax.set_ylim(limits[2] - 100, limits[3] + 100)
    ax.scatter(xs, ys, c='red', s=30, label='Keypoints')

    # Plot lines connecting the keypoints based on POSE_CONNECTIONS
    for (start, end) in POSE_CONNECTIONS:
        if confidences[start] > 0 and confidences[end] > 0:  # Only plot if both points are valid
            ax.plot([xs[start], xs[end]], [ys[start], ys[end]], 'b-', linewidth=2)

    # Invert y-axis (because images usually have origin at top-left)
    ax.invert_yaxis()
    ax.axis('off')  # Hide axes for cleaner visualization


def animate_combined(i, master_array, joint_indices, joint_names, fig, ax_pose, ax_time_series_x, ax_time_series_y, lines, frame_counter, skeleton_scale):
    """Animation function that updates the pose plot and the time-series plots for each frame."""
    global paused
    if paused:
        return lines + [frame_counter]
    # Update the pose keypoints plot
    plot_keypoints(ax_pose, master_array[i])

    # Update time-series plot
    for j, joint_index in enumerate(joint_indices):
        joint_x = master_array[:i + 1, joint_index, 0]  # X coordinate up to current frame
        joint_y = master_array[:i + 1, joint_index, 1]  # Y coordinate up to current frame

        lines[j * 2].set_data(range(i + 1), joint_x)  # Update X coordinate line
        lines[j * 2 + 1].set_data(range(i + 1), joint_y)  # Update Y coordinate line

    # Update the frame counter
    frame_counter.set_text(f"Frame: {i}")

    return lines + [frame_counter]


def plot_combined_animation_with_time_series(master_array, joint_indices, joint_names=None, skeleton_scale=2):
    num_frames = master_array.shape[0]

    if joint_names is None:
        joint_names = [f"Joint {i}" for i in joint_indices]

    # Create the figure and axes for the pose plot and time-series plots (side-by-side)
    fig, (ax_pose, ax_time_series_x, ax_time_series_y) = plt.subplots(1, 3, figsize=(15, 6))

    # Initialize time-series lines for joints (X and Y)
    lines = []
    for i, joint_name in enumerate(joint_names):
        line_x, = ax_time_series_x.plot([], [], label=f"{joint_name} X")  # X-coordinate line
        line_y, = ax_time_series_y.plot([], [], label=f"{joint_name} Y")  # Y-coordinate line
        lines.append(line_x)
        lines.append(line_y)

    # Time-series plot settings for X coordinates
    ax_time_series_x.set_xlim(0, num_frames)
    ax_time_series_x.set_ylim(np.min(master_array[:, joint_indices, 0]) - 50,
                              np.max(master_array[:, joint_indices, 0]) + 50)
    ax_time_series_x.set_title("Joint X Coordinates Over Time")
    ax_time_series_x.set_xlabel("Frame")
    ax_time_series_x.set_ylabel("X Coordinate")
    ax_time_series_x.legend()

    # Time-series plot settings for Y coordinates
    ax_time_series_y.set_xlim(0, num_frames)
    ax_time_series_y.set_ylim(np.min(master_array[:, joint_indices, 1]) - 50,
                              np.max(master_array[:, joint_indices, 1]) + 50)
    ax_time_series_y.set_title("Joint Y Coordinates Over Time")
    ax_time_series_y.set_xlabel("Frame")
    ax_time_series_y.set_ylabel("Y Coordinate")
    ax_time_series_y.legend()

    # Frame counter for pose plot
    frame_counter = ax_pose.text(0.5, 0.9, '', transform=ax_pose.transAxes, ha='center', va='center', fontsize=12)

    # Set up the FuncAnimation with a frame generator
    fig.canvas.mpl_connect('button_press_event', on_click)
    anim = FuncAnimation(
        fig, animate_combined, frames=frame_generator(num_frames), interval=500,
        fargs=(master_array, joint_indices, joint_names, fig, ax_pose, ax_time_series_x, ax_time_series_y, lines, frame_counter, skeleton_scale),
        blit=False  # Disable blitting for simplicity
    )

    plt.tight_layout()
    plt.show()


def main_func(kick_number):
    fix_plant_foot = True  # fix the location of the plant foot?
    plant_foot_loc = [200, 500]  # location of the plant foot [x, y]
    i = 0
    first_real_point = 0
    global master_array
    master_array = []  # initialize the array to contain all time-series pose data
    global ax
    global limits
    while True:
        # gather the pose key points for each frame and store into the master array
        try:

            if i < 10:
                numb = str(0) + str(i)
            else:
                numb = str(i)
            src_dir = os.path.dirname(__file__)
            src_dir = os.path.abspath(os.path.join(src_dir, '..')) # go up one level.
            json_file = os.path.join(src_dir, 'output/pose_estimation_results_1/Kick_' + str(
               kick_number) + '_0000000000' + numb + '_keypoints.json')
            # json_file = '..\output\pose_estimation_results_1\Kick_' + str(
            #     kick_number) + '_0000000000' + numb + '_keypoints.json'
            pose_keypoints = load_keypoints_from_json(json_file)
            if pose_keypoints is not None:
                master_array.append(pose_keypoints)
                if first_real_point is None:
                    first_real_point = i
            i += 1
        except Exception as e:
            print(f"Error encountered: {e}")
            break
    print('total size: ' + str(i))

    # if we want to set the location of the plant foot, adjust the location of key points
    if fix_plant_foot:
        for i in range(len(master_array)):
            master_array[i] = adjust_keypoints_to_fixed_foot(master_array[i], plant_foot_loc, [14, 19, 20, 21])
    else:
        # Ensure keypoints are reshaped to (25, 3) even if not adjusting the plant foot
        for i in range(len(master_array)):
            master_array[i] = np.array(master_array[i]).reshape(-1, 3)
    master_array = np.array(master_array)
    print(f"Shape of master array is {master_array.shape}.")
    # find limits to set the axes of the plot
    limits = calculate_limits(master_array)
    # turn into a numpy array
    master_array = np.array(master_array)

    # Unresolved Comment: plot a single frame using plot_keypoints
    if True:
        fig, ax = plt.subplots(figsize=(8, 8))  # Create a figure and axis for plotting
        print('first real point: ' + str(first_real_point))
        print('frame of contact: ' + str(contact_frames[kick_number]))
        adjusted_frame = contact_frames[kick_number] - first_real_point  # Plot the keypoints for the selected frame
        # Check if the adjusted frame is within the bounds of master_array
        if 0 <= adjusted_frame < len(master_array):
            print(f"Plotting frame {adjusted_frame} for kick {kick_number}")
            plot_keypoints(ax, master_array[adjusted_frame])  # Plot the keypoints for the selected frame
            plt.show()  # Show the plot
        else:
            print(f"Adjusted frame {adjusted_frame} is out of bounds for kick {kick_number}. Skipping.")
        if kick_number < 20:
            main_func(kick_number + 1)

    # plot the time-series data of the position of select joints
    if False:
        plot_joint_trajectories_over_time(master_array, [11, 23, 24])
        plt.figure()

    # plot the animation of the pose skeleton over time.
    if False:
        # Set up the figure and axis for animation
        fig, ax = plt.subplots(figsize=(8, 8))
        # Create the animation, updating every 500 milliseconds
        anim = FuncAnimation(fig, animate, frames=len(master_array), interval=1000)
        plt.show()

    # plot the combination of animation + time-series plot of joint locations.
    if False:
        joint_indices = [22, 10, 14]  # Example: Left ankle, left knee, left hip
        joint_names = ["right foot", "right knee", "left ankle"]
        # Call the function to animate and plot time-series
        plot_combined_animation_with_time_series(master_array, joint_indices, joint_names, skeleton_scale=2)


if __name__ == "__main__":
    kick_numb = 10
    main_func(kick_numb)

