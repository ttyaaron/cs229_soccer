# How to Use `contact_foot_trajs.npy`

## Overview
Using the previously extracted information about the `plant_foot_frame` and `contact_frame`, I derived the motion trajectory of the contact foot between these two frames. Specifically, I performed regularization on each frame's pose (using the position of the plant foot as a reference and translating and scaling the individual based on the height of the person in the `plant_foot_frame` to 1). After that, I applied polynomial fitting, resulting in the 5 coefficients for the 4th-degree polynomial of `x(t)` as well as the 5 coefficients for the 4th-degree polynomial of `y(t)`.

The coefficients are stored in `contact_foot_trajs.npy` in the form of three items: `'x_fit'`, `'y_fit'`, and `'frame_nums'`. Taking session 1 as an example, both `x_fit` and `y_fit` contain 20 entries, where each entry is an array of size `[5]`. The `frame_nums` has 20 entries, each of which representing the number of frames from `plant_foot_frame` to `contact_frame`, indicating the duration of the trajectory.

## Data Access
You can retrieve this data using the following code snippet:

```python
# Define the path to the contact foot trajectories file
contact_foot_trajs_file = os.path.join(output_dir, "contact_foot_trajs.npy")

# Load the data from the .npy file
loaded_data = np.load(contact_foot_trajs_file, allow_pickle=True).item()  # Use .item() to convert from array to dictionary

# Access the elements in the loaded dictionary
x_fit_array = loaded_data['x_fit']
y_fit_array = loaded_data['y_fit']
frame_num_array = loaded_data['frame_nums']

# Now you can use these variables in your code
print("Loaded x_fit_array:", x_fit_array)
print("Loaded y_fit_array:", y_fit_array)
print("Loaded frame_num_array:", frame_num_array)