import numpy as np
import os

input_dir = "../output/contact_foot_traj_1"

# Load and preprocess the data
data = np.load("../output/Processed_keypoints_1.npy")
data = data[:, :1, :, :]  # Only take the contact frame
data = np.reshape(data, (20, 50)).astype(np.float64)  # Ensure data is float64

# Define the path to the contact foot trajectories file
contact_foot_trajs_file = os.path.join(input_dir, "contact_foot_trajs.npy")

# Load the data from the .npy file
loaded_data = np.load(contact_foot_trajs_file, allow_pickle=True).item()  # Use .item() to convert from array to dictionary

# Access the elements in the loaded dictionary
x_fit_array = loaded_data['x_fit']  # It is of shape (20,)
y_fit_array = loaded_data['y_fit']  # It is of shape (20,)
frame_num_array = loaded_data['frame_nums']  # It is of shape (20,)

# Convert to arrays if they are not already
x_fit_array = np.array(x_fit_array)  # Should remain shape (20, 5)
y_fit_array = np.array(y_fit_array)  # Should remain shape (20, 5)
frame_num_array = np.array(frame_num_array)  # Should remain shape (20,)

# Reshape frame_num_array to be (20, 1) so it can concatenate properly
frame_num_array = frame_num_array.reshape(-1, 1)

# Concatenate the x_fit_array, y_fit_array, and frame_num_array with the original data
# We need to stack x_fit and y_fit vertically since each row has more features; first combine x_fit and y_fit
combined_fit = np.concatenate((x_fit_array, y_fit_array), axis=1)  # shape will be (20, 10)

# Finally, we concatenate combined_fit with frame_num_array which has shape (20, 1)
new_data = np.concatenate((data, combined_fit, frame_num_array), axis=1)  # Final shape will be (20, 61)

print("Shape of new data: ", new_data.shape)

output_dir = "../output/contact_foot_traj_1"
np.save(os.path.join(output_dir, "training_input_with_traj.npy"), new_data)