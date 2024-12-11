import os
import numpy as np
from PIL import Image


def preprocess_player_images(batch_number, output_path):
    """
    Preprocesses player images for a given batch, considering only 10 frames (9 preceding + frame of contact)
    based on the contact_frames.npy file. Pads images to 200x200 and saves as a numpy array.

    Parameters:
    - batch_number (int): The batch number to process.
    - output_path (str): Path to save the master list numpy array.

    Output:
    - Saves a numpy file containing the master list of preprocessed images.
    """
    # Load the contact frames for the batch
    contact_frames_path = f"/Users/nolanjetter/Documents/GitHub/Soccer ML Project Main/output/Batch {batch_number}/contact_frames_{batch_number}/contact_frames.npy"
    contact_frames = np.load(contact_frames_path, allow_pickle=True)

    # Directory path for the batch
    batch_dir = f"/Users/nolanjetter/Documents/GitHub/Soccer ML Project Main/output/Batch {batch_number}/Player Images"

    # Initialize master list
    master_list = []

    # Loop through each sample (assuming 20 samples in total)
    sample_num = 1
    continue_loop = True
    while True:
        try:
            sample_frames = []

            # Get the frame of contact for this sample
            contact_frame = contact_frames[sample_num - 1]  # Adjust for 0-based index

            # Process the 10 frames (9 before + contact frame)
            for frame_num in range(contact_frame - 9, contact_frame + 1):
                frame_path = os.path.join(batch_dir, f"sample_{sample_num}_frame_{frame_num}.png")

                # Check if the frame exists
                if not os.path.exists(frame_path):
                    print(f"Frame {frame_path} does not exist. Skipping...")
                    continue

                # Open the image
                image = Image.open(frame_path)

                # Resize image to 200x200 with padding
                padded_image = Image.new("RGB", (200, 200), (0, 0, 0))  # Black padding
                paste_x = (200 - image.width) // 2
                paste_y = (200 - image.height) // 2
                padded_image.paste(image, (paste_x, paste_y))

                # Convert to grayscale and numpy array
                grayscale_image = padded_image.convert("L")  # Convert to grayscale
                sample_frames.append(np.array(grayscale_image))  # Add frame to sample list

            # If frames were found, add the sample to the master list
            if len(sample_frames) == 10:  # Ensure all 10 frames are present
                master_list.append(np.array(sample_frames))  # Shape: (10, 200, 200)
            else:
                print(f"Sample {sample_num} does not have 10 valid frames. Skipping...")
        except:
            continue_loop = False

    # Save the master list as a numpy file
    np.save(output_path, master_list, allow_pickle=True)

    print(f"Preprocessed images saved to {output_path}")

# Example usage:
batch_number = 1
preprocess_player_images(batch_number=batch_number, output_path=f"/Users/nolanjetter/Documents/GitHub/Soccer ML Project Main/output/master_list_{batch_number}.npy")