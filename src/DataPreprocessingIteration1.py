import numpy as np
import os
from FindBallLocation import FindBallLocation
from VideoSoundAnalysis import process_kick_videos


# Using the video sound analysis module, generate a list corresponding to the frame of impact.
# ---------------------------------------------------------------
num_kicks = 20  # Adjust this to the number of kick videos you have
batch_number = 1  # Set your batch number here
output_dir = f"..\\output\\contact_frames_{batch_number}"
os.makedirs(output_dir, exist_ok=True)
contact_frames_array = process_kick_videos(num_kicks, batch_number)
np.save(os.path.join(output_dir, "contact_frames.npy"), contact_frames_array)
# ---------------------------------------------------------------

# Using the Soccer Ball Location module, find the location of the soccer ball
# ---------------------------------------------------------------
for i in range(20):
    # if we can find the location of the ball continue preprocessing.
    FindBallLocation(i)
    # otherwise we can either manually select the location of the ball or pass on the data sample.
# ---------------------------------------------------------------

# find the top of the backswing

# taking a moving average of the frames from the backswing -> impact point.
