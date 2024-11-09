import numpy as np
import os
from FindBallLocation import FindBallLocation
from VideoSoundAnalysis import process_kick_videos
from FindFirstPhase import find_foot_plant_information

# ---------------------------------------------------------------
# Using the video sound analysis module, generate a list corresponding to the frame of impact.
num_kicks = 20  # Adjust this to the number of kick videos you have
batch_number = 1  # Set your batch number here
output_dir = f"..\\output\\contact_frames_{batch_number}"
os.makedirs(output_dir, exist_ok=True)
contact_frames_array = process_kick_videos(num_kicks, batch_number)
np.save(os.path.join(output_dir, "contact_frames.npy"), contact_frames_array)
# ---------------------------------------------------------------


# ---------------------------------------------------------------
# Using the Soccer Ball Location module, find the location of the soccer ball
ball_location = []
for i in range(20):
    ball_location.append(FindBallLocation(i))
# ---------------------------------------------------------------

# ---------------------------------------------------------------
# find which foot is the plant foot and the frame the foot is planted.
plant_foot_info = []
for i in range(20):
    plant_foot_info.append(find_foot_plant_information(i))
# ---------------------------------------------------------------


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

