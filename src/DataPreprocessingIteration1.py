import numpy as np
import os
from FindBallLocation import FindBallLocation
from VideoSoundAnalysis import process_kick_videos
from FindFirstPhase import find_foot_plant_information

num_kicks = 42  # Adjust this to the number of kick videos you have
batch_number = 3  # Set your batch number here

# ---------------------------------------------------------------
# Using the video sound analysis module, generate a list corresponding to the frame of impact.
# contact_frames_array = process_kick_videos(num_kicks, batch_number)
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
# find which foot is the plant foot and the frame the foot is planted.
# plant_foot_info = []
# for i in range(1, num_kicks):
#     plant_foot_info.append(find_foot_plant_information(i, batch_number))
# np.save(f"/Users/nolanjetter/Documents/GitHub/Soccer ML Project Main/output/Batch {batch_number}/plant_foot.npy",
#         plant_foot_info, allow_pickle=True)
# # ---------------------------------------------------------------


# # ---------------------------------------------------------------
# """""
# Next preprocessing steps:
# 1. gather the pose estimation keypoints corresponding to the frame of plant foot and frame of contact.
# 2. translate the points such that the ball is located at 0,0
# 3. normalize the data such that points lie from (-1) to 1.
# 4. gather labels from the annotations
# 5. organize everything in a numpy array and write to local.
# 6. train some models :)
#     - we can try using both the contact frame and the plant frame, or just the plant frame and see if
#       there is a difference in performance.
# """""
#
