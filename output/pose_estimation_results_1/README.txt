These .json files are posture recognition results of the kicks generated from OpenPose.

To understand these data, please search for "Pose25". Each point comes in triples (x, y, c), with x, y being the x, y coordinates of the joint point, c being the confidence. You can visualize the results using the coarse program plot_keypoints.py by entering command:

python plot_keypoints.py path/to/Kick_1_000000000048_keypoints.json

Since I was using CPU, applying the model to each of the videos takes about 1,500 seconds (25 mins). It is time-consuming, so using cloud computing resources or discarding the blank frames in the videos helps.