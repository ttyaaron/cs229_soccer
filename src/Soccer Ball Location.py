import numpy as np
import cv2
import os
import json


# choose the shot number
index = 4


# The contact frame represents the frame at which the soccer player strikes the soccer ball.
contact_frames = np.load("contact_frames.npy")
target_frame = 0 # the target frame will always be the first frame. The target frame allows us to get the position of the soccer ball.

# Set up paths and load data
src_dir = os.path.dirname(__file__)
video_path = os.path.join(src_dir, '../dataset/Session 1/kick ' + str(index) + '.mp4')
# this path needs to dynamically update based on the contact frame. For example, if the contact frame is 30, we need the 30th keypoint from the path.
if contact_frames[index] < 10:
    numb = '0' + str(contact_frames[index])
else:
    numb = str(contact_frames[index])
pose_estimation_path = os.path.join(src_dir, '../output/pose_estimation_results_1/Kick_' + str(index) + '_0000000000' + numb + '_keypoints.json')

# Open the video file and set to the target frame
cap = cv2.VideoCapture(video_path)
cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
ret, frame = cap.read()


def get_foot_position(joint_names):
    all_points = pose_data['people'][0]['pose_keypoints_2d']
    x_avg = 0
    y_avg = 0
    count = 0  # Tracks the actual count of valid points

    for i in joint_names:
        # Ensure the indices are within bounds
        if i * 3 + 2 < len(all_points):
            x = all_points[i * 3]
            y = all_points[i * 3 + 1]
            confidence = all_points[i * 3 + 2]

            # Only include points with valid coordinates and confidence > 0
            if confidence > 0 and x > 0 and y > 0:
                x_avg += x
                y_avg += y
                count += 1

    # Calculate the average if valid points were found
    if count > 0:
        return x_avg / count, y_avg / count
    else:
        return -1, -1  # Return -1, -1 if no valid points were found


if ret:
    # Convert the frame to grayscale and blur it
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_blurred = cv2.GaussianBlur(gray, (9, 9), 2)

    # Detect circles using Hough Circle Transform
    circles = cv2.HoughCircles(
        gray_blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=20,
        param1=100,
        param2=30,
        minRadius=10,
        maxRadius=50
    )

    # Load pose estimation data for the contact frame
    with open(pose_estimation_path, 'r') as f:
        pose_data = json.load(f)

    left_foot_joints = [13, 21, 19]  # Left ankle, heel, big toe
    right_foot_joints = [10, 24, 22]  # Right ankle, heel, big toe
    left_foot = get_foot_position(left_foot_joints)
    right_foot = get_foot_position(right_foot_joints)

    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        closest_ball = None
        min_distance = float('inf')

        # Find the circle closest to either foot
        for (x, y, r) in circles:
            left_distance = 1000
            right_distance = 1000
            if left_foot[0] != -1:
                left_distance = np.sqrt((x - left_foot[0]) ** 2 + (y - left_foot[1]) ** 2)
            if right_foot[0] != -1:
                right_distance = np.sqrt((x - right_foot[0]) ** 2 + (y - right_foot[1]) ** 2)
            distance = min(left_distance, right_distance)
            if distance < min_distance:
                min_distance = distance
                closest_ball = (x, y, r)

        # Display the frame with only the closest ball visualized
        if closest_ball:
            (x, y, r) = closest_ball
            cv2.circle(frame, (x, y), r, (0, 255, 0), 4)  # Draw circle around the closest ball
            cv2.circle(frame, (x, y), 2, (0, 0, 255), 3)  # Center point of the ball
            print(f"Closest ball at (x={x}, y={y}) with radius {r}")

    cv2.imshow('Detected Soccer Ball', frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Could not retrieve the specified frame.")

cap.release()