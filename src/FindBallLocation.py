import numpy as np
import cv2
import os
import json


def load_contact_frames(batch_number):
    """Load the array representing frames of ball contact."""
    try:
        current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        filename = f"output/Batch {batch_number}/contact_frames_{batch_number}/contact_frames.npy"
        final_path = os.path.join(current_dir, filename)
        return np.load(final_path)
    except FileNotFoundError:
        print(f"Error: {filename} not found.")
        return None


def build_video_path(src_dir, video_number, batch_number):
    """Build the video path based on the video number."""
    return os.path.join(src_dir, f'../dataset/Session {batch_number}/kick {video_number}.mp4')


def build_pose_estimation_path(src_dir, video_number, frame_number, batch_number):
    """Build the path for the pose estimation JSON file based on video and frame number."""
    frame_str = str(frame_number).zfill(2)  # Ensure consistent zero-padding
    if batch_number == 1:
        return os.path.join(src_dir, f'../output/pose_estimation_results_{batch_number}/Kick_{video_number}_0000000000{frame_str}_keypoints.json')
    else:
        return os.path.join(src_dir, f'../output/pose_estimation_results_{batch_number}/Kick {video_number}_0000000000{frame_str}_keypoints.json')


def open_video(video_path, target_frame=0):
    """Open the video and set to the target frame."""
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
    ret, frame = cap.read()
    if not ret:
        print("Could not retrieve the specified frame.")
        cap.release()
    return cap, frame


def load_pose_data(pose_estimation_path):
    """Load pose estimation data from a JSON file."""
    print(pose_estimation_path)
    with open(pose_estimation_path, 'r') as f:
        pose_data = json.load(f)
    return pose_data


def get_foot_position(pose_data, joint_indices):
    """Calculate the average position of the specified foot joints."""
    all_points = pose_data['people'][0]['pose_keypoints_2d']
    x_avg = y_avg = count = 0

    for i in joint_indices:
        if i * 3 + 2 < len(all_points):
            x, y, confidence = all_points[i * 3], all_points[i * 3 + 1], all_points[i * 3 + 2]
            if confidence > 0 and x > 0 and y > 0:
                x_avg += x
                y_avg += y
                count += 1

    if count > 0:
        return x_avg / count, y_avg / count
    return -1, -1  # Return -1, -1 if no valid points were found


def detect_circles(frame):
    """Detect circles using the Hough Circle Transform."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    circles = cv2.HoughCircles(
        gray_blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=20,
        param1=60,
        param2=30,
        minRadius=10,
        maxRadius=50
    )
    return np.round(circles[0, :]).astype("int") if circles is not None else None


def find_closest_ball(circles, left_foot, right_foot):
    """Find the circle closest to either the left or right foot."""
    closest_ball = None
    min_distance = float('inf')

    # foot needs to be this close to the ball for it to count.
    threshold_distance = 100

    for (x, y, r) in circles:
        left_distance = np.sqrt((x - left_foot[0]) ** 2 + (y - left_foot[1]) ** 2) if left_foot[0] != -1 else 1000
        right_distance = np.sqrt((x - right_foot[0]) ** 2 + (y - right_foot[1]) ** 2) if right_foot[0] != -1 else 1000
        distance = min(left_distance, right_distance)
        if distance < min_distance and distance < threshold_distance:
            min_distance = distance
            closest_ball = (x, y, r)

    return closest_ball


def draw_annotations(frame, left_foot, right_foot, closest_ball):
    """Draw circles around the detected feet and the closest ball on the frame."""
    if left_foot[0] != -1:
        cv2.circle(frame, (int(left_foot[0]), int(left_foot[1])), 10, (255, 0, 0), 4)
    if right_foot[0] != -1:
        cv2.circle(frame, (int(right_foot[0]), int(right_foot[1])), 10, (255, 0, 0), 4)
    if closest_ball:
        (x, y, r) = closest_ball
        cv2.circle(frame, (x, y), r, (0, 255, 0), 4)
        cv2.circle(frame, (x, y), 2, (0, 0, 255), 3)
    return frame


def user_find_ball(video_path):
    """Refine ball location by letting the user click on the image."""
    curr_frame = 0
    cap, frame = open_video(video_path, curr_frame)
    if frame is None:
        return None

    coordinates = []

    def click_event(event, x, y, flags, param):
        nonlocal coordinates
        if event == cv2.EVENT_LBUTTONDOWN:
            if coordinates:
                last_x, last_y = coordinates[-1]
                distance = np.sqrt((x - last_x) ** 2 + (y - last_y) ** 2)
                if distance < 50:  # Double-click at the same location
                    avg_x = (last_x + x) // 2
                    avg_y = (last_y + y) // 2
                    print(f"Ball location determined at: ({avg_x}, {avg_y})")
                    coordinates.append((avg_x, avg_y))
                    return
            coordinates.append((x, y))

    cv2.namedWindow("Select Ball Location")
    cv2.setMouseCallback("Select Ball Location", click_event)

    while len(coordinates) < 2:  # Allow up to 2 clicks
        ret, frame = cap.read()
        if not ret:
            print("End of video reached or error loading frame.")
            break

        cv2.imshow("Select Ball Location", frame)
        key = cv2.waitKey(1)
        if key == ord('q'):  # Quit selection
            break

    cap.release()
    cv2.destroyAllWindows()

    # Return the average location if two clicks were made
    if len(coordinates) >= 2:
        avg_x, avg_y = np.mean(coordinates, axis=0).astype(int)
        return avg_x, avg_y
    return -1, -1


def FindBallLocation(video_number, batch_number):
    """Main function to perform ball location analysis on a specified video."""
    print("Finding ball location on video", video_number)
    contact_frames = load_contact_frames(batch_number)
    if contact_frames is None or video_number < 1 or video_number > len(contact_frames):
        print(f"Invalid video number {video_number}.")
        return

    frame_number = contact_frames[video_number - 1]  # Adjust for zero-indexing
    src_dir = os.path.dirname(__file__)
    video_path = build_video_path(src_dir, video_number, batch_number)
    pose_estimation_path = build_pose_estimation_path(src_dir, video_number, frame_number, batch_number)
    print(f"pose estimation path: {pose_estimation_path}")
    print(f"video path: {video_path}")

    # Open video and load pose data
    cap, frame = open_video(video_path, target_frame=2)
    if frame is None:
        return
    pose_data = load_pose_data(pose_estimation_path)

    # Determine foot positions
    left_foot_joints = [11, 22, 23, 24]  # Left ankle, heel, big toe
    right_foot_joints = [14, 19, 20, 21]  # Right ankle, heel, big toe
    left_foot = get_foot_position(pose_data, left_foot_joints)
    right_foot = get_foot_position(pose_data, right_foot_joints)

    # Detect circles and find the closest ball
    circles = detect_circles(frame)
    closest_ball = find_closest_ball(circles, left_foot, right_foot) if circles is not None else None

    ball_found = closest_ball is not None
    manual_coordinates = []

    def click_event(event, x, y, flags, param):
        nonlocal manual_coordinates
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(manual_coordinates) == 1:
                last_x, last_y = manual_coordinates[0]
                distance = np.sqrt((x - last_x) ** 2 + (y - last_y) ** 2)
                if distance < 50:  # Double-click to confirm the new location
                    avg_x = (last_x + x) // 2
                    avg_y = (last_y + y) // 2
                    manual_coordinates.append((avg_x, avg_y))
            else:
                manual_coordinates.append((x, y))

    # Attach mouse callback
    cv2.namedWindow("Detected Soccer Ball")
    cv2.setMouseCallback("Detected Soccer Ball", click_event)
    while True:
        # Annotate the frame
        annotated_frame = draw_annotations(frame.copy(), left_foot, right_foot, closest_ball)
        cv2.imshow('Detected Soccer Ball', annotated_frame)

        key = cv2.waitKey(0)
        if key == ord('q'):  # Skip to the next frame
            break
        elif len(manual_coordinates) == 2:  # Manually set location by double-clicking
            avg_x, avg_y = np.mean(manual_coordinates, axis=0).astype(int)
            closest_ball = (avg_x, avg_y, 1)
            break
    if len(manual_coordinates) > 0:
        closest_ball = (manual_coordinates[-1][0], manual_coordinates[-1][1], 1)
    cv2.destroyAllWindows()
    cap.release()
    return closest_ball


def FindNextBallLocation(frame_number, batch_number, video_number, prev_ball_location):
    """
    Function to determine the location of the ball in a specific frame based on the previous frame's ball location.
    Displays all detected circles with bounding rectangles.
    """
    src_dir = os.path.dirname(__file__)
    video_path = build_video_path(src_dir, video_number, batch_number)

    # Open the video at the target frame
    cap, frame = open_video(video_path, target_frame=frame_number)
    if frame is None:
        print(f"Could not load frame {frame_number} from video {video_number}.")
        return None

    # Detect circles in the frame
    circles = detect_circles(frame)
    if circles is None:
        print("No potential balls detected in the frame.")
        return None

    # Draw bounding rectangles around all detected circles
    for (x, y, r) in circles:
        top_left = (int(x - r), int(y - r))
        bottom_right = (int(x + r), int(y + r))
        cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)  # Green rectangle

    # Display the frame with rectangles
    cv2.imshow("Detected Balls", frame)

    # Wait for any key press
    print("Press any key to continue to the next frame.")
    cv2.waitKey(0)  # Wait indefinitely for a key press

    # Find the circle closest to the previous ball location
    closest_ball = None
    min_distance = float('inf')

    for (x, y, r) in circles:
        distance = np.sqrt((x - prev_ball_location[0]) ** 2 + (y - prev_ball_location[1]) ** 2)
        if distance < min_distance:
            min_distance = distance
            closest_ball = (x, y, r)

    # If no valid ball is found, notify the user
    if closest_ball is None:
        print("Could not find a valid ball close to the previous location.")
    else:
        print(f"Ball found at: {closest_ball[:2]} with radius {closest_ball[2]}.")

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

    return closest_ball