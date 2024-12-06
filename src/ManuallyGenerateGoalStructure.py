import cv2
import numpy as np


def process_goal_structure(left_post, right_post, crossbar):
    """
    Processes the goal structure by creating adjusted vertical lines (goalposts) and a connected crossbar.

    :param left_post: Line representing the left vertical post (in the format [[x1, y1, x2, y2]]).
    :param right_post: Line representing the right vertical post (in the format [[x1, y1, x2, y2]]).
    :param crossbar: Line representing the crossbar (in the format [[x1, y1, x2, y2]]).
    :return: Processed left_post, right_post, and crossbar lines in the same format.
    """

    # Determine the intersection points of the horizontal line (crossbar) with each vertical line
    x1_left, y1_left, x2_left, y2_left = left_post[0]
    x1_right, y1_right, x2_right, y2_right = right_post[0]
    x1_cross, y1_cross, x2_cross, y2_cross = crossbar[0]

    # The intersection points of the crossbar with each vertical line
    intersection_left = (int(x1_left), int(y1_cross)) if y1_left < y2_left else (int(x2_left), int(y1_cross))
    intersection_right = (int(x1_right), int(y1_cross)) if y1_right < y2_right else (int(x2_right), int(y1_cross))

    # Create a new crossbar line between the intersection points
    processed_crossbar = [[intersection_left[0], intersection_left[1], intersection_right[0], intersection_right[1]]]

    # Calculate the length of the post between the two intersections
    post_length = abs(intersection_right[0] - intersection_left[0])

    # Calculate the length of each vertical segment as one-third of the post length
    segment_length = int(post_length / 3)

    # Draw vertical lines extending downward from each intersection point by one-third of the post length
    processed_left_post = [[
        intersection_left[0], intersection_left[1],
        intersection_left[0], intersection_left[1] + segment_length
    ]]

    processed_right_post = [[
        intersection_right[0], intersection_right[1],
        intersection_right[0], intersection_right[1] + segment_length
    ]]

    return processed_left_post, processed_right_post, processed_crossbar


# Function to handle mouse clicks
def click_and_draw(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        param['points'].append((x, y))
        if len(param['points']) % 2 == 0:
            cv2.line(param['frame'], param['points'][-2], param['points'][-1], (0, 255, 0), 2)
            cv2.imshow("Video", param['frame'])

# Main function to capture clicks and draw goal structure
def capture_goal_structure(video_directory, session_number, kick_number, number_of_kicks):
    video = cv2.VideoCapture(video_directory)

    if not video.isOpened():
        print(f"Error: Cannot open video {video_directory}")
        return

    ret, frame = video.read()
    if not ret:
        print("Error: Cannot read video frame.")
        return

    param = {'points': [], 'frame': frame.copy()}
    cv2.imshow("Video", frame)
    cv2.setMouseCallback("Video", click_and_draw, param)

    while True:
        cv2.imshow("Video", param['frame'])
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or len(param['points']) == 6:  # ESC key or 6 clicks
            break

    cv2.destroyAllWindows()

    if len(param['points']) != 6:
        print("Error: 6 points were not selected.")
        return

    # Parse the points into goal structure
    left_post = [[param['points'][0][0], param['points'][0][1], param['points'][1][0], param['points'][1][1]]]
    right_post = [[param['points'][2][0], param['points'][2][1], param['points'][3][0], param['points'][3][1]]]
    crossbar = [[param['points'][4][0], param['points'][4][1], param['points'][5][0], param['points'][5][1]]]

    # Process the goal structure
    processed_left_post, processed_right_post, processed_crossbar = process_goal_structure(left_post, right_post, crossbar)

    # Save the processed structure
    for i in range(1, number_of_kicks+1):
        directory = f"/Users/nolanjetter/Documents/GitHub/Soccer ML Project Main/output/Batch {session_number}/GoalPosts/sample_{i}"
        np.save(directory, (processed_left_post, processed_right_post, processed_crossbar), allow_pickle=True)

    # Visualize the result
    for line in [processed_left_post[0], processed_right_post[0], processed_crossbar[0]]:
        cv2.line(frame, (line[0], line[1]), (line[2], line[3]), (255, 0, 0), 2)

    cv2.imshow("Goal Structure Detected", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    video.release()

# Example call
# Replace session_number and kick_number with actual values
session_number = 2
kick_number = 1
number_of_kicks = 28
video_directory = f"/Users/nolanjetter/Documents/GitHub/cs229_soccer/dataset/Session {session_number}/Kick {kick_number}.mp4"
capture_goal_structure(video_directory, session_number, kick_number, number_of_kicks)
