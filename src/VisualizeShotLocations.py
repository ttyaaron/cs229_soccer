import cv2
import numpy as np

# Function to visualize the net and ball hits

def transform_goalposts(left_post, right_post, crossbar):
    """
    Transforms the goalpost locations such that:
    - The left upper corner of the left goalpost is at (0, 0).
    - The right goalpost's x-coordinate is scaled to 1.

    :param left_post: Coordinates of the left post [[x1, y1, x2, y2]].
    :param right_post: Coordinates of the right post [[x1, y1, x2, y2]].
    :param crossbar: Coordinates of the crossbar [[x1, y1, x2, y2]].
    :return: Transformed goalposts, x_shift, y_shift, and scale_factor.
    """
    # Determine translation amounts
    x_shift = left_post[0][0]  # Left post x value
    y_shift = min(left_post[0][1], left_post[0][3])  # Smaller y value of the left post

    # Translate the goalposts
    def translate(line):
        return [[
            line[0] - x_shift, line[1] - y_shift,
            line[2] - x_shift, line[3] - y_shift
        ]]

    left_post_translated = translate(left_post[0])
    right_post_translated = translate(right_post[0])
    crossbar_translated = translate(crossbar[0])

    # Determine scale factor
    scale_factor = 1 / (right_post_translated[0][2] - left_post_translated[0][0])

    # Scale the goalposts
    def scale(line):
        return [[
            line[0] * scale_factor, line[1] * scale_factor,
            line[2] * scale_factor, line[3] * scale_factor
        ]]

    left_post_scaled = scale(left_post_translated[0])
    right_post_scaled = scale(right_post_translated[0])
    crossbar_scaled = scale(crossbar_translated[0])

    return (left_post_scaled, right_post_scaled, crossbar_scaled), x_shift, y_shift, scale_factor

def transform_ball_hits(ball_hits, x_shift, y_shift, scale_factor):
    """
    Adjusts ball hit locations based on the translation and scaling applied to the goalposts.

    :param ball_hits: List of ball hit coordinates [(frame_number, x, y), ...].
    :param x_shift: Translation in the x direction.
    :param y_shift: Translation in the y direction.
    :param scale_factor: Scale factor applied to both x and y directions.
    :return: Transformed ball hit locations [(frame_number, x, y), ...].
    """
    transformed_hits = []
    for ball_hit in ball_hits:
        frame_number, x, y = ball_hit[0]
        x_transformed = (x - x_shift) * scale_factor
        y_transformed = (y - y_shift) * scale_factor
        transformed_hits.append((frame_number, x_transformed, y_transformed))

    return transformed_hits


def visualize_transformed_data(goalposts, ball_locations, image_size=500, padding=100):
    """
    Visualizes the scaled goalposts and ball locations on a resized canvas for better visibility.

    :param goalposts: Transformed goalposts as a tuple (left_post, right_post, crossbar).
                      Each is a list of scaled coordinates [[x1, y1, x2, y2]].
    :param ball_locations: List of transformed ball hit locations [(frame_number, x, y), ...].
    :param image_size: The size of the visualization canvas (width and height in pixels).
    :param padding: The padding to be applied on all sides of the canvas.
    """
    left_post, right_post, crossbar = goalposts

    # Adjust the image size to include padding
    total_size = image_size + 2 * padding

    # Create a blank image with padding
    frame = np.zeros((total_size, total_size, 3), dtype=np.uint8)

    # Map scaled coordinates ([0, 1]) to the visualization grid ([padding, image_size + padding])
    def map_to_image_coords(line):
        return [
            int(line[0] * image_size) + padding,
            int(line[1] * image_size) + padding,
            int(line[2] * image_size) + padding,
            int(line[3] * image_size) + padding
        ]

    # Scale goalpost coordinates to the image grid
    left_post_mapped = map_to_image_coords(left_post[0])
    right_post_mapped = map_to_image_coords(right_post[0])
    crossbar_mapped = map_to_image_coords(crossbar[0])

    # Draw the goalposts
    for line in [left_post_mapped, right_post_mapped, crossbar_mapped]:
        cv2.line(frame, (line[0], line[1]), (line[2], line[3]), (255, 255, 255), 2)

    # Draw the ball hit locations
    for loc in ball_locations:
        frame_number, x, y = loc  # Correctly unpack each tuple
        x_mapped = int(x * image_size) + padding
        y_mapped = int(y * image_size) + padding
        cv2.circle(frame, (x_mapped, y_mapped), 5, (0, 0, 255), -1)

    # Display the visualization
    cv2.imshow("Transformed Net and Ball Hits", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def visualize_net_and_hits(goalposts, ball_locations, original=True):
    left_post, right_post, crossbar = goalposts

    # Create a blank image for visualization
    if original:
        frame = np.zeros((2000, 2000, 3), dtype=np.uint8)
    else:
        frame = np.zeros((100, 100, 3))

    # Draw the net
    for line in [left_post[0], right_post[0], crossbar[0]]:
        cv2.line(frame, (line[0], line[1]), (line[2], line[3]), (255, 255, 255), 2)

    # Loop through all kicks to visualize ball hits
    for ball_hits in ball_locations:
        # Draw the balls
        for (_, x, y) in ball_hits:
            cv2.circle(frame, (int(x), int(y)), 5, (0, 0, 255), -1)

        # Show the visualization
    cv2.imshow(f"Net and Ball Hits - Kick {kick_number}", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage
# Replace with actual batch number, session number, and kick range
session_number = 3
number_kicks = 42

# load the goalposts
goalpost_path = f"/Users/nolanjetter/Documents/GitHub/Soccer ML Project Main/output/Batch {session_number}/GoalPosts/sample_1.npy"
goalposts = np.load(goalpost_path, allow_pickle=True)
left_post, right_post, crossbar = goalposts

# load the kicks
ball_locations = []
for kick_number in range(1, number_kicks+1):
    hits_path = f"/Users/nolanjetter/Documents/GitHub/Soccer ML Project Main/output/Batch {session_number}/Net Hits/Batch 1_Session{session_number}_Kick{kick_number}_net_hits.npy"
    ball_hits = np.load(hits_path, allow_pickle=True)
    ball_locations.append(ball_hits)

visualize_net_and_hits(goalposts, ball_locations, original=True)

# now transform the data and visualize again
(left_post_scaled, right_post_scaled, crossbar_scaled), x_shift, y_shift, scale_factor = transform_goalposts(left_post, right_post, crossbar)
transformed_ball_locations = transform_ball_hits(ball_locations, x_shift, y_shift, scale_factor)
file_path = f"/Users/nolanjetter/Documents/GitHub/Soccer ML Project Main/output/Batch {session_number}/transformed_ball_locations_{session_number}.npy"
np.save(file_path, transformed_ball_locations, allow_pickle=True)
print(len(transformed_ball_locations))


visualize_transformed_data((left_post_scaled, right_post_scaled, crossbar_scaled), transformed_ball_locations)