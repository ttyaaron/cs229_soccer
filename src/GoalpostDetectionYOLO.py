import cv2
from ultralytics import YOLO

def run_yolo_model(model_path, video_path, conf_threshold=0.05):
    """
    Run YOLO model on the first frame of a video and return the bounding box for goalposts
    (rescaled to the original image size) and their confidence.
    """
    # Load YOLOv8 model
    model = YOLO(model_path)

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Cannot open video file.")
        exit()

    # Extract the first frame
    ret, first_frame = cap.read()
    cap.release()

    if not ret:
        print("Error: Unable to read the first frame.")
        exit()

    # Remember the original dimensions
    original_height, original_width = first_frame.shape[:2]

    # Resize the frame to 640x640 for YOLO inference
    frame_resized = cv2.resize(first_frame, (640, 640))
    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)

    # Run the model
    results = model.predict(source=frame_rgb, save=False, show=False, conf=conf_threshold)

    # Extract detections and rescale bounding boxes to the original image size
    detections = []
    if results[0].boxes:
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0]  # Bounding box coordinates (scaled to 640x640)
            conf = box.conf[0].item()  # Confidence score
            cls_id = int(box.cls[0].item())  # Class ID

            # Assuming goalposts have a specific class ID (replace `goalpost_class_id` with actual ID)
            goalpost_class_id = 0  # Replace with the actual goalpost class ID
            if cls_id == goalpost_class_id:
                # Rescale the bounding box to the original image size
                x1 = int(x1 / 640 * original_width)
                y1 = int(y1 / 640 * original_height)
                x2 = int(x2 / 640 * original_width)
                y2 = int(y2 / 640 * original_height)

                # Append detection (bounding box and confidence)
                detections.append(((x1, y1, x2, y2), conf))

    return first_frame, detections


def visualize_predictions(image, detections):
    """
    Visualize the predictions on the original image with bounding boxes and confidence scores.
    """
    for (x1, y1, x2, y2), conf in detections:
        # Draw the bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Add the confidence score
        cv2.putText(
            image,
            f"Conf: {conf:.2f}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )

    # Display the image
    cv2.imshow('Goalpost Predictions', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    model_path = '/Users/nolanjetter/Documents/GitHub/Soccer ML Project Main/goalpost_YOLO/bestmodelweights.pt'
    video_path = '/Users/nolanjetter/Documents/GitHub/Soccer ML Project Main/dataset/Session 1/Kick 1.mp4'

    # Run the YOLO model and get predictions
    original_image, detections = run_yolo_model(model_path, video_path)

    if detections:
        print("Detections:", detections)
        # Visualize the predictions
        visualize_predictions(original_image, detections)
    else:
        print("No goalposts detected.")