from collections import defaultdict
from collections import Counter

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import cv2
import numpy as np

from ultralytics import YOLO

#import pytesseract

# Load the YOLOv8 model
model = YOLO('yolov8n.pt')

# Define the codec using VideoWriter_fourcc and create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or use 'XVID'
out = cv2.VideoWriter('traffic3.mp4', fourcc, 30.0, (640, 480))  # adjust the frame size (640, 480) to match your frame size

# Open the video file
video_path = "streetFootage.mp4"
cap = cv2.VideoCapture(video_path)

# Store the track history
track_history = defaultdict(list)

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        results = model.track(frame, persist=True)

        # Get the boxes and track IDs
        if results is not None and results[0] is not None and results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xywh.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist()

            # Get the class IDs
            class_ids = results[0].classes.cpu().tolist()

            # Count the occurences of each class ID
            class_id_counts = Counter(class_ids)

            print(f"Class ID counts: {class_id_counts}")
        else:
            print("No results found")
            continue

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Save output video
        out.write(annotated_frame)

        # Extract the bounding boxes for each of the bus tracked detections
        # for box in boxes:
        #     x, y, w, h = box
        #     x, y, w, h = int(x), int(y), int(w), int(h)  # convert to integers
        #     # Crop the image to the bounding box
        #     cropped_image = frame[y:y+h, x:x+w]
        #     # Use pytesseract to read the text on the bus
        #     bus_number = pytesseract.image_to_string(cropped_image)
        #     # Add the bus number to the annotation
        #     cv2.putText(annotated_frame, bus_number, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)


        # Plot the tracks
        for box, track_id in zip(boxes, track_ids):
            x, y, w, h = box
            track = track_history[track_id]
            track.append((float(x), float(y)))  # x, y center point
            if len(track) > 30:  # retain 90 tracks for 90 frames
                track.pop(0)

            # Draw the tracking lines
            points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=10)


        # Plot the detections
        for r in results:
            im_array = r.plot()  # plot a BGR numpy array of predictions


        # Display the annotated frame
        cv2.imshow("YOLOv8 Tracking", annotated_frame)


        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
out.release()
cv2.destroyAllWindows()