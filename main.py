import numpy as np
import cv2

# Capturing video through webcam
webcam = cv2.VideoCapture(0)

# Start a while loop
while True:
    # Reading the video from the webcam in image frames
    _, imageFrame = webcam.read()

    # Convert the imageFrame in BGR to HSV color space
    hsvFrame = cv2.cvtColor(imageFrame, cv2.COLOR_BGR2HSV)

    # Define color ranges in HSV and their corresponding BGR values
    colors = {
        "Red": {"range": [(136, 87, 111), (180, 255, 255)], "bgr": (0, 0, 255)},
        "Green": {"range": [(25, 52, 72), (102, 255, 255)], "bgr": (0, 255, 0)},
        "Blue": {"range": [(94, 80, 2), (120, 255, 255)], "bgr": (255, 0, 0)},
        "Yellow": {"range": [(22, 93, 0), (45, 255, 255)], "bgr": (0, 255, 255)},
        "Orange": {"range": [(10, 100, 20), (25, 255, 255)], "bgr": (0, 165, 255)},
        "Purple": {"range": [(129, 50, 70), (158, 255, 255)], "bgr": (128, 0, 128)},
        "Black": {"range": [(0, 0, 0), (180, 255, 50)], "bgr": (0, 0, 0)},
        # "White": {"range": [(0, 0, 200), (180, 20, 255)], "bgr": (255, 255, 255)},
    }

    # Morphological kernel
    kernel = np.ones((5, 5), "uint8")

    # Function to detect and label colors
    def detect_and_label(mask, color_name, color_code, min_area=300, max_area=5000):
        """
        Detect and label contours for a given mask, filtered by area.
        """
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            if min_area < area < max_area:
                # Get bounding box coordinates
                x, y, w, h = cv2.boundingRect(contour)
                # Draw rectangle around the object
                cv2.rectangle(imageFrame, (x, y), (x + w, y + h), color_code, 2)
                # Label the object with its color name
                cv2.putText(imageFrame, color_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color_code, 2)

    # Loop through colors, apply mask, and detect labels
    for color_name, color_info in colors.items():
        lower_bound = np.array(color_info["range"][0], np.uint8)
        upper_bound = np.array(color_info["range"][1], np.uint8)
        color_mask = cv2.inRange(hsvFrame, lower_bound, upper_bound)
        color_mask = cv2.dilate(color_mask, kernel)  # Apply dilation
        detect_and_label(color_mask, color_name, color_info["bgr"])

    # Show the resulting frame with detected colors
    cv2.imshow("Multiple Color Detection in Real-Time", imageFrame)
    
    if cv2.waitKey(10) & 0xFF == ord('q'):
        webcam.release()
        cv2.destroyAllWindows()
        break
