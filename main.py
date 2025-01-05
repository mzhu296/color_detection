import numpy as np
import cv2

# Capturing video through webcam
webcam = cv2.VideoCapture(0)

# Start a while loop
while(1):
    
    # Reading the video from the webcam in image frames
    _, imageFrame = webcam.read()

    # Convert the imageFrame in BGR to HSV color space
    hsvFrame = cv2.cvtColor(imageFrame, cv2.COLOR_BGR2HSV)

    # Set range for red color and define mask
    red_lower = np.array([136, 87, 111], np.uint8)
    red_upper = np.array([180, 255, 255], np.uint8)
    red_mask = cv2.inRange(hsvFrame, red_lower, red_upper)

    # Set range for green color and define mask
    green_lower = np.array([25, 52, 72], np.uint8)
    green_upper = np.array([102, 255, 255], np.uint8)
    green_mask = cv2.inRange(hsvFrame, green_lower, green_upper)

    # Set range for blue color and define mask
    blue_lower = np.array([94, 80, 2], np.uint8)
    blue_upper = np.array([120, 255, 255], np.uint8)
    blue_mask = cv2.inRange(hsvFrame, blue_lower, blue_upper)

    # Set range for yellow color and define mask
    yellow_lower = np.array([22, 93, 0], np.uint8)
    yellow_upper = np.array([45, 255, 255], np.uint8)
    yellow_mask = cv2.inRange(hsvFrame, yellow_lower, yellow_upper)

    # Set range for orange color and define mask
    orange_lower = np.array([10, 100, 20], np.uint8)
    orange_upper = np.array([25, 255, 255], np.uint8)
    orange_mask = cv2.inRange(hsvFrame, orange_lower, orange_upper)

    # Set range for purple color and define mask
    purple_lower = np.array([129, 50, 70], np.uint8)
    purple_upper = np.array([158, 255, 255], np.uint8)
    purple_mask = cv2.inRange(hsvFrame, purple_lower, purple_upper)

    # Set range for black color and define mask
    black_lower = np.array([0, 0, 0], np.uint8)
    black_upper = np.array([180, 255, 50], np.uint8)
    black_mask = cv2.inRange(hsvFrame, black_lower, black_upper)

    # Set range for white color and define mask
    white_lower = np.array([0, 0, 200], np.uint8)
    white_upper = np.array([180, 20, 255], np.uint8)
    white_mask = cv2.inRange(hsvFrame, white_lower, white_upper)

    # Morphological Transform, Dilation for each color and bitwise_and operator
    kernel = np.ones((5, 5), "uint8")

    # For red color
    red_mask = cv2.dilate(red_mask, kernel)
    res_red = cv2.bitwise_and(imageFrame, imageFrame, mask=red_mask)

    # For green color
    green_mask = cv2.dilate(green_mask, kernel)
    res_green = cv2.bitwise_and(imageFrame, imageFrame, mask=green_mask)

    # For blue color
    blue_mask = cv2.dilate(blue_mask, kernel)
    res_blue = cv2.bitwise_and(imageFrame, imageFrame, mask=blue_mask)

    # For yellow color
    yellow_mask = cv2.dilate(yellow_mask, kernel)
    res_yellow = cv2.bitwise_and(imageFrame, imageFrame, mask=yellow_mask)

    # For orange color
    orange_mask = cv2.dilate(orange_mask, kernel)
    res_orange = cv2.bitwise_and(imageFrame, imageFrame, mask=orange_mask)

    # For purple color
    purple_mask = cv2.dilate(purple_mask, kernel)
    res_purple = cv2.bitwise_and(imageFrame, imageFrame, mask=purple_mask)

    # For black color
    black_mask = cv2.dilate(black_mask, kernel)
    res_black = cv2.bitwise_and(imageFrame, imageFrame, mask=black_mask)

    # For white color
    white_mask = cv2.dilate(white_mask, kernel)
    res_white = cv2.bitwise_and(imageFrame, imageFrame, mask=white_mask)

    # Function to detect and label colors
    def detect_and_label(mask, color_name, color_code):
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for pic, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if area > 300:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(imageFrame, (x, y), (x + w, y + h), color_code, 2)
                cv2.putText(imageFrame, color_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color_code, 2)

    # Detect and label colors
    detect_and_label(red_mask, "Red Colour", (0, 0, 255))
    detect_and_label(green_mask, "Green Colour", (0, 255, 0))
    detect_and_label(blue_mask, "Blue Colour", (255, 0, 0))
    detect_and_label(yellow_mask, "Yellow Colour", (0, 255, 255))
    detect_and_label(orange_mask, "Orange Colour", (0, 165, 255))
    detect_and_label(purple_mask, "Purple Colour", (128, 0, 128))
    detect_and_label(black_mask, "Black Colour", (0, 0, 0))
    detect_and_label(white_mask, "White Colour", (255, 255, 255))

    # Program Termination
    cv2.imshow("Multiple Color Detection in Real-Time", imageFrame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        webcam.release()
        cv2.destroyAllWindows()
        break
