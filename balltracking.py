import cv2
import numpy as np

# Define the HSV color range for a red ball (red has two hue ranges)
LOWER_COLOR1 = np.array([0, 120, 70])   # Lower bound for red
UPPER_COLOR1 = np.array([10, 255, 255]) # Upper bound for red

LOWER_COLOR2 = np.array([170, 120, 70])  # Second range for red
UPPER_COLOR2 = np.array([180, 255, 255]) # Second range for red

# Capture video
cap = cv2.VideoCapture(0)

# Create a mask for drawing the trail
trail_mask = None
prev_center = None

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Flip frame for better control
    frame = cv2.flip(frame, 1)
    
    # Initialize the mask
    if trail_mask is None:
        trail_mask = np.zeros_like(frame)
    
    # Convert frame to HSV and create mask for the ball
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Create masks for both red color ranges
    mask1 = cv2.inRange(hsv, LOWER_COLOR1, UPPER_COLOR1)
    mask2 = cv2.inRange(hsv, LOWER_COLOR2, UPPER_COLOR2)
    mask = cv2.bitwise_or(mask1, mask2)
    
    # Find contours of the ball
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest_contour) > 500:
            (x, y), radius = cv2.minEnclosingCircle(largest_contour)
            center = (int(x), int(y))
            
            # Draw the tracking line
            if prev_center is not None:
                cv2.line(trail_mask, prev_center, center, (0, 0, 255), 5)  # Red trail
            prev_center = center
            
            # Draw the ball
            cv2.circle(frame, center, int(radius), (0, 0, 255), 2)  # Red ball outline
    
    # Combine the frame with the trail
    combined = cv2.add(frame, trail_mask)
    
    # Show the output
    cv2.imshow("Ball Tracking", combined)
    
    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
