import cv2
import numpy as np
import time

def setup_camera():
    # Initialize camera (try different indices if 0 doesn't work)
    cap = cv2.VideoCapture(0)
    
    # Set camera resolution (optional)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        return None
    return cap

def extract_card_corner(frame, contour):
    # Get the bounding rect of the card
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    box = np.intp(box)
    
    # Get width and height of the card
    width = rect[1][0]
    height = rect[1][1]
    
    # Ensure width is smaller than height
    if width > height:
        width, height = height, width
    
    # Calculate corner size (approximately 1/4 of card width)
    corner_width = int(width * 0.2)
    corner_height = int(height * 0.35)  # Maintain aspect ratio
    
    # Get source points (top-left corner of card)
    src_pts = np.float32(box)
    # Sort points to get top-left, top-right, bottom-right, bottom-left
    src_pts = src_pts[np.argsort(src_pts[:, 1])]  # Sort by y
    top_pts = src_pts[:2]
    top_pts = top_pts[np.argsort(top_pts[:, 0])]  # Sort by x
    top_left = top_pts[0]
    top_right = top_pts[1]
    
    # Define destination points for the corner
    dst_pts = np.float32([
        [0, 0],
        [corner_width, 0],
        [corner_width, corner_height],
        [0, corner_height]
    ])
    
    # Calculate source points for just the corner
    card_top_vector = top_right - top_left
    card_top_unit = card_top_vector / np.linalg.norm(card_top_vector)
    corner_right = top_left + card_top_unit * corner_width
    corner_bottom_left = top_left + np.array([0, corner_height])
    corner_bottom_right = corner_right + np.array([0, corner_height])
    
    src_pts = np.float32([
        top_left,
        corner_right,
        corner_bottom_right,
        corner_bottom_left
    ])
    
    # Get transform matrix and apply perspective transform
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    corner = cv2.warpPerspective(frame, M, (corner_width, corner_height))
    
    return corner

def split_rank_suit(corner_img):
    # Convert to grayscale if not already
    if len(corner_img.shape) == 3:
        corner_gray = cv2.cvtColor(corner_img, cv2.COLOR_BGR2GRAY)
    else:
        corner_gray = corner_img

    corner_gray = cv2.resize(corner_gray, (0, 0), fx=4, fy=4)
        
    # Threshold to get clean black and white image
    _, thresh = cv2.threshold(corner_gray, 160, 255, cv2.THRESH_BINARY)
    
    # Get dimensions
    height, width = thresh.shape
    
    # Split into top (rank) and bottom (suit)
    rank_region = thresh[0:int(height/2), :]
    suit_region = thresh[int(height/2):, :]
    
    cv2.imshow('Rank', rank_region)
    cv2.imshow('Suit', suit_region)
    
    return rank_region, suit_region

def detect_card(frame):
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Add slight blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (7,7), 0)
    
    # Threshold the image
    _, thresh = cv2.threshold(blurred, 160, 255, cv2.THRESH_BINARY)
    cv2.imshow('3. Simple Threshold', thresh)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a copy for drawing all contours
    contour_img = frame.copy()
    cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 2)
    cv2.imshow('4. All Contours', contour_img)
    
    # Process each contour
    for contour in contours:
        area = cv2.contourArea(contour)
        # These values assume the card takes up a reasonable portion of the frame
        min_area = 15000    # pixels
        max_area = 120000   # pixels
        
        if min_area < area < max_area:
            peri = cv2.arcLength(contour, True)
            # Made approximation more precise (0.02 instead of 0.04)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            
            if len(approx) == 4:
                # Add aspect ratio check for additional accuracy - standard poker card ratio is approximately 1.56 (89/57)
                rect = cv2.minAreaRect(contour)
                width = min(rect[1][0], rect[1][1])
                height = max(rect[1][0], rect[1][1])
                aspect_ratio = height/width if width > 0 else 0
                
                # Allow some tolerance in aspect ratio (1.4 to 1.7)
                if 1.4 < aspect_ratio < 1.7:
                    cv2.drawContours(frame, [approx], -1, (0, 255, 0), 3)

                    # Extract card corner
                    corner = extract_card_corner(frame, contour)
                    if corner is not None:
                        cv2.imshow('5. Card Corner', corner)
                        rank_region, suit_region = split_rank_suit(corner)

                        # Convert corner to grayscale for recognition
                        corner_gray = cv2.cvtColor(corner, cv2.COLOR_BGR2GRAY)
                        _, corner_thresh = cv2.threshold(corner_gray, 160, 255, cv2.THRESH_BINARY)
                        cv2.imshow('6. Corner Threshold', corner_thresh)
    return frame

def main():
    # Setup camera
    cap = setup_camera()
    if cap is None:
        return

    # FPS calculation variables
    fps_start_time = time.time()
    fps_counter = 0
    fps = 0
    
    print("Camera feed started. Press 'q' to quit.")
    
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Can't receive frame")
            break

        # FPS calculation
        fps_counter += 1
        if time.time() - fps_start_time > 1:
            fps = fps_counter
            fps_counter = 0
            fps_start_time = time.time()
        
        # Add FPS text to frame
        cv2.putText(frame, f"FPS: {fps}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Process frame to detect cards
        processed_frame = detect_card(frame)

        # Display the frame
        cv2.imshow('Card Detection Feed', processed_frame)
        
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release everything when done
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()