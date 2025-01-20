import cv2
import numpy as np
import time
import os

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
    
    # Calculate corner size
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
    """Split corner into rank and suit regions."""
    # Convert to grayscale if not already
    if len(corner_img.shape) == 3:
        corner_gray = cv2.cvtColor(corner_img, cv2.COLOR_BGR2GRAY)
    else:
        corner_gray = corner_img

    # Resize for better recognition but maintain aspect ratio
    scale = 4
    corner_gray = cv2.resize(corner_gray, (0, 0), fx=scale, fy=scale)
        
    # Get dimensions
    height, width = corner_gray.shape
    
    # Split into top (rank) and bottom (suit)
    rank_region = corner_gray[0:int(height*0.5), :]
    suit_region = corner_gray[int(height*0.5):, :]
    
    cv2.imshow('Split Rank', rank_region)
    cv2.imshow('Split Suit', suit_region)
    
    return rank_region, suit_region

def load_templates():
    """Load and store reference templates for ranks and suits."""
    # Define the ranks and suits we want to recognize
    ranks = ['ace', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'jack', 'queen', 'king']
    suits = ['hearts', 'diamonds', 'clubs', 'spades']
    
    # Load templates from a templates directory
    rank_templates = {}
    suit_templates = {}
    
    for rank in ranks:
        template_path = f'templates/output/rank_{rank}.jpg'
        if os.path.exists(template_path):
            template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
            rank_templates[rank] = template
            
    for suit in suits:
        template_path = f'templates/output/suit_{suit}.jpg'
        if os.path.exists(template_path):
            template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
            suit_templates[suit] = template
            
    return rank_templates, suit_templates

def recognize_rank(rank_img, templates):
    """Recognize the rank from the rank region using template matching."""
    best_score = float('-inf')
    best_rank = None
    
    # Simple preprocessing - just binary threshold since our input is already pretty clean
    _, rank_img = cv2.threshold(rank_img, 160, 255, cv2.THRESH_BINARY)
    
    # Ensure consistent size
    rank_img = cv2.resize(rank_img, (50, 70))
    
    # Show preprocessed image for debugging
    cv2.imshow('Preprocessed Rank', rank_img)
    
    # For better edge detection
    rank_img = 255 - rank_img  # Invert to match template (black on white)
    
    for rank, template in templates.items():
        # Preprocess template
        template = cv2.resize(template, (50, 70))
        
        # Use TM_CCOEFF_NORMED as it's good for exact matches
        result = cv2.matchTemplate(rank_img, template, cv2.TM_CCOEFF_NORMED)
        score = result.max()
        print(f"Rank {rank} score: {score:.4f}")
        
        if score > best_score:
            best_score = score
            best_rank = rank
    
    print(f"Best rank match: {best_rank} with score {best_score:.4f}")
    return best_rank if best_score > 0.2 else None  # Lowered threshold for testing

def recognize_suit(suit_img, templates):
    """Recognize the suit from the suit region using template matching."""
    best_score = float('-inf')
    best_suit = None
    
    # Simple preprocessing - just binary threshold
    _, suit_img = cv2.threshold(suit_img, 160, 255, cv2.THRESH_BINARY)
    
    # Ensure consistent size
    suit_img = cv2.resize(suit_img, (40, 40))
    
    # Show preprocessed image for debugging
    cv2.imshow('Preprocessed Suit', suit_img)
    
    # For better edge detection
    suit_img = 255 - suit_img  # Invert to match template (black on white)
    
    for suit, template in templates.items():
        # Preprocess template
        template = cv2.resize(template, (40, 40))
        
        # Use TM_CCOEFF_NORMED
        result = cv2.matchTemplate(suit_img, template, cv2.TM_CCOEFF_NORMED)
        score = result.max()
        print(f"Suit {suit} score: {score:.4f}")
        
        if score > best_score:
            best_score = score
            best_suit = suit
    
    print(f"Best suit match: {best_suit} with score {best_score:.4f}")
    return best_suit if best_score > 0.2 else None  # Lowered threshold for testing

def detect_card(frame, rank_templates, suit_templates):
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
                
                        # Recognize rank and suit
                        rank = recognize_rank(rank_region, rank_templates)
                        suit = recognize_suit(suit_region, suit_templates)

                        # Calculate center of the card
                        M = cv2.moments(contour)
                        if M["m00"] != 0:
                            cx = int(M["m10"] / M["m00"])
                            cy = int(M["m01"] / M["m00"])
                            
                            # Display recognized values
                            if rank or suit:  # Changed from 'and' to 'or' for testing
                                text = f"{rank if rank else '?'}{suit if suit else '?'}"
                                print(f"Displaying text: {text} at ({cx}, {cy})")  # Debug print
                                cv2.putText(frame, text, (cx-20, cy), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, 
                                        (0, 0, 255), 3)
    return frame

def main():
    # Setup camera
    cap = setup_camera()
    if cap is None:
        return

    # Load templates
    rank_templates, suit_templates = load_templates()
    print(f"Loaded {len(rank_templates)} rank templates: {list(rank_templates.keys())}")
    print(f"Loaded {len(suit_templates)} suit templates: {list(suit_templates.keys())}")

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
        processed_frame = detect_card(frame, rank_templates, suit_templates)

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