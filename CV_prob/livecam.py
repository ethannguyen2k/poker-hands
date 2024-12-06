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

        # Display the frame
        cv2.imshow('Card Detection Feed', frame)
        
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release everything when done
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()