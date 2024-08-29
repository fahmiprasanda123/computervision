import cv2
import numpy as np
import pandas as pd
import time


battle_data = []
start_time = None
winner_detected = False

def detect_beyblade(frame):

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=50, param1=50, param2=30, minRadius=10, maxRadius=50)
    
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        return circles
    return []

def analyze_video(video_path):
    global start_time, winner_detected
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Error opening video file")
        return
    
    start_time = time.time()
    
    beyblade_positions = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        
        if not ret:
            break
        
        circles = detect_beyblade(frame)
        
        if len(circles) > 0:
            beyblade_positions.append(circles)
            for (x, y, r) in circles:
                cv2.circle(frame, (x, y), r, (0, 255, 0), 4)
        
        cv2.imshow("Beyblade Battle", frame)
        

        if len(beyblade_positions) > 20:
            prev_positions = beyblade_positions[-20]
            curr_positions = beyblade_positions[-1]
            
            if len(curr_positions) < len(prev_positions) and not winner_detected:
                winner_detected = True
                end_time = time.time()
                duration = end_time - start_time
                winner = "Beyblade 1" if len(curr_positions) == 1 else "Beyblade 2"
                save_battle_data(duration, winner)
                break
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

def save_battle_data(duration, winner):
    global battle_data
    battle_data.append({
        'Duration (s)': duration,
        'Winner': winner,
    })
    df = pd.DataFrame(battle_data)
    df.to_csv("beyblade_battle_data.csv", index=False)
    print("Battle data saved to beyblade_battle_data.csv")


analyze_video("beyblade.mp4")
