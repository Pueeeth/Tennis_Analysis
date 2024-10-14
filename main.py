from utils import (read_video,
                   save_video)
from trackers import PlayerTracker,BallTracker
from court_line_detector import CourtLineDetector
import cv2

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def main():
    #Read Video
    input_video_path = "input_videos/input_video.mp4"
    video_frames = read_video(input_video_path)
    
    #Detecting Players and Ball
    player_tracker = PlayerTracker(model_path = 'yolov8x')
    ball_tracker = BallTracker(model_path='models/yolov5_last.pt')
    player_detections = player_tracker.detect_frames(video_frames,
                                                     read_from_stub=True,
                                                     stub_path="tracker_stubs/player_detections.pkl")
    ball_detections = ball_tracker.detect_frames(video_frames,
                                                     read_from_stub=True,
                                                     stub_path="tracker_stubs/ball_detections.pkl")
    ball_detections = ball_tracker.interpolate_ball_positions(ball_detections)
    
    
    #Court Line Detector model
    court_model_path = "models/keypoints_model.pth"
    court_line_detector = CourtLineDetector(court_model_path)
    court_keypoints = court_line_detector.predict(video_frames[0])
    
    
    #Draw output
    
    ##Draw Player Bounding Boxes
    output_video_frames = player_tracker.draw_bboxes(video_frames,player_detections)
    output_video_frames = ball_tracker.draw_bboxes(video_frames,ball_detections)
    
    ##Draw court KeyPoints
    output_video_frames = court_line_detector.draw_keypoints_on_video(output_video_frames,court_keypoints)
    
    ## Draw frame number on top left corner
    for i, frame in enumerate(output_video_frames):
        cv2.putText(frame, f"Frame: {i}",(10,30),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    
    save_video(output_video_frames, "output_videos/output_video.avi")
    
if __name__=="__main__":
    main()