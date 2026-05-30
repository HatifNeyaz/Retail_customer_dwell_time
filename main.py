# import cv2
# import numpy as np
# import supervision as sv
# from ultralytics import YOLO
# from collections import defaultdict

# # ==========================================
# # 1. CONFIGURATION & SETUP
# # ==========================================
# VIDEO_PATH = "data/input/aisle.mp4"
# OUTPUT_PATH = "data/output/processed_aisle.mp4"
# MODEL_NAME = "yolov8n.pt" # 'n' stands for nano (fastest). Will download automatically.

# # PASTE YOUR POLYGON COORDINATES HERE
# POLYGON =  np.array([[1895, 803], [1862, 887], [1421, 703], [1202, 832], [997, 746], 
#                      [587, 479], [539, 541], [663, 829], [434, 915], [523, 1065], [284, 1070], 
#                      [86, 617], [96, 446], [84, 312], [139, 236], [72, 21], [136, 19], [239, 260], 
#                      [439, 207], [368, 83], [411, 52], [596, 153], [701, 133], [701, 36], [806, 105], 
#                      [1025, 231], [1633, 548]])
# def main():
#     # ==========================================
#     # 2. INITIALIZE AI MODELS & TOOLS
#     # ==========================================
#     model = YOLO(MODEL_NAME)
#     tracker = sv.ByteTrack() 
    
#     # Set up video ingestion and extraction
#     video_info = sv.VideoInfo.from_video_path(VIDEO_PATH)
#     frames_generator = sv.get_video_frames_generator(VIDEO_PATH)
    
#     # Initialize the spatial zone using our polygon coordinates
#     zone = sv.PolygonZone(polygon=POLYGON)
    
#     # Initialize visual annotators (to draw on the frame)
#     box_annotator = sv.BoxAnnotator()
#     label_annotator = sv.LabelAnnotator(text_scale=0.5)
#     zone_annotator = sv.PolygonZoneAnnotator(zone=zone, color=sv.Color.WHITE)
    
#     # Dictionary to track how many frames a specific ID spends in the zone
#     dwell_time_frames = defaultdict(int)
    
#     # ==========================================
#     # 3. THE VIDEO PROCESSING LOOP
#     # ==========================================
#     # sv.VideoSink writes our modified frames back into a new .mp4 file
#     with sv.VideoSink(OUTPUT_PATH, video_info=video_info) as sink:
        
#         for frame in frames_generator:
            
#             # Step A: Run YOLO inference on the current frame
#             # imgsz=640 is a standard, efficient resolution for processing
#             results = model(frame, imgsz=640, verbose=False)[0]
#             detections = sv.Detections.from_ultralytics(results)
            
#             # Step B: Filter detections to ONLY include humans
#             # In the COCO dataset, Class ID 0 is 'person'
#             detections = detections[detections.class_id == 0]
            
#             # Step C: Pass detections to ByteTrack to assign/update tracking IDs
#             detections = tracker.update_with_detections(detections)
            
#             # Step D: Spatial Logic - Check who is inside the polygon zone
#             # trigger() returns a boolean list (e.g., [True, False] if person 1 is in zone, person 2 is not)
#             is_in_zone = zone.trigger(detections)
            
#             # Step E: Temporal Logic - Calculate Dwell Time
#             labels = []
            
#             # We must ensure tracker_id exists before looping
#             if detections.tracker_id is not None:
#                 for i, tracker_id in enumerate(detections.tracker_id):
#                     # If the person is inside the shelf zone, increase their frame count
#                     if is_in_zone[i]:
#                         dwell_time_frames[tracker_id] += 1
                    
#                     # Convert frames to seconds (frames / frames-per-second)
#                     time_in_seconds = dwell_time_frames[tracker_id] / video_info.fps
                    
#                     # Create the text label that will float above their head
#                     labels.append(f"ID: #{tracker_id} | Dwell: {time_in_seconds:.1f}s")
            
#             # ==========================================
#             # 4. DRAWING & RENDERING
#             # ==========================================
#             annotated_frame = frame.copy()
            
#             # Draw the shelf zone on the floor
#             annotated_frame = zone_annotator.annotate(scene=annotated_frame)
            
#             # Draw the boxes and labels on the people
#             annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=detections)
#             annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
            
#             # Write the final beautifully annotated frame to the output video
#             sink.write_frame(annotated_frame)
            
#             # (Optional) Display the video live while it processes
#             cv2.imshow("Retail Analytics Engine", annotated_frame)
#             if cv2.waitKey(1) & 0xFF == ord("q"):
#                 break
                
#     # Clean up the display windows when finished
#     cv2.destroyAllWindows()
#     print(f"Processing complete. Video saved to {OUTPUT_PATH}")

# if __name__ == "__main__":
#     main()












import cv2
import csv
import numpy as np
import supervision as sv
from ultralytics import YOLO
from collections import defaultdict
from datetime import datetime

# ==========================================
# 1. CONFIGURATION & SETUP
# ==========================================
VIDEO_PATH = "data/input/aisle.mp4"
OUTPUT_VIDEO_PATH = "data/output/processed_aisle.mp4"
OUTPUT_CSV_PATH = "data/output/dwell_times.csv"  # <-- NEW: Path for our data
MODEL_NAME = "yolov8n.pt"

# Your specific shelf coordinates
POLYGON = np.array([[1895, 803], [1862, 887], [1421, 703], [1202, 832], [997, 746], 
                     [587, 479], [539, 541], [663, 829], [434, 915], [523, 1065], [284, 1070], 
                     [86, 617], [96, 446], [84, 312], [139, 236], [72, 21], [136, 19], [239, 260], 
                     [439, 207], [368, 83], [411, 52], [596, 153], [701, 133], [701, 36], [806, 105], 
                     [1025, 231], [1633, 548]])

def main():
    model = YOLO(MODEL_NAME)
    tracker = sv.ByteTrack()
    
    video_info = sv.VideoInfo.from_video_path(VIDEO_PATH)
    frames_generator = sv.get_video_frames_generator(VIDEO_PATH)
    
    zone = sv.PolygonZone(polygon=POLYGON)
    
    box_annotator = sv.BoxAnnotator() # Updated for newest supervision
    label_annotator = sv.LabelAnnotator(text_scale=0.5)
    zone_annotator = sv.PolygonZoneAnnotator(zone=zone, color=sv.Color.WHITE)
    
    dwell_time_frames = defaultdict(int)
    
    print("Processing video and gathering analytics. Please wait...")
    
    # Process the video
    with sv.VideoSink(OUTPUT_VIDEO_PATH, video_info=video_info) as sink:
        for frame in frames_generator:
            results = model(frame, imgsz=640, verbose=False)[0]
            detections = sv.Detections.from_ultralytics(results)
            detections = detections[detections.class_id == 0]
            detections = tracker.update_with_detections(detections)
            
            is_in_zone = zone.trigger(detections)
            labels = []
            
            if detections.tracker_id is not None:
                for i, tracker_id in enumerate(detections.tracker_id):
                    if is_in_zone[i]:
                        dwell_time_frames[tracker_id] += 1
                    
                    time_in_seconds = dwell_time_frames[tracker_id] / video_info.fps
                    labels.append(f"ID: #{tracker_id} | Dwell: {time_in_seconds:.1f}s")
            
            annotated_frame = frame.copy()
            annotated_frame = zone_annotator.annotate(scene=annotated_frame)
            annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=detections)
            annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
            
            sink.write_frame(annotated_frame)
            
            # Optional: Comment these 3 lines out if you don't want the live popup window
            cv2.imshow("Retail Analytics Engine", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cv2.destroyAllWindows()
    
    # ==========================================
    # 4. NEW: DATA EXPORTING LOGIC
    # ==========================================
    # Once the video loop finishes, we save the dictionary to a CSV
    print(f"Video saved to {OUTPUT_VIDEO_PATH}. Generating CSV...")
    
    with open(OUTPUT_CSV_PATH, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write the headers
        writer.writerow(["Timestamp", "Customer_ID", "Dwell_Time_Seconds"])
        
        # Write the data for each tracked customer
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        for customer_id, frames in dwell_time_frames.items():
            # Only save data for customers who actually stepped into the zone
            if frames > 0:
                seconds = round(frames / video_info.fps, 2)
                writer.writerow([current_time, customer_id, seconds])

    print(f"Analytics successfully saved to {OUTPUT_CSV_PATH}")

if __name__ == "__main__":
    main()