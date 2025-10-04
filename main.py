import cv2
import time
from src.real_time_ocr import CasinoOCRProcessor
from src.mlflow_tracking import MLflowTracker
import argparse
import json

def main():
    parser = argparse.ArgumentParser(description='Casino Slot Machine OCR System')
    parser.add_argument('--camera', type=int, default=0, help='Camera device ID')
    parser.add_argument('--machine_id', type=str, required=True, help='Slot Machine ID')
    parser.add_argument('--headless', action='store_true', help='Run without display')
    args = parser.parse_args()
    
    # Initialize components
    ocr_processor = CasinoOCRProcessor()
    mlflow_tracker = MLflowTracker()
    
    # Start MLflow run
    mlflow_tracker.start_training_run(f"realtime_ocr_{args.machine_id}")
    
    # Initialize camera
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    print(f"Starting real-time OCR for machine {args.machine_id}")
    
    try:
        frame_count = 0
        success_count = 0
        start_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture frame")
                break
            
            # Process frame
            result = ocr_processor.process_frame(frame, args.machine_id)
            frame_count += 1
            
            # Update database if successful
            if result['status'] == 'success':
                success_count += 1
                ocr_processor.update_database(result)
            
            # Display results (if not headless)
            if not args.headless:
                display_frame = frame.copy()
                cv2.putText(display_frame, f"Machine: {args.machine_id}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                status_text = f"Status: {result['status']}"
                cv2.putText(display_frame, status_text, (10, 70), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                if result['prize_amount']:
                    prize_text = f"Prize: ${result['prize_amount']:.2f}"
                    cv2.putText(display_frame, prize_text, (10, 110), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                cv2.imshow('Casino OCR System', display_frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            time.sleep(0.1)  # Control processing rate
        
    except KeyboardInterrupt:
        print("Stopping OCR system...")
    
    finally:
        # Calculate performance metrics
        processing_time = time.time() - start_time
        accuracy = success_count / frame_count if frame_count > 0 else 0
        
        # Log to MLflow
        mlflow_tracker.log_ocr_performance(accuracy, processing_time, frame_count)
        
        # Cleanup
        cap.release()
        if not args.headless:
            cv2.destroyAllWindows()
        
        print(f"Processing completed: {frame_count} frames, {success_count} successful detections")

if __name__ == "__main__":
    main()