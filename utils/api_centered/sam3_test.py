import os
import cv2
import torch
from ultralytics import SAM

def detect_top_objects(image_path, model_path="sam3_b.pt"):
    """
    Detects objects using SAM 3 and visualizes the top highest confidence bounding boxes.
    """
    # 1. Load the SAM 3 model
    # Note: Ensure you have the correct model weight file (e.g., sam3_b.pt, sam3_l.pt)
    model = SAM(model_path)

    # 2. Run Inference
    # Calling the model without specific prompts triggers the "Segment Everything" or 
    # automatic mask generation mode, detecting all discernible objects.
    # SAM 3 can also accept text prompts (e.g., model(source, text="objects")) if specific concepts are needed.
    results = model(image_path)

    # 3. Process Results
    result = results[0]  # Get result for the first (and only) image
    
    # Extract boxes (xyxy format) and confidence scores
    # Note: In SAM, 'confidence' is often the predicted IoU (stability score) of the mask.
    boxes = result.boxes.xyxy.cpu().numpy()
    scores = result.boxes.conf.cpu().numpy()

    # Pair boxes with their scores
    detections = list(zip(boxes, scores))

    # 4. Filter: Sort by confidence (descending) and keep top 5
    detections.sort(key=lambda x: x[1], reverse=True)
    top_detections = detections[:5]

    # 5. Visualize
    # Load image for drawing
    image = cv2.imread(image_path)
    
    print(f"Total objects detected: {len(detections)}")
    print("Top Detections:")

    for i, (box, score) in enumerate(top_detections):
        x1, y1, x2, y2 = map(int, box)
        confidence_text = f"#{i+1} Conf: {score:.2f}"
        
        print(f"  {confidence_text} at [{x1}, {y1}, {x2}, {y2}]")
        
        # Draw rectangle
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Put label
        cv2.putText(image, confidence_text, (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Save or Display the output
    output_path = "datasets/bboxed/sam3_top_output.jpg"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, image)
    print(f"\nSaved visualization to {output_path}")

# Example Usage
if __name__ == "__main__":
    # Replace 'dummy.jpg' with your image path
    detect_top_objects("dummy.jpg", "sam3.pt")