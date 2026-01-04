"""
Simple codes to draw BBoxes on images 
"""

from PIL import Image, ImageDraw

def draw_bbox_on_image(image_path, output_path, bbox_norm):
    """
    Draws a bounding box on an image based on normalized coordinates.
    
    Args:
    - image_path: Path to the input image.
    - output_path: Path to save the output image.
    - bbox_norm: List of [ymin, xmin, ymax, xmax] in normalized coordinates (0 to 1).
    """
    try:
        # Load the image
        img = Image.open(image_path).convert("RGB")
        width, height = img.size
        
        # Unpack normalized coordinates [ymin, xmin, ymax, xmax]
        ymin, xmin, ymax, xmax = bbox_norm
        
        # Convert normalized coordinates to pixel coordinates
        left = xmin * width
        top = ymin * height
        right = xmax * width
        bottom = ymax * height
        
        # Initialize drawing context
        draw = ImageDraw.Draw(img)
        
        # Draw the rectangle
        # outline="red" sets the color, width=3 sets the line thickness
        draw.rectangle([left, top, right, bottom], outline="red", width=3)
        
        # Save or show the result
        img.save(output_path)
        print(f"Image with bbox saved to {output_path}")
        img.show()
        
    except Exception as e:
        print(f"An error occurred: {e}")

# Define the relative coordinates provided [ymin, xmin, ymax, xmax]
bbox_coordinates = [0.03, 0.41, 0.34, 0.82]

# Replace 'giraffe_drawing.jpg' with your actual image filename
input_image_filename = 'dummy.jpg'
output_image_filename = 'datasets/bboxed/dear_on_grass.jpg'

if __name__ == "__main__":
    # Create a dummy image for demonstration if the file doesn't exist
    import os
    if not os.path.exists(input_image_filename):
        print(f"'{input_image_filename}' not found. Please ensure the image file is in the directory.")
    else:
        os.makedirs(os.path.dirname(output_image_filename), exist_ok=True)
        draw_bbox_on_image(input_image_filename, output_image_filename, bbox_coordinates)