from PIL import Image, ImageDraw
import numpy as np

def create_test_image():
    """Create a simple test image with a person silhouette"""
    
    # Create a 400x600 image (typical portrait)
    width, height = 400, 600
    img = Image.new('RGB', (width, height), color='lightblue')
    draw = ImageDraw.Draw(img)
    
    # Draw a simple person silhouette
    # Head
    head_center = (width//2, 100)
    head_radius = 40
    draw.ellipse([
        head_center[0] - head_radius, head_center[1] - head_radius,
        head_center[0] + head_radius, head_center[1] + head_radius
    ], fill='peachpuff', outline='black')
    
    # Body (shirt area)
    shirt_top = head_center[1] + head_radius + 10
    shirt_bottom = shirt_top + 200
    shirt_left = width//2 - 80
    shirt_right = width//2 + 80
    
    draw.rectangle([
        shirt_left, shirt_top,
        shirt_right, shirt_bottom
    ], fill='red', outline='darkred')
    
    # Arms
    arm_width = 30
    # Left arm
    draw.rectangle([
        shirt_left - arm_width, shirt_top + 20,
        shirt_left, shirt_top + 120
    ], fill='red', outline='darkred')
    
    # Right arm
    draw.rectangle([
        shirt_right, shirt_top + 20,
        shirt_right + arm_width, shirt_top + 120
    ], fill='red', outline='darkred')
    
    # Pants
    pants_top = shirt_bottom
    pants_bottom = height - 50
    pants_left = width//2 - 60
    pants_right = width//2 + 60
    
    draw.rectangle([
        pants_left, pants_top,
        pants_right, pants_bottom
    ], fill='blue', outline='darkblue')
    
    # Legs
    leg_width = 25
    # Left leg
    draw.rectangle([
        pants_left + 10, pants_bottom,
        pants_left + 10 + leg_width, height - 10
    ], fill='blue', outline='darkblue')
    
    # Right leg
    draw.rectangle([
        pants_right - 10 - leg_width, pants_bottom,
        pants_right - 10, height - 10
    ], fill='blue', outline='darkblue')
    
    # Save the image
    img.save('test_person.jpg', 'JPEG', quality=95)
    print("✅ Test image created: test_person.jpg")
    
    return 'test_person.jpg'

if __name__ == "__main__":
    create_test_image()