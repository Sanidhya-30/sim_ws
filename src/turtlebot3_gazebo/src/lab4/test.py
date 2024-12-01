import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

# Initialize variables
current_image_index = 0
min_area = 10000

# Create a Matplotlib figure and axes globally
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))


def process_image(frame, min_area):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    lower1 = np.array([0, 0, 0])  
    upper1 = np.array([255, 255, 15])  
    
    mask = cv2.inRange(hsv, lower1, upper1)
    
    kernel = np.ones((9, 9), np.uint8)
    opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    contours, _ = cv2.findContours(opening, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    objects = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > min_area:  # Filter based on minimum area
            x, y, w, h = cv2.boundingRect(contour)
            cx, cy = x + w // 2, y + h // 2
            objects.append([x, y, w, h, cx, cy, area])
    
    return opening, objects

def update(val):
    global min_area
    min_area = slider.val
    redraw_image()

def next_image(event):
    global current_image_index
    current_image_index = (current_image_index + 1) % len(image_files)
    load_image()

def prev_image(event):
    global current_image_index
    current_image_index = (current_image_index - 1) % len(image_files)
    load_image()

def load_image():
    global image, processed_image, objects
    image_path = os.path.join(folder_path, image_files[current_image_index])
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load image: {image_files[current_image_index]}")
        return
    redraw_image()

def redraw_image():
    global processed_image, objects, ax1, ax2, fig
    processed_image, objects = process_image(image, min_area)
    
    # Clear the axes
    ax1.clear()
    ax2.clear()
    
    # Update original image with detections
    image_copy = image.copy()
    for obj in objects:
        x, y, w, h, cx, cy, area = obj
        cv2.rectangle(image_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.circle(image_copy, (cx, cy), 5, (0, 0, 255), -1)
        ax1.text(x, y - 10, f"Area: {area:.0f}", color='red', fontsize=8, backgroundcolor='white')
    
    original_image_rgb = cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB)
    processed_image_rgb = cv2.cvtColor(processed_image, cv2.COLOR_GRAY2RGB)
    
    ax1.imshow(original_image_rgb)
    ax1.set_title(f"Original Image with Detections (Image {current_image_index + 1}/{len(image_files)})")
    ax1.axis('off')
    
    ax2.imshow(processed_image_rgb, cmap='gray')
    ax2.set_title("Processed Image")
    ax2.axis('off')
    
    fig.canvas.draw_idle()

# Folder and files
folder_path = "/home/lucifer/sim_ws/src/turtlebot3_gazebo/pics"
image_files = [file for file in os.listdir(folder_path) if file.lower().endswith(('.png', '.jpg', '.jpeg'))]

if not image_files:
    print("No images found in the folder.")
    exit()

# Initialize variables
current_image_index = 0
min_area = 10000

# Load the first image
load_image()

# Create a Matplotlib figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# Slider for adjusting minimum area
ax_slider = plt.axes([0.2, 0.02, 0.6, 0.03])  # [left, bottom, width, height]
slider = Slider(ax_slider, "Min Area", 0, 1000000, valinit=min_area, valstep=500)
slider.on_changed(update)

# Buttons for navigation
ax_button_next = plt.axes([0.85, 0.02, 0.1, 0.05])
button_next = Button(ax_button_next, "Next")
button_next.on_clicked(next_image)

ax_button_prev = plt.axes([0.05, 0.02, 0.1, 0.05])
button_prev = Button(ax_button_prev, "Previous")
button_prev.on_clicked(prev_image)

# Initial draw
redraw_image()

# Show the plot
plt.show()
