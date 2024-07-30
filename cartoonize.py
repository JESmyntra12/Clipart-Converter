import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from PIL import Image, ImageTk

def cartoonize_image(image):
    # Convert to gray scale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 7)

    # Detect edges
    edges = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)

    # Apply bilateral filter to smooth the image while preserving edges
    color = cv2.bilateralFilter(image, 9, 75, 75)  # Moderate parameters

    # Combine edges and color
    cartoon = cv2.bitwise_and(color, color, mask=edges)

    # Further simplify colors to match the cartoon style
    Z = cartoon.reshape((-1, 3))
    Z = np.float32(Z)

    # Define criteria and apply KMeans
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    K = 10  # Moderate number of clusters
    _, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((cartoon.shape))

    return res2

def show_images(original_image, cartoon_image):
    if original_image is not None and cartoon_image is not None:
        original_img = Image.fromarray(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
        cartoon_img = Image.fromarray(cv2.cvtColor(cartoon_image, cv2.COLOR_BGR2RGB))
        
        original_img_tk = ImageTk.PhotoImage(image=original_img)
        cartoon_img_tk = ImageTk.PhotoImage(image=cartoon_img)
        
        panel_original.configure(image=original_img_tk)
        panel_original.image = original_img_tk
        
        panel_cartoon.configure(image=cartoon_img_tk)
        panel_cartoon.image = cartoon_img_tk
    else:
        print("Error: No images to display.")

def choose_from_gallery():
    file_path = filedialog.askopenfilename(
        filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
    if file_path:
        global original_image, cartoon_image
        original_image = cv2.imread(file_path)
        cartoon_image = cartoonize_image(original_image)
        show_images(original_image, cartoon_image)

def choose_from_camera():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()
    if ret:
        global original_image, cartoon_image
        original_image = frame
        cartoon_image = cartoonize_image(original_image)
        show_images(original_image, cartoon_image)
    else:
        print("Error: Could not access the camera.")

def save_image():
    if cartoon_image is not None:
        save_path = filedialog.asksaveasfilename(
            defaultextension=".jpg", filetypes=[("JPEG files", "*.jpg"), ("PNG files", "*.png")])
        if save_path:
            cv2.imwrite(save_path, cartoon_image)
    else:
        messagebox.showerror("Error", "No image to save!")

# Initialize Tkinter window
root = tk.Tk()
root.title("Cartoonizer")
root.geometry("1200x600")

original_image = None
cartoon_image = None

# Buttons
btn_gallery = tk.Button(root, text="Choose from Gallery", command=choose_from_gallery)
btn_gallery.pack(side=tk.LEFT, padx=10, pady=10)

btn_camera = tk.Button(root, text="Choose from Camera", command=choose_from_camera)
btn_camera.pack(side=tk.LEFT, padx=10, pady=10)

btn_save = tk.Button(root, text="Save Cartoon Image", command=save_image)
btn_save.pack(side=tk.LEFT, padx=10, pady=10)

# Image display panels
frame = tk.Frame(root)
frame.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

panel_original = tk.Label(frame)
panel_original.pack(side=tk.LEFT, padx=10, pady=10, fill=tk.BOTH, expand=True)

panel_cartoon = tk.Label(frame)
panel_cartoon.pack(side=tk.RIGHT, padx=10, pady=10, fill=tk.BOTH, expand=True)

root.mainloop()
