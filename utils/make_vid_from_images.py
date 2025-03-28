import os
import cv2
import numpy as np
import glob

def make_vid_from_images(image_folder, video_path):
    # Get image files in reverse order
    images = sorted(glob.glob(os.path.join(image_folder, "*")), reverse=True)

    # Read first image to get size
    frame = cv2.imread(images[0])
    height, width, layers = frame.shape

    # Define video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec for .mp4
    video = cv2.VideoWriter(video_path, fourcc, 30, (width, height))  # 30 FPS

    # Add images to video
    for image in images:
        frame = cv2.imread(image)
        video.write(frame)

    video.release()
    print("Video saved:", video_path)
