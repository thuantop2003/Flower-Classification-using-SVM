#Viết hàm lọc nhiễu
#Viết hàm để loại bỏ vật thể nhỏ
#Viết hàm để cải thiện độ tương phản
#Hàm phát hiện cạnh
import cv2
import numpy as np

def denoise(image):
    # Lọc nhiễu bằng bộ lọc Gaussian
    denoised = cv2.GaussianBlur(image, (5, 5), 0)
    return denoised

def remove_small_objects(denoised_image):
    if denoised_image is None:
        return None
    
    # Loại bỏ vật thể nhỏ
    _, binary = cv2.threshold(denoised_image, 128, 255, cv2.THRESH_BINARY)
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    return sure_bg

def enhance_contrast(denoised_image):
    if denoised_image is None:
        return None
    
    # Cải thiện độ tương phản
    equalized = cv2.equalizeHist(denoised_image)
    return equalized

def detect_edges(equalized_image):
    if equalized_image is None:
        return None
    
    # Phát hiện cạnh bằng Canny
    edges = cv2.Canny(equalized_image, 100, 200)
    return edges



