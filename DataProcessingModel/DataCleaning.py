#Viết hàm lọc nhiễu
#Viết hàm để loại bỏ vật thể nhỏ
#Viết hàm để cải thiện độ tương phản
#Hàm phát hiện cạnh
import cv2
import numpy as np
import matplotlib.pyplot as plt

def process_image(image_path):
    # Đọc ảnh
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print("Error: Unable to load image.")
        return
    
    # Bước 1: Lọc nhiễu bằng bộ lọc Gaussian
    denoised = cv2.GaussianBlur(image, (5, 5), 0)
    
    # Bước 2: Loại bỏ vật thể nhỏ
    # Áp dụng ngưỡng để tạo ra ảnh nhị phân
    _, binary = cv2.threshold(denoised, 128, 255, cv2.THRESH_BINARY)
    # Áp dụng các phép toán hình thái học
    kernel = np.ones((3,3), np.uint8)
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    
    # Bước 3: Cải thiện độ tương phản
    equalized = cv2.equalizeHist(denoised)
    
    # Bước 4: Phát hiện cạnh bằng Canny
    edges = cv2.Canny(equalized, 100, 200)
    
    # Hiển thị ảnh gốc và các kết quả
    titles = ['Original Image', 'Denoised Image', 'Removed Small Objects', 'Equalized Image', 'Edge Detection']
    images = [image, denoised, sure_bg, equalized, edges]
    
    for i in range(5):
        plt.subplot(2, 3, i+1), plt.imshow(images[i], 'gray')
        plt.title(titles[i])
        plt.xticks([]), plt.yticks([])
    
    plt.show()

# Gọi hàm với đường dẫn tới ảnh
process_image(r"C:\Users\BaoPhuc\Downloads\phuc.jpg")




