import cv2
import numpy as np
def delete_green(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Phân đoạn ảnh dựa trên màu sắc xanh lá
    lower_green = np.array([32, 50, 50]) 
    upper_green = np.array([100, 255, 255])  
    mask_green = cv2.inRange(hsv_image, lower_green, upper_green)

# Loại bỏ các chi tiết màu xanh lá khỏi ảnh gốc
    filtered_image_green = cv2.bitwise_and(image, image, mask=~mask_green) 
    return filtered_image_green
def delete_black(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 30]) 
    mask_black = cv2.inRange(hsv_image, lower_black, upper_black)

# Đảo ngược mask để lấy phần không phải màu đen
    mask_not_black = cv2.bitwise_not(mask_black)

# Loại bỏ các chi tiết màu đen khỏi ảnh gốc
    filtered_image_black = cv2.bitwise_and(image, image, mask=mask_not_black)
    return filtered_image_black
def delete_brown(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_brown1 = np.array([0, 10, 10]) 
    upper_brown1 = np.array([20, 255, 255])
    mask_brown1 = cv2.inRange(hsv_image, lower_brown1, upper_brown1)

    lower_brown2 = np.array([160, 10, 10]) 
    upper_brown2 = np.array([180, 255, 255]) 
    mask_brown2 = cv2.inRange(hsv_image, lower_brown2, upper_brown2)

# Kết hợp mask để lấy toàn bộ màu nâu
    mask_brown = mask_brown1 | mask_brown2

    # Đảo ngược mask để lấy phần không phải màu nâu
    mask_not_brown = cv2.bitwise_not(mask_brown)

    # Loại bỏ các chi tiết màu nâu khỏi ảnh gốc
    filtered_image_brown = cv2.bitwise_and(image, image, mask=mask_not_brown)
    return filtered_image_brown
def delete_sky_blue(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Tạo mask để loại bỏ các màu xanh dương
    lower_sky_blue = np.array([90, 50, 50])
    upper_sky_blue = np.array([130, 255, 255])
    mask_sky_blue = cv2.inRange(hsv_image, lower_sky_blue, upper_sky_blue)
    mask_not_sky_blue = cv2.bitwise_not(mask_sky_blue)

    # Loại bỏ các chi tiết màu xanh dương khỏi ảnh gốc
    filtered_image_sky_blue = cv2.bitwise_and(image, image, mask=mask_not_sky_blue)
    return filtered_image_sky_blue
def delete_blue(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Tạo mask để loại bỏ các màu xanh dương
    lower_blue = np.array([100, 50, 50])
    upper_blue = np.array([140, 255, 255])
    mask_blue = cv2.inRange(hsv_image, lower_blue, upper_blue)
    mask_not_blue = cv2.bitwise_not(mask_blue)

    # Loại bỏ các chi tiết màu xanh lam khỏi ảnh gốc
    filtered_image_blue = cv2.bitwise_and(image, image, mask=mask_not_blue)
    return filtered_image_blue
def delete_deepgreen(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Phân đoạn ảnh dựa trên màu sắc xanh lá đậm
    H = int(50 / 360 * 255)
    S = int(50 / 100 * 255)
    V = int(20 / 100 * 255)

    # Định nghĩa giới hạn màu trong không gian HSV
    lower_deepgreen = np.array([H - 30, S - 100, V - 100])
    upper_deepgreen = np.array([H + 20, S + 60, V + 100])
    mask_green = cv2.inRange(hsv_image, lower_deepgreen, upper_deepgreen)

    # Loại bỏ các chi tiết màu xanh lá đậm khỏi ảnh gốc
    filtered_image_deep_green = cv2.bitwise_and(image, image, mask=~mask_green) 
    return filtered_image_deep_green
# loại bỏ các hình ảnh bị nhiễu và nhiễu hạt
def denoise(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Error: Unable to load image {image_path}.")
        return None
    denoised = cv2.GaussianBlur(image, (5, 5), 0)
    return denoised
# Loại bỏ vật thể nhỏ
def remove_small_objects(denoised_image):
    if denoised_image is None:
        return None
    _, binary = cv2.threshold(denoised_image, 128, 255, cv2.THRESH_BINARY)
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    return sure_bg
 # Cải thiện độ tương phản
def enhance_contrast(denoised_image):
    if denoised_image is None:
        return None
    equalized = cv2.equalizeHist(denoised_image)
    return equalized
# Phát hiện cạnh bằng Canny
def detect_edges(equalized_image):
    if equalized_image is None:
        return None
    edges = cv2.Canny(equalized_image, 100, 200)
    return edges
    return edges

