import cv2
import numpy as np
def delete_green(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Phân đoạn ảnh dựa trên màu sắc xanh lá

    lower_green = np.array([32, 50, 50])  # Màu xanh lá thấp nhất trong phạm vi HSV
    upper_green = np.array([100, 255, 255])  # Màu xanh lá cao nhất trong phạm vi HSV
    mask_green = cv2.inRange(hsv_image, lower_green, upper_green)

# Loại bỏ các chi tiết màu xanh lá khỏi ảnh gốc
    filtered_image_green = cv2.bitwise_and(image, image, mask=~mask_green)  # Sử dụng ~ để đảo ngược mask
    return filtered_image_green
def delete_black(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_black = np.array([0, 0, 0])  # Màu đen thấp nhất trong phạm vi HSV
    upper_black = np.array([180, 255, 30])  # Màu đen cao nhất trong phạm vi HSV
    mask_black = cv2.inRange(hsv_image, lower_black, upper_black)

# Đảo ngược mask để lấy phần không phải màu đen
    mask_not_black = cv2.bitwise_not(mask_black)

# Loại bỏ các chi tiết màu đen khỏi ảnh gốc
    filtered_image_black = cv2.bitwise_and(image, image, mask=mask_not_black)
    return filtered_image_black
def delete_brown(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Phân đoạn ảnh dựa trên màu sắc không phải là màu nâu trong không gian màu HSV
    lower_brown1 = np.array([0, 10, 10])  # Màu nâu thấp nhất (màu đỏ) trong phạm vi HSV
    upper_brown1 = np.array([20, 255, 255])  # Màu nâu cao nhất (màu đỏ) trong phạm vi HSV
    mask_brown1 = cv2.inRange(hsv_image, lower_brown1, upper_brown1)

    lower_brown2 = np.array([160, 10, 10])  # Màu nâu thấp nhất (màu xanh) trong phạm vi HSV
    upper_brown2 = np.array([180, 255, 255])  # Màu nâu cao nhất (màu xanh) trong phạm vi HSV
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
    # Tạo mask để loại bỏ các màu xanh lam
    lower_sky_blue = np.array([90, 50, 50])
    upper_sky_blue = np.array([130, 255, 255])
    mask_sky_blue = cv2.inRange(hsv_image, lower_sky_blue, upper_sky_blue)
    mask_not_sky_blue = cv2.bitwise_not(mask_sky_blue)

    # Loại bỏ các chi tiết màu xanh dương khỏi ảnh gốc
    filtered_image_sky_blue = cv2.bitwise_and(image, image, mask=mask_not_sky_blue)
    return filtered_image_sky_blue
def delete_blue(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([100, 50, 50])
    upper_blue = np.array([140, 255, 255])
    mask_blue = cv2.inRange(hsv_image, lower_blue, upper_blue)
    mask_not_blue = cv2.bitwise_not(mask_blue)

    # Loại bỏ các chi tiết màu xanh dương khỏi ảnh gốc
    filtered_image_blue = cv2.bitwise_and(image, image, mask=mask_not_blue)
    return filtered_image_blue
def delete_deepgreen(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Phân đoạn ảnh dựa trên màu sắc xanh lá
    H = int(50 / 360 * 255)
    S = int(50 / 100 * 255)
    V = int(20 / 100 * 255)

    # Định nghĩa giới hạn màu trong không gian HSV
    lower_deepgreen = np.array([H - 30, S - 100, V - 100])
    upper_deepgreen = np.array([H + 20, S + 60, V + 100])
    mask_green = cv2.inRange(hsv_image, lower_deepgreen, upper_deepgreen)

    # Loại bỏ các chi tiết màu xanh lá khỏi ảnh gốc
    filtered_image_deep_green = cv2.bitwise_and(image, image, mask=~mask_green) 
    return filtered_image_deep_green
