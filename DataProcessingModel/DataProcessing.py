import os
import cv2
from skimage.feature import hog
DataLink="/Data"

def proImange(image_path):
   #Đưa tất cả ảnh về cùng 1 size
   #Đọc ảnh dưới dạng ma trận pixel
   #Chuyển đổi ảnh thành vectow n chiều (sử dụng 1 trong các phương pháp trích xuất đặc trưng HOG, SIFT, LBP )
   #hàm trả về 1 vector n chiều dưới dạng numpy array
   image=cv2.imread(image_path)
   newsize=(300,300)
   image =cv2.resize(image,newsize)
   gray_image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
   features = hog(gray_image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=False)
   return features
def proData(folder_path,label,count):
    #sử dụng proImange để xử lý toàn bộ dataset thành vector n chiều
    #trả về 2 list: mảng các vector n chiều và mảng nhãn của chúng
    image_files = os.listdir(folder_path)
    X=[]
    Y=[]
    c=0
    for image_file in image_files:
      if(c==count):
         break
      image_path = os.path.join(folder_path, image_file)
      image = proImange(image_path)
      X.append(image)
      Y.append(label)
      c=c+1
    return X,Y