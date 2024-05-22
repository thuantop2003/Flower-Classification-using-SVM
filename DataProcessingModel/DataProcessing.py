import os
import cv2
from skimage.feature import hog
DataLink="/Data"

def proImange(image_path):
   #Đưa tất cả ảnh về cùng 1 size
   #Đọc ảnh dưới dạng ma trận pixel
   #Chuyển đổi ảnh thành vectow n chiều (sử dụng 1 trong các phương pháp trích xuất đặc trưng HOG, SIFT, LBP )
   #hàm trả về 1 vector n chiều dưới dạng numpy array

   image=cv2.imread(image_path)       #đọc ảnh từ đường dẫn
   newsize=(300,300)
   image =cv2.resize(image,newsize)    #chỉnh lại size của ảnh
   gray_image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)     #đổi ảnh thành ảnh ko màu
   features = hog(gray_image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=False) # trích xuất đặc trưng theo HOG
   return features
def proData(folder_path,label,count):
    #sử dụng proImange để xử lý toàn bộ dataset thành vector n chiều
    #trả về 2 list có count phần tử : mảng các vector n chiều và mảng nhãn của chúng

    image_files = os.listdir(folder_path) #đọc folders
    X=[]
    Y=[]
    c=0
    for image_file in image_files:
      if(c==count): #khi đủ count phần tử thì dừng lại
         break
      image_path = os.path.join(folder_path, image_file) #tạo đường dẫn ảnh
      image = proImange(image_path) #trích xuất đặc trưng ảnh
      X.append(image) #X thêm ma trận ảnh
      Y.append(label) #Y thêm nhãn ảnh
      c=c+1
    return X,Y