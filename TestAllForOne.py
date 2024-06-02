import numpy as np
from SVMModel import PredictModel as PM
from SVMModel import TrainingModel as TM
from DataProcessingModel import DataProcessing as DP
#gắn đường dẫn 5 folder loại hoa cần phân loại
folder_daisy_path=r"Data/daisy"
folder_rose_path=r"Data/rose"
folder_dandelion_path=r"Data/dandelion"
folder_sunflower_path=r"Data/sunflower"
folder_tulip_path=r"Data/tulip"
folder_all=[folder_daisy_path,folder_rose_path,folder_dandelion_path,folder_sunflower_path,folder_tulip_path]



#2 tập để test cả 5 loại hoa
X_Test_All=[]
Y_Test_All=[]



#ham tao tap du lieu de phan loai  1 bong hoa
def devideData(path,amount,label):
    X1=[]
    Y1=[]
    X2=[]
    Y2=[]
    #Lấy dữ liệu từ các folder và chia theo tỉ lệ 7-3
    for flower_path in folder_all:
        if(path==flower_path):
            X1,Y1=DP.proData(path,1,amount)
        else:
            X,Y=DP.proData(flower_path,-1,amount)
            X2=X2+X
            Y2=Y2+Y
    X1_train,Y1_train,X1_test,Y1_test=TM.devideData(X1,Y1,0.3)
    for i in X1_test:
        X_Test_All.append(i)
    for i in Y1_test:
        Y_Test_All.append(label)
    #Ghép các tập để tạo tập test hoàn chỉnh
    X2_train,Y2_train,X2_test,Y2_test=TM.devideData(X2,Y2,0.3)
    X_train=np.concatenate((X1_train, X2_train), axis=0)
    X_test=np.concatenate((X1_test, X2_test), axis=0)
    Y_train=np.concatenate((Y1_train, Y2_train), axis=0)
    Y_test=np.concatenate((Y1_test, Y2_test), axis=0)
    return X_train,Y_train,X_test, Y_test



#Tạo các tập train và test với từng loại hoa
X_daisy_train,Y_daisy_train,X_daisy_test,Y_daisy_test=devideData(folder_daisy_path,500,1)
X_dandelion_train,Y_dandelion_train,X_dandelion_test,Y_dandelion_test=devideData(folder_dandelion_path,500,2)
X_rose_train,Y_rose_train,X_rose_test,Y_rose_test=devideData(folder_rose_path,500,3)
X_sunflower_train,Y_sunflower_train,X_sunflower_test,Y_sunflower_test=devideData(folder_sunflower_path,500,4)
X_tulip_train,Y_tulip_train,X_tulip_test,Y_tulip_test=devideData(folder_tulip_path,500,5)




#train model tìm w1,b1 để phân loại hoa daisy hay không
w1,b1=TM.trainingSVM(X_daisy_train,Y_daisy_train,10)

#train model tìm w2,b2 để phân loại hoa dandelion hay không
w2,b2=TM.trainingSVM(X_dandelion_train,Y_dandelion_train,10)

#train model tìm w3,b3 để phân loại hoa hồng hay không
w3,b3=TM.trainingSVM(X_rose_train,Y_rose_train,10)

#train model tìm w4,b4 để phân loại hoa hướng dương hay không
w4,b4=TM.trainingSVM(X_sunflower_train,Y_sunflower_train,10)

#train model tìm w5,b5 để phân loại hoa tulip hay không
w5,b5=TM.trainingSVM(X_tulip_train,Y_tulip_train,10)



#Tạo list lưu các giá trị sau khi train
w=[w1,w2,w3,w4,w5]
b=[b1,b2,b3,b4,b5]



#đánh giấ tất cả  w,b dựa trên tập test
#train model tìm w1,b1 để phân loại hoa hồng hay không
print(PM.Evaluate(w1,b1,X_daisy_test,Y_daisy_test))
print(PM.Evaluate(w2,b2,X_dandelion_test,Y_dandelion_test))
print(PM.Evaluate(w3,b3,X_rose_test,Y_rose_test))
print(PM.Evaluate(w4,b4,X_sunflower_test,Y_sunflower_test))
print(PM.Evaluate(w5,b5,X_tulip_test,Y_tulip_test))



#đánh giá phân loại 5 hoa
print(PM.EvaluateAllForOne(w,b,X_Test_All,Y_Test_All))




