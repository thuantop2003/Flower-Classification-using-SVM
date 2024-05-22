import numpy as np
from SVMModel import PredictModel as PM
from SVMModel import TrainingModel as TM
from DataProcessingModel import DataProcessing as DP
#gắn đường dẫn 2 folder loại hoa cần phân loại
folder_daisy_path=r"Data/daisy"
folder_rose_path=r"Data/rose"

#Tạo dataset 2 loại hoa
X_daisy,Y_daisy=DP.proData(folder_daisy_path,1,100)
X_rose,Y_rose=DP.proData(folder_rose_path,-1,100)

#chia tập train-test theo tỉ lệ 3-7
X_daisy_train,Y_daisy_train,X_daisy_test,Y_daisy_test=TM.devideData(X_daisy,Y_daisy,0.3)
X_rose_train,Y_rose_train,X_rose_test,Y_rose_test=TM.devideData(X_rose,Y_rose,0.3)

#ghép 2 tập để tạo 2 tập train và test hoàn chỉnh
X_train=np.concatenate((X_rose_train, X_daisy_train), axis=0)
X_test=np.concatenate((X_rose_test, X_daisy_test), axis=0)
Y_train=np.concatenate((Y_rose_train, Y_daisy_train), axis=0)
Y_test=np.concatenate((Y_rose_test, Y_daisy_test), axis=0)

#train model tìm w,b
w,b=TM.trainingSVM(X_train,Y_train,0.1)

#đánh giấ w,b dựa trên tập test
print(PM.Evaluate(w,b,X_test,Y_test))