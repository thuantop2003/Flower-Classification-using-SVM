import numpy as np
from SVMModel import PredictModel as PM
from SVMModel import TrainingModel as TM
from DataProcessingModel import DataProcessing as DP
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
#gắn đường dẫn 5 folder loại hoa cần phân loại
folder_daisy_path=r"Data/daisy"
folder_rose_path=r"Data/rose"
folder_dandelion_path=r"Data/dandelion"
folder_sunflower_path=r"Data/sunflower"
folder_tulip_path=r"Data/tulip"

#Tạo dataset 5 loại hoa
X_daisy,Y_daisy=DP.proData(folder_daisy_path,1,10)
X_rose,Y_rose=DP.proData(folder_rose_path,3,10)
X_dandelion,Y_dandelion=DP.proData(folder_dandelion_path,2,10)
X_sunflower,Y_sunflower=DP.proData(folder_sunflower_path,4,10)
X_tulio,Y_tulip=DP.proData(folder_rose_path,5,10)


#chia tập train-test theo tỉ lệ 3-7
X_daisy_train,Y_daisy_train,X_daisy_test,Y_daisy_test=TM.devideData(X_daisy,Y_daisy,0.3)
X_rose_train,Y_rose_train,X_rose_test,Y_rose_test=TM.devideData(X_rose,Y_rose,0.3)
X_dandelion_train,Y_dandelion_train,X_dandelion_test,Y_dandelion_test=TM.devideData(X_dandelion,Y_dandelion,0.3)
X_sunflower_train,Y_sunflower_train,X_sunflower_test,Y_sunflower_test=TM.devideData(X_sunflower,Y_sunflower,0.3)
X_tulip_train,Y_tulip_train,X_tulip_test,Y_tulip_test=TM.devideData(X_sunflower,Y_sunflower,0.3)

X_Test=X_daisy_test+X_dandelion_test+X_rose_test+X_sunflower_test+X_tulip_test
Y_Test=Y_daisy_test+Y_dandelion_test+Y_rose_test+Y_sunflower_test+Y_tulip_test
X_Train=X_daisy_train+X_dandelion_train+X_rose_train+X_sunflower_train+X_tulip_train
Y_Train=Y_daisy_train+Y_dandelion_train+Y_rose_train+Y_sunflower_train+Y_tulip_train


model=SVC(kernel='linear',C=0.1)

    #train model
model.fit(X_Train,Y_Train)

y_pred = model.predict(X_Test)
accuracy = accuracy_score(Y_Test, y_pred)
print("Accuracy:", accuracy)

















