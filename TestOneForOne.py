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

#Tạo dataset 5 loại hoa
X_daisy,Y_daisy=DP.proData(folder_daisy_path,1,100)
X_rose,Y_rose=DP.proData(folder_rose_path,3,100)
X_dandelion,Y_dandelion=DP.proData(folder_dandelion_path,2,100)
X_sunflower,Y_sunflower=DP.proData(folder_sunflower_path,4,100)
X_tulip,Y_tulip=DP.proData(folder_rose_path,5,100)


#chia tập train-test theo tỉ lệ 3-7
X_daisy_train,Y_daisy_train,X_daisy_test,Y_daisy_test=TM.devideData(X_daisy,Y_daisy,0.3)
X_rose_train,Y_rose_train,X_rose_test,Y_rose_test=TM.devideData(X_rose,Y_rose,0.3)
X_dandelion_train,Y_dandelion_train,X_dandelion_test,Y_dandelion_test=TM.devideData(X_dandelion,Y_dandelion,0.3)
X_sunflower_train,Y_sunflower_train,X_sunflower_test,Y_sunflower_test=TM.devideData(X_sunflower,Y_sunflower,0.3)
X_tulip_train,Y_tulip_train,X_tulip_test,Y_tulip_test=TM.devideData(X_tulip,Y_tulip,0.3)

X_Test=X_daisy_test+X_dandelion_test+X_rose_test+X_sunflower_test+X_tulip_test
Y_Test=Y_daisy_test+Y_dandelion_test+Y_rose_test+Y_sunflower_test+Y_tulip_test

Y_train_fake=[];
Y_test_fake=[];
for i in range(0,len(Y_daisy_train)):
    Y_train_fake.append(1)
for i in range(0,len(Y_daisy_train)):
    Y_train_fake.append(-1)
for i in range(0,len(Y_daisy_test)):
    Y_test_fake.append(1)
for i in range(0,len(Y_daisy_test)):
    Y_test_fake.append(-1)

#ghép 2 tập phan loai 1 2 để tạo 2 tập train và test hoàn chỉnh
X12_train=np.concatenate((X_daisy_train,X_dandelion_train), axis=0)
X12_test=np.concatenate((X_daisy_test,X_dandelion_test), axis=0)

#ghép 2 tập phan loai 1 3 để tạo 2 tập train và test hoàn chỉnh
X13_train=np.concatenate((X_daisy_train,X_rose_train), axis=0)
X13_test=np.concatenate((X_daisy_test,X_rose_test), axis=0)

#ghép 2 tập phan loai 1 4 để tạo 2 tập train và test hoàn chỉnh
X14_train=np.concatenate((X_daisy_train,X_sunflower_train), axis=0)
X14_test=np.concatenate((X_daisy_test,X_sunflower_test), axis=0)

#ghép 2 tập phan loai 1 5 để tạo 2 tập train và test hoàn chỉnh
X15_train=np.concatenate((X_daisy_train,X_tulip_train), axis=0)
X15_test=np.concatenate((X_daisy_test,X_tulip_test), axis=0)

#ghép 2 tập phan loai 2 3 để tạo 2 tập train và test hoàn chỉnh
X23_train=np.concatenate((X_dandelion_train,X_rose_train), axis=0)
X23_test=np.concatenate((X_dandelion_test,X_rose_test), axis=0)

#ghép 2 tập phan loai 2 4 để tạo 2 tập train và test hoàn chỉnh
X24_train=np.concatenate((X_dandelion_train,X_sunflower_train), axis=0)
X24_test=np.concatenate((X_dandelion_test,X_sunflower_test), axis=0)

#ghép 2 tập phan loai 2 3 để tạo 2 tập train và test hoàn chỉnh
X25_train=np.concatenate((X_dandelion_train,X_tulip_train), axis=0)
X25_test=np.concatenate((X_dandelion_test,X_tulip_test), axis=0)

#ghép 2 tập phan loai 3 4 để tạo 2 tập train và test hoàn chỉnh
X34_train=np.concatenate((X_rose_train,X_sunflower_train), axis=0)
X34_test=np.concatenate((X_rose_test,X_sunflower_test), axis=0)

#ghép 2 tập phan loai 3 5 để tạo 2 tập train và test hoàn chỉnh
X35_train=np.concatenate((X_rose_train,X_tulip_train), axis=0)
X35_test=np.concatenate((X_rose_test,X_tulip_test), axis=0)

#ghép 2 tập phan loai 4 5 để tạo 2 tập train và test hoàn chỉnh
X45_train=np.concatenate((X_sunflower_train,X_tulip_train), axis=0)
X45_test=np.concatenate((X_sunflower_test,X_tulip_test), axis=0)


w12,b12=TM.trainingSVM(X12_train,Y_train_fake,0.1)
w13,b13=TM.trainingSVM(X13_train,Y_train_fake,0.1)
w14,b14=TM.trainingSVM(X14_train,Y_train_fake,0.1)
w15,b15=TM.trainingSVM(X15_train,Y_train_fake,0.1)
w23,b23=TM.trainingSVM(X23_train,Y_train_fake,0.1)
w24,b24=TM.trainingSVM(X24_train,Y_train_fake,0.1)
w25,b25=TM.trainingSVM(X25_train,Y_train_fake,0.1)
w34,b34=TM.trainingSVM(X34_train,Y_train_fake,0.1)
w35,b35=TM.trainingSVM(X35_train,Y_train_fake,0.1)
w45,b45=TM.trainingSVM(X45_train,Y_train_fake,0.1)

print(PM.Evaluate(w12,b12,X12_test,Y_test_fake));
print(PM.Evaluate(w13,b13,X13_test,Y_test_fake));
print(PM.Evaluate(w14,b14,X14_test,Y_test_fake));
print(PM.Evaluate(w15,b15,X15_test,Y_test_fake));
print(PM.Evaluate(w23,b23,X23_test,Y_test_fake));
print(PM.Evaluate(w24,b24,X24_test,Y_test_fake));
print(PM.Evaluate(w25,b25,X25_test,Y_test_fake));
print(PM.Evaluate(w34,b34,X34_test,Y_test_fake));
print(PM.Evaluate(w35,b35,X35_test,Y_test_fake));
print(PM.Evaluate(w45,b45,X45_test,Y_test_fake));


rows = 6
cols = 6
w = []
value = 0
for _ in range(rows):
    row = []
    for _ in range(cols):
        row.append(value)
    w.append(row)
b = []
value = 0
for _ in range(rows):
    row = []
    for _ in range(cols):
        row.append(value)
    b.append(row)
w[1][2]=w12
w[1][3]=w13
w[1][4]=w14
w[1][5]=w15
w[2][3]=w23
w[2][4]=w24
w[2][5]=w25
w[3][4]=w34
w[3][5]=w35
w[4][5]=w45
b[1][2]=b12
b[1][3]=b13
b[1][4]=b14
b[1][5]=b15
b[2][3]=b23
b[2][4]=b24
b[2][5]=b25
b[3][4]=b34
b[3][5]=b35
b[4][5]=b45

print(PM.EvaluateOneForOne(w,b,X_Test,Y_Test))















