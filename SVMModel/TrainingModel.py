import numpy as np
from sklearn.svm import SVC
import json
def trainingSVM(X,Y,mC):
    #sử dụng hệ số C và tập dữ liệu để training model và tìm w và b

    #thiết lập model
    model=SVC(kernel='linear',C=mC)

    #train model
    model.fit(X,Y)

    #trả về w,b
    return model.coef_, model.intercept_
def devideData(X,Y,p):
    #hàm này chia 2 tập X,Y thành 2 phần X_test,Y_test: p*n  phần tử đầu, X_train, Y_train là số phần tử còn lại

    
    n=len(X)
    count=p*n
    X_train=[]
    Y_train=[]
    X_test=[]
    Y_test=[]
    for i in range (n):
        if i>count :
            X_train.append(X[i])
            Y_train.append(Y[i])
        else:
            X_test.append(X[i])
            Y_test.append(Y[i])
    return X_train,Y_train,X_test,Y_test
def makeRandomData(n):
    #hàm này tạo random các tập X (data),Y(nhãn) để test thuật toán 
    X=[]
    Y=[]
    for i in range(n):
        X.append([i,i+9+np.random.rand()])
        Y.append(1)
    for i in range(n):
        X.append([i,i-9-np.random.rand()])
        Y.append(-1)
    return np.array(X), np.array(Y)
def saveModel(w,b):
    #hàm lưu các thông số của model sau khi train (w,b) vào file SVM.json
    sw=w.tolist()
    sb=b.tolist()
    data = {'w': sw, 'b': sb}
    with open('SVM.json', 'w') as f:
        json.dump(data, f)
def callModel():
    #hàm gọi các thông số của model sau khi train (w,b) từ file SVM.json
    with open('SVM.json', 'r') as f:
        data = json.load(f)
    w = data['w']
    b = data['b']
    return w,b
                 