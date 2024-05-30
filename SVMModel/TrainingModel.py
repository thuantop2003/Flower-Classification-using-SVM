import numpy as np
from sklearn.svm import SVC
import json
from cvxopt import matrix,solvers
def trainingSVMbyScikitlearn(X,Y,mC):
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
                 
def trainingSVM(XX,YY,mC):
    # chuyển các list về ma trận numpy để tính
    N=len(XX)
    X=np.array(XX)
    X=X.T
    Y=np.array(YY)
    Y=Y.T

    #Tính hệ số V của biến bậc 2
    V=X*Y
    K=matrix(V.T.dot(V))

    #Tính hệ số p của biến bậc 1
    p=matrix(-np.ones((N,1)))

    #Tính các ma trận ràng buộc
    G=matrix(np.vstack((-np.eye(N),np.eye(N))))
    h=matrix(np.vstack((np.zeros(((N,1))),mC*np.ones((N,1)))))
    A=matrix(Y.reshape((-1,N)),tc='d')
    b=matrix(np.zeros((1,1)))

    #gọi hàm tính tối ưu với các hàm số
    solvers.options['show_progress']=False
    sol=solvers.qp(K,p,G,h,A,b)

    #Tính lamda
    l=np.array(sol['x'])
    #check điều kiện để 0<lamda<mC
    S=np.where(l>1e-5)[0]
    S2=np.where(l<0.999*mC)
    #tính w và b theo biểu thức KKT
    M=[val for val in S  if np.any(val == S2)]
    VS=V[:,S]
    XT=X
    print(XT.shape)
    LS=l[S]
    yM=Y[M]
    XM=XT[:,M]
    w_dual=VS.dot(LS).reshape(-1,1)
    b_dual=np.mean(yM.T-w_dual.T.dot(XM))
    return w_dual.T,np.array(b_dual)
