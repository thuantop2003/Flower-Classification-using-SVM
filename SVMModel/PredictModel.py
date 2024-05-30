import numpy as np
def predict(w,b,x):
    # w,b là vector siêu phẳng của module

    #tính theo công thức predict
    a=np.array(w).dot(x)+np.array(b)
    if(a[0]>=0): return 1 
    else: return -1

def Evaluate(w,b,X,Y):
    #hàm này đánh giá độ chính xác của w và b trên tập test X,Y
    T=0
    for i in range(len(X)):
        if(predict(w,b,X[i])==Y[i]):
            T=T+1
    return float(T)/len(X)
def EvaluateAllForOne(w,b,X,Y):
    #hàm này đánh giá độ chính xác của w và b trên tập test X,Y theo các AllforOne cho phân loại đa nhãn
    T=0
    for i in range(len(X)):
        if(predictAllForOne(w,b,X[i])==Y[i]):
            T=T+1
    return float(T)/len(X)
def predictAllForOne(w,b,x):
    #dự phân loại bằng ALlforOne
    for i in range(0,len(w)):
        a=w[i].dot(x)+b[i]
        if(a[0]>=0): return i+1 
    return -1

def predictOneForOne(w,b,x):
    #dự đoán phân loại bằng OneforOne
    n=np.zeros(6)
    max=0;
    index=0;
    for i in range(1,5):
        for j in range(i,6):
            if(predict(w[i][j],b[i][j],x)==1):
                n[i]=n[i]+1
            else:
                n[j]=n[j]+1
    for i in range(1,5):
        if(max<n[i]):
            max=n[i]
            index=i
    return index

def EvaluateOneForOne(w,b,X,Y):
    #hàm này đánh giá độ chính xác của w và b trên tập test X,Y theo cách OneforOne cho phân loại đa nhãn
    T=0
    for i in range(len(X)):
        if(predictOneForOne(w,b,X[i])==Y[i]):
            T=T+1
    return float(T)/len(X)

