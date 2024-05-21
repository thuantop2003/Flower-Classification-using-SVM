import numpy as np
def predict(w,b,x):
    # w,b là vector siêu phẳng của module
    a=w.dot(x)+b
    if(a[0]>=0): return 1 
    else: return -1
def Evaluate(w,b,X,Y):
    #hàm này đánh giá độ chính xác của w và b trên tập test X,Y
    T=0
    for i in range(len(X)):
        if(predict(w,b,X[i])==Y[i]):
            T=T+1
    return float(T)/len(X)