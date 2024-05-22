import numpy as np
from SVMModel import TrainingModel as TM
from SVMModel import PredictModel as PM
import json
X,Y=TM.makeRandomData(50)
X_train,Y_train,X_test,Y_test=TM.devideData(X,Y,0.3)
w,b=TM.trainingSVM(X_train,Y_train,mC=0.01)
print(PM.Evaluate(w,b,X_test,Y_test))

