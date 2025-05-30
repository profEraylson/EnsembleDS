
import numpy as np

class ModeloTeste:

    def __init__(self, bias):
        self.bias = bias

    def predict(self, X):        
        
        preds = []
        for i in range(X.shape[0]): 
            pred = X[i][-1] + self.bias
            #pred = np.mean(X[i])
            preds.append(pred)

        return np.array(preds) 