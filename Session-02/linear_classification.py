#! \usr\bin\env python
### LinearClassifier Example with perceptron training etc
###
from sklearn.datasets import make_classification
from sklearn.datasets import make_circles
import numpy as np
import matplotlib.pyplot as plt


class LinearClassifier:
    def __init__(self):
        pass
    def sigmoid(self,z):
        return 1.0/(1. + np.exp(-z))

    def loss(self,y,y_ht):
        loss =  -np.mean( y*(np.log(y_ht)) - (1-y)*np.log(1-y_ht) )
        return loss

    def gradient(self,x,y,y_ht):
        m = x.shape[0]
        dw = (1/m)*np.dot(x.T,(y_ht-y)) #wrt w
        db = (1/m)*np.sum((y_ht-y)) # wrt bias
        return dw,db 

    def plot_dec_boundry(self,x,w,b,y,debug):
        x1 = [min(x[:,0]), max(x[:,0])]
        m = -w[0]/w[1]
        c = -b/w[1]
        x2 = m*x1 + c
        plt.plot(x[:,0][y==0], x[:,1][y==0],'r^')
        plt.plot(x[:,0][y==1], x[:,1][y==1],'bs')
        plt.plot(x1,x2,'y-')
        if debug:
            plt.show()

    def normalize(self,x):
        m,n = x.shape   #m : trg exmple, n: features
        for i in range(n):
            x = (x - x.mean(axis=0))/x.std(axis=0)
        return x    

    def train(self, x,y,bs,epochs,lr):
        m,n = x.shape
        w = np.zeros((n,1))  ## or random ?
        b = 0
        y = y.reshape(m,1)
        x = self.normalize(x)

        losses = []

        for epoch in range(epochs):
            for i in range((m-1)//bs+1):
                start_i = i*bs
                end_i = start_i + bs
                xb = x[start_i:end_i]
                yb = y[start_i:end_i]

                y_ht = self.sigmoid(np.dot(xb,w)+b)

                dw,db = self.gradient(xb,yb,y_ht)

                w -=lr*dw
                b -=lr*db

            l = self.loss(y,self.sigmoid(np.dot(x,w)+b))
            losses.append(l)
            # self.plot_dec_boundry(x,w,b,y,1)

        return w,b,losses

    def step_func(self,z):
        return 1.0 if (z > 0) else 0.0

    def percep_train(self,x, y, epochs, lr):
    
        m, n = x.shape
        
        w = np.zeros((n+1,1))   ## weight plus bias
        
        losses = []
        
        for epoch in range(epochs):
            n_miss = 0
            for idx, x_i in enumerate(x):
                x_i = np.insert(x_i, 0, 1).reshape(-1,1)
                y_hat = self.step_func(np.dot(x_i.T, w))
          
                if (np.squeeze(y_hat) - y[idx]) != 0:
                    w += lr*((y[idx] - y_hat)*x_i)
                    n_miss += 1
       
            losses.append(n_miss)
            
        return np.array([ [w[1][0]],[w[2][0]] ]),w[0][0], losses
    

    def predict(self,x,w,b):

        x = self.normalize(x)
        preds = self.sigmoid((np.dot(x,w)+b))
        pred_class = []

        pred_class = [1 if i>0.5 else 0 for i in preds]

        return np.array(pred_class)
    
    def accuracy(self,y, y_ht):
        accuracy = np.sum(y == y_ht) / len(y)
        return accuracy


x,Y = make_classification(n_features=2,n_classes=2,n_samples=100,n_redundant=0,n_clusters_per_class=1)
# x,Y = make_circles(n_samples=100,noise=0.03,factor=0.7)

p = LinearClassifier()
w,b,loss = p.train(x,Y,bs=10,epochs=9000,lr=0.01)
print(w,b)
class_pred = p.predict(x,w,b)
print('Accuracy= ',p.accuracy(Y,class_pred) * 100,'%')
plt.subplot(1,2,1)
p.plot_dec_boundry(x,w,b,Y,0)
plt.subplot(1,2,2)
plt.plot(loss)
plt.suptitle(['Accuray' + str(p.accuracy(Y,class_pred) * 100)])
plt.show()

w1,b1,loss1 = p.percep_train(x,Y,epochs=1000,lr=0.01)
print(w1,b1)

class_pred = p.predict(x,w1,b1)
print('Accuracy= ',p.accuracy(Y,class_pred) * 100,'%')

plt.subplot(1,2,1)
p.plot_dec_boundry(x,w1,b1,Y,0)
plt.subplot(1,2,2)
plt.plot(loss1)
plt.suptitle(['Accuray' + str(p.accuracy(Y,class_pred) * 100)])
plt.show()

