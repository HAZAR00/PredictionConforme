

import  numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
np.random.seed(0)


#def gaussien_1d(x,moy,sigma) :
#    return (1/sigma/np.sqrt(2*np.pi))*np.exp(-(x-moy)**2/sigma**2/2)
n=1000
X = np.random.uniform(-1,1,n)
eps = np.random.normal(2,1,n)
Y = 3*X+eps
plt.scatter(X,Y)
X_train=X[:int(0.7*n)]
Y_train=Y[:int(0.7*n)]
X_test, Y_test=X[int(0.7*n):int(0.85*n)],Y[int(0.7*n):int(0.85*n)]
X_cal, Y_cal=X[int(0.85*n):],Y[int(0.85*n):]
plt.title('ensemble des données')
plt.scatter(X_train,Y_train,label='données d\'entrainement')
plt.scatter(X_test,Y_test,label='données de test')
plt.scatter(X_cal,Y_cal,label='données de calibrage')