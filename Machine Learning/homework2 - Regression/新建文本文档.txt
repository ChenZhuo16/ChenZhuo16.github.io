#1a.1
import numpy as np
train = np.loadtxt("train_graphs_f16_autopilot_cruise.csv", delimiter=",", skiprows=1, usecols=(1,2,3,4,5,6,7))
temp1=train[0:3427,0:6]
phi=np.zeros((3426,37))
for i in range(0,3426):
    for j in range(0,36):
        m=j%6
        phi[i,j+1]=np.power(temp1[i,m],(j-m)/6+1)
phi[:,0]=1
ytemp1=train[:,6]
ytemp1=ytemp1.astype(float)
test = np.loadtxt("test_graphs_f16_autopilot_cruise.csv", delimiter=",", skiprows=1, usecols=(1,2,3,4,5,6,7))
temp2=test[0:2284,0:6]
phi2=np.zeros((2283,37))
for i in range(0,2283):
    for j in range(0,36):
        m=j%6
        phi2[i,j+1]=np.power(temp2[i,m],(j-m)/6+1)
phi2[:,0]=1
ytemp2=test[0:2284,6]
ytemp2=ytemp2.astype(float)
egtrain=np.zeros((6,1))
egtest=np.zeros((6,1))
for i in range(0,6):
    pt=phi[:,0:(6*(i+1)+1)]
    w=np.linalg.inv(pt.T.dot(pt)).dot(pt.T).dot(ytemp1)
    trainfit=phi[:,0:(6*(i+1)+1)].dot(w)
    egtrain1=((np.subtract(trainfit.T,ytemp1.T)).T)**2
    egtrain[i]=np.mean(egtrain1,axis=0)**0.5
    testfit=phi2[:,0:(6*(i+1)+1)].dot(w)
    egtest1=((np.subtract(testfit.T,ytemp2.T)).T)**2
    egtest[i]=np.mean(egtest1,axis=0)**0.5
from matplotlib import pyplot as plt
x=np.linspace(1,6,6)
pl.title('plot 2')
plt.plot(x,egtrain,'r')
plt.plot(x,egtest,'g')
plt.show()
#1a.2
I=np.eye(37)
regtrain=np.ones(61)
regtest=np.ones(61)
for i in range(0,61):
    wtr=np.linalg.inv(phi.T.dot(phi)+np.exp(i-40)*I).dot(phi.T).dot(ytemp1)
    wte=wtr
    trainfit1=phi.dot(wtr)
    testfit1=phi2.dot(wte)
    regtrain1=((np.subtract(trainfit1.T,ytemp1.T)).T)**2
    regtrain[i]=np.mean(regtrain1,axis=0)**0.5
    regtest1=((np.subtract(testfit1.T,ytemp2.T)).T)**2
    regtest[i]=np.mean(regtest1,axis=0)**0.5
x1=np.linspace(-40,19,61)
pl.title('plot 2')
plt.plot(x1,regtrain,'r')
plt.plot(x1,regtest,'g')
plt.show()
#1.b
testn=np.loadtxt("test_locreg_f16_autopilot_cruise.csv", delimiter=",", skiprows=1, usecols=(1,2,3,4,5,6,7))
xb=testn[:,0:6]
yb=testn[:,6]
xa=train[:,0:6]
ya=train[:,6]
T=np.logspace(-2,1,10,'true',2)
r=np.zeros((100,3426))
yfit=np.zeros((10,100))
rmse1=np.zeros((10,100))
import numpy as np
for t in range(10):
   for i in range(100) :
       for j in range(3426):
           r[i,j]=np.exp(-(np.linalg.norm(xb[i,:]-xa[j,:]))**2/(2*T[t]**2))
       R=np.sqrt(np.diag(r[i,:]))
       w=np.linalg.pinv(R.dot(xa)).dot(R).dot(ya)
       yfit[t,i]=xb[i,:].dot(w)
       rmse1[t,i]=(yfit[t,i]-yb[i])**2
       print rmse1[t,i]
rmse=np.mean(rmse1,axis=1)**0.5
from matplotlib import pyplot as plt       
xxb=np.linspace(0,10,10)
pl.title('plot 3') 
plt.plot(T,rmse,'g')       
plt.show()

#2
import numpy as np
train = np.loadtxt("steel_composition_train.csv", delimiter=",", skiprows=1, usecols=(1,2,3,4,5,6,7,8,9))
test = np.loadtxt("steel_composition_test.csv", delimiter=",", skiprows=1, usecols=(1,2,3,4,5,6,7,8))
x1=train[:,0:8]
y1=train[:,8]
x2=test
phi=np.zeros((618,33))
for i in range(0,618):
    for j in range(0,32):
        m=j%8
        phi[i,j+1]=np.power(x1[i,m],(j-m)/8+1)
phi[:,0]=1
w=np.linalg.inv(phi.T.dot(phi)).dot(phi.T).dot(y1)
phi2=np.zeros((412,33))
for i in range(0,412):
    for j in range(0,32):
        m=j%8
        phi2[i,j+1]=np.power(x2[i,m],(j-m)/8+1)
phi2[:,0]=1
yfit=phi2.dot(w)

#4
import numpy as np
train = np.loadtxt("spambase.train", delimiter=",")
test = np.loadtxt("spambase.test", delimiter=",")
x1=train[:,0:57]
y1=train[:,57]
x2=test[:,0:57]
y2=test[:,57]
x=np.concatenate((x1,x2))
Nspam=np.sum(y1)
Nspam2=np.sum(y2)
catx1=np.zeros((2000,57))
catx2=np.zeros((2601,57))
p=np.zeros((2,57))
p1=np.zeros((2,57))
p0=np.ones((2601,1))
p10=np.ones((2601,1))

for j in range(57):
    count1=0
    count2=0
    for i in range(2000):
        if x1[i,j] < np.median(x[:,j]):
            catx1[i,j]=1
            if y1[i]==1:
                count1=count1+1
            else:
                count2=count2+1
        else:
            catx1[i,j]=2
    p[0,j]=(count1+1)/(Nspam+2)
    p[1,j]=1-p[0,j]
    p1[0,j]=(count2+1)/(2000-Nspam+2)
    p1[1,j]=1-p1[0,j]
for i in range(2601):
    for j in range(57):
        if x2[i,j] < np.median(x[:,j]):
            catx2[i,j]=1
        else:
            catx2[i,j]=2
for i in range(2601):
    for j in range(57):
        if catx2[i,j]==1:
            p0[i]=p0[i]*p[0,j]      
            p10[i]=p10[i]*p1[0,j]
        else:
            p0[i]=p0[i]*p[0,j] 
            p10[i]=p10[i]*p1[0,j]            
p0=p0*(Nspam/2000)
p10=p10*(1-Nspam/2000)
check=(p0>p10)
sm=0
for i in range(2601):
    if check[i]==1:
        sm=sm
    else:
        sm=sm+1
from __future__ import division
error=sm/2601