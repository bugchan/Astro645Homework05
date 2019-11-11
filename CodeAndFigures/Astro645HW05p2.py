# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 12:51:02 2019

@author: Sandra Bustamante
"""

import numpy as np
import matplotlib.pyplot as plt
import NumericIntegrations as NI
import SetupPlots as SP
import pandas as pd
import time

#%% Definitions
def dvdt(t,z1,z2):
  #Pendulum equation of motion
  dvdt=-np.sin(z1)
  #print(dvdt)
  return dvdt

def energy(z1,z2):
  #Energy of pendulum
  E=(1/2)*z2**2+(1-np.cos(z1))
  return E

#%%

Ecrit=2 #
indexNames=['Rotation','Libration','Separatrix']
z1=np.array([-3*np.pi/2,-np.pi/2,-3*np.pi/2])
z2=np.array([np.sqrt(2)+.5,np.sqrt(2)-.5,np.sqrt(2)])

IV=np.concatenate((z1.reshape(3,1),z2.reshape(3,1)),
                  axis=1)

a=0
b=100
h=0.1
N=round(b/h)

#%%
start = time.time()
t,thetaRot,thetaDotRot=NI.RK4(dvdt,a,b,h,IV[0],dim=1)
t,thetaLib,thetaDotLib=NI.RK4(dvdt,a,b,h,IV[1],dim=1)
t,thetaSep,thetaDotSep=NI.RK4(dvdt,a,b,h,IV[2],dim=1)
end = time.time()
print('Time to run: %0.2f'%(end - start))

#%%
start = time.time()
rU=1/2
thetaU=np.linspace(0,2*np.pi,32)
q0=rU*np.cos(thetaU)
p0=rU*np.sin(thetaU)+1
lenq=len(q0)
IVb=np.concatenate((q0.reshape(lenq,1),
                   p0.reshape(lenq,1)),
                  axis=1)
tau=2*np.pi
bArray=np.array([.25,.5,1])*tau
N=int(round(bArray/h))
qfinalArray=np.zeros((len(bArray),N+1))
pfinalArray=np.zeros((len(bArray),N+1))


for j in range(len(bArray)):
  N=np.int(np.round(bArray[j]/h))
  tArray=np.zeros((lenq,N+1))
  qArray=np.zeros((lenq,N+1))
  pArray=np.zeros((lenq,N+1))
  
  for i in range(lenq):
    t,q,p=NI.RK4(dvdt,a,bArray[j],h,IVb[i],dim=1)
    tArray[i]=t
    qArray[i]=q[:,0]
    pArray[i]=p[:,0]
  
  qfinalArray[j]=qArray[:,-1]
  pfinalArray[j]=pArray[:,-1]

end = time.time()
print('Time to run: %0.2f'%(end - start))

#%%
width,height=SP.setupPlot(singleColumn=False)
grid = plt.GridSpec(1,1)
fig1 = plt.figure(figsize=(width,height))

ax1 = fig1.add_subplot(grid[0,0])
ax1.plot(thetaRot,thetaDotRot,label=indexNames[0])
ax1.plot(thetaLib,thetaDotLib,label=indexNames[1])
ax1.plot(thetaSep,thetaDotSep,label=indexNames[2])
ax1.set_xlabel(r'$\theta$')
ax1.set_xlim(-np.pi,np.pi)
ax1.set_ylabel(r'$\dot{\theta}$')
#ax1.set_title('Leapfrog Integration')
ax1.grid()
ax1.legend()

fig1.tight_layout()
fig1.savefig('PendulumPhaseSpace.pdf')

#%%
width,height=SP.setupPlot(singleColumn=True)
grid = plt.GridSpec(1,1)
fig2 = plt.figure(figsize=(width,height))

ax2 = fig2.add_subplot(grid[0,0])
for i in range(lenq):
  ax2.plot(qArray[i],pArray[i])
  ax2.plot(IVb[i,0],IVb[i,1],'o')
ax2.set_xlabel(r'$\theta$')
ax2.set_xlim(-np.pi,np.pi)
ax2.set_ylabel(r'$\dot{\theta}$')
#ax1.set_title('Leapfrog Integration')
ax2.grid()
#ax2.legend()

fig2.tight_layout()
fig2.savefig('PendulumPhaseSpaceUs.pdf')

