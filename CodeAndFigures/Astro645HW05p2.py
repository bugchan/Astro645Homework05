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

def calcUs(rU,q,p,bArray):
  start = time.time()
  #rU=1/2
  thetaU=np.linspace(0,2*np.pi,32)
  q0=rU*np.cos(thetaU)+q
  p0=rU*np.sin(thetaU)+p
  lenq=len(q0)
  IVb=np.concatenate((q0.reshape(lenq,1),
                     p0.reshape(lenq,1)),
                    axis=1)
  #tau=2*np.pi
  #bArray=np.array([.25,.5,1])*tau
  #NArray=np.round(bArray/h)
  qfinalArray=np.zeros((len(bArray),lenq))
  pfinalArray=np.zeros((len(bArray),lenq))
  
  
  for j in range(len(bArray)):
    N=np.int(np.ceil((bArray[j]/h)))
    #tArray=np.zeros((lenq,N+1))
    qArray=np.zeros((lenq,N+1))
    pArray=np.zeros((lenq,N+1))
    
    for i in range(lenq):
      t,q,p=NI.RK4(dvdt,a,bArray[j],h,IVb[i],dim=1)
      #saving values of all 32 orbits
      qArray[i]=q[:,0]
      pArray[i]=p[:,0]
    #Array of the final values of each value of time bArray
    qfinalArray[j]=qArray[:,-1]
    pfinalArray[j]=pArray[:,-1]
  
  
  end = time.time()
  print('Time to run: %0.2f'%(end - start))
  return IVb,qArray,pArray,qfinalArray,pfinalArray

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

#%% 2.D q,p=0,1
#start = time.time()
#rU=1/2
#thetaU=np.linspace(0,2*np.pi,32)
#q0=rU*np.cos(thetaU)
#p0=rU*np.sin(thetaU)+1
#lenq=len(q0)
#IVb=np.concatenate((q0.reshape(lenq,1),
#                   p0.reshape(lenq,1)),
#                  axis=1)
#tau=2*np.pi
#bArray=np.array([.25,.5,1])*tau
#NArray=np.round(bArray/h)
#qfinalArray=np.zeros((len(bArray),lenq))
#pfinalArray=np.zeros((len(bArray),lenq))
#
#for j in range(len(bArray)):
#  N=np.int(np.ceil((bArray[j]/h)))
#  tArray=np.zeros((lenq,N+1))
#  qArray=np.zeros((lenq,N+1))
#  pArray=np.zeros((lenq,N+1))
#  
#  for i in range(lenq):
#    t,q,p=NI.RK4(dvdt,a,bArray[j],h,IVb[i],dim=1)
#    #saving values of all 32 orbits
#    qArray[i]=q[:,0]
#    pArray[i]=p[:,0]
#Array of the final values of each value of time bArray
#  qfinalArray[j]=qArray[:,-1]
#  pfinalArray[j]=pArray[:,-1]
#
#end = time.time()
#print('Time to run: %0.2f'%(end - start))

#%% 2.D q,p=0,1

tau=2*np.pi
bArrayd=np.array([.25,.5,1])*tau
IVb,qArray,pArray,qfinalArray,pfinalArray=calcUs(rU=1/2,q=0,p=1,bArray=bArrayd)

#%% 2.E q,p=0,1.5
bArraye=np.array([.2,.4,.75])*tau
Part2e=calcUs(rU=1/2,q=0,p=1,bArray=bArraye)
IVb,qArray,pArray,qfinalArray,pfinalArray=Part2e

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

#%% PendulumPhaseSpaceUs
width,height=SP.setupPlot(singleColumn=False)
grid = plt.GridSpec(1,1)
fig2 = plt.figure(figsize=(width,height))

ax2 = fig2.add_subplot(grid[0,0])
ax2.plot(qArray.T,pArray.T,'k-')
ax2.plot(IVb[:,0],IVb[:,1],'o',label=r'$\tau=0$')
for i in range(3):
  ax2.plot(qfinalArray[i,:],pfinalArray[i,:],'o',label=r'$\tau=$%1.2f'%bArraye[i])
ax2.set_xlabel(r'$\theta$')
ax2.set_xlim(-np.pi,np.pi)
ax2.set_ylabel(r'$\dot{\theta}$')
#ax1.set_title('Leapfrog Integration')
ax2.grid()
ax2.legend()

fig2.tight_layout()
fig2.savefig('PendulumPhaseSpaceUs.pdf')

