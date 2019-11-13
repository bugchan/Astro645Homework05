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
  thetaU=np.linspace(0,2*np.pi,32)
  q0=rU*np.cos(thetaU)+q
  p0=rU*np.sin(thetaU)+p
  lenq=len(q0)
  IVb=np.concatenate((q0.reshape(lenq,1),
                     p0.reshape(lenq,1)),
                    axis=1)
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

#%% 2.C
start = time.time()
t,thetaRot,thetaDotRot=NI.RK4(dvdt,a,b,h,IV[0],dim=1)
t,thetaLib,thetaDotLib=NI.RK4(dvdt,a,b,h,IV[1],dim=1)
t,thetaSep,thetaDotSep=NI.RK4(dvdt,a,b,h,IV[2],dim=1)
end = time.time()
print('Time to run: %0.2f'%(end - start))

#%% Plot 2.C
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
ax1.grid()
ax1.legend()

fig1.tight_layout()
fig1.savefig('PendulumPhaseSpace.pdf')

#%% 2.D q,p=0,1
tau0=2*np.pi
bArray2d=np.array([.25,.5,1])*tau0
P2d=calcUs(rU=1/2,q=0,p=1,bArray=bArray2d)
IV2d,qA2d,pA2d,qfinalA2d,pfinalA2d=P2d

#%% 2.E q,p=0,1.5
bArray2e=np.array([.2,.4,.75])*tau0
P2e=calcUs(rU=1/2,q=0,p=1.5,bArray=bArray2e)
IV2e,qA2e,pA2e,qfinalA2e,pfinalA2e=P2e

#%% 2.F q,p=0,2
bArray2f=np.array([.1,.25,.5])*tau0
P2f=calcUs(rU=1/2,q=0,p=2,bArray=bArray2f)
IV2f,qA2f,pA2f,qfinalA2f,pfinalA2f=P2f

#%% Plot 2d
#width,height=SP.setupPlot(singleColumn=False)
#grid = plt.GridSpec(1,1)
fig2 = plt.figure(figsize=(width,height))

ax2 = fig2.add_subplot(grid[0,0])
ax2.plot(qA2d.T,pA2d.T,'k-')
ax2.plot(IV2d[:,0],IV2d[:,1],'-o',label=r'$\tau=0$')
for i in range(3):
  ax2.plot(qfinalA2d[i,:],pfinalA2d[i,:],'-o',
           label=r'$\tau=$%1.2f'%bArray2d[i])
ax2.set_xlabel(r'$\theta$')
#ax2.set_xlim(-np.pi,2.5)
ax2.set_ylabel(r'$\dot{\theta}$')
ax2.grid()
ax2.legend()
#ax2.set_aspect('equal')

fig2.tight_layout()
fig2.savefig('PendulumPhaseSpaceUs2d.pdf')

#%% Plot 2e
#width,height=SP.setupPlot(singleColumn=False)
#grid = plt.GridSpec(1,1)
fig3 = plt.figure(figsize=(width,height))

ax3 = fig3.add_subplot(grid[0,0])
ax3.plot(qA2e.T,pA2e.T,'k-')
ax3.plot(IV2e[:,0],IV2e[:,1],'-o',label=r'$\tau=0$')
for i in range(3):
  ax3.plot(qfinalA2e[i,:],pfinalA2e[i,:],'-o',
           label=r'$\tau=$%1.2f'%bArray2e[i])
ax3.set_xlabel(r'$\theta$')
#ax3.set_xlim(-np.pi,np.pi)
ax3.set_ylabel(r'$\dot{\theta}$')
ax3.grid()
ax3.legend()
#ax3.set_aspect('equal')

fig3.tight_layout()
fig3.savefig('PendulumPhaseSpaceUs2e.pdf')

#%% Plot 2f
#width,height=SP.setupPlot(singleColumn=False)
#grid = plt.GridSpec(1,1)
fig4 = plt.figure(figsize=(width,height))

ax4 = fig4.add_subplot(grid[0,0])
ax4.plot(qA2f.T,pA2f.T,'k-')
ax4.plot(IV2f[:,0],IV2f[:,1],'-o',label=r'$\tau=0$')
for i in range(3):
  ax4.plot(qfinalA2f[i,:],pfinalA2f[i,:],'-o',
           label=r'$\tau=$%1.2f'%bArray2f[i])
ax4.set_xlabel(r'$\theta$')
#ax4.set_xlim(-np.pi,np.pi)
ax4.set_ylabel(r'$\dot{\theta}$')
ax4.grid()
ax4.legend()
#ax4.set_aspect('equal')

fig4.tight_layout()
fig4.savefig('PendulumPhaseSpaceUs2f.pdf')

#%% Save Data to csv file

#names=np.array(['x'    ,'y'   , '$v_x$','$v_y$','Time'])
#indexNames=['Box Orbit','Tube Orbit ','Box Orbit 2']
#row1=np.array([x10[0],x10[1],v10[0],v10[1],B])
#row2=np.array([x20[0],x20[1],v20[0],v20[1],B])
#row3=np.array([x30[0],x30[1],v30[0],v30[1],B])
#
#rows=[row1, row2,row3]
#
#df = pd.DataFrame(rows,columns=names,index=indexNames)
#
#with open('LogPotentialIV.tex','w') as tf:
#    tf.write(df.to_latex(float_format='%2.2f',
#                         index=True,
#                         escape=False))