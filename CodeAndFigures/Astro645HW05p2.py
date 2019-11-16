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
  qfinalArray[0]=q0
  pfinalArray[0]=p0
  
  for j in np.arange(1,len(bArray)):
    N=np.int(np.ceil((bArray[j]/h)))
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

def calcArea(qArray,pArray):
  n=len(qArray)
  A=np.zeros(n)
  for i in np.arange(n):
    q=qArray[i]
    p=pArray[i]
    A[i]=.5*(q*np.roll(p,-1)-p*np.roll(q,-1)).sum()
  print(A)
  return A

#%%

Ecrit=2
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
ax1.set_xlabel(r'$z_1$')
ax1.set_xlim(-np.pi,np.pi)
ax1.set_ylabel(r'$z_2$')
ax1.grid()
ax1.legend()

fig1.tight_layout()
fig1.savefig('PendulumPhaseSpace.pdf')

#%% 2.D q,p=0,1
l=1
omega=1#constants.g/l
tau0=2*np.pi/omega
bArray2d=np.array([0,.25,.5,1])
tau2d=bArray2d*tau0
qp2d=(0,1)
P2d=calcUs(rU=1/2,q=qp2d[0],p=qp2d[1],bArray=tau2d)
IV2d,qA2d,pA2d,qfinalA2d,pfinalA2d=P2d

A2d=calcArea(qfinalA2d,pfinalA2d)

#%% 2.E q,p=0,1.5
bArray2e=np.array([0,.2,.4,.75])
tau2e=bArray2e*tau0
qp2e=(0,1.5)
P2e=calcUs(rU=1/2,q=qp2e[0],p=qp2e[1],bArray=tau2e)
IV2e,qA2e,pA2e,qfinalA2e,pfinalA2e=P2e
A2e=calcArea(qfinalA2e,pfinalA2e)

#%% 2.F q,p=0,2
bArray2f=np.array([0,.1,.25,.5])
tau2f=bArray2f*tau0
qp2f=(0,2)
P2f=calcUs(rU=1/2,q=qp2f[0],p=qp2f[1],bArray=tau2f)
IV2f,qA2f,pA2f,qfinalA2f,pfinalA2f=P2f
A2f=calcArea(qfinalA2f,pfinalA2f)

#%% Plot 2d
#width,height=SP.setupPlot(singleColumn=False)
#grid = plt.GridSpec(1,1)
fig2 = plt.figure(figsize=(width,height))

ax2 = fig2.add_subplot(grid[0,0])
ax2.plot(qA2d.T,pA2d.T,'k-')
for i in range(4):
  ax2.plot(qfinalA2d[i,:],pfinalA2d[i,:],'o')
  ax2.fill(qfinalA2d[i,:],pfinalA2d[i,:],'o',
           label=r'$\tau=%1.2f \tau_0$'%bArray2d[i])
ax2.set_xlabel(r'$z_1$')
#ax2.set_xlim(-np.pi,2.5)
ax2.set_ylabel(r'$z_2$')
ax2.grid()
ax2.legend()

fig2.tight_layout()
fig2.savefig('PendulumPhaseSpaceUs2d.pdf')

#%% Plot 2e
#width,height=SP.setupPlot(singleColumn=False)
#grid = plt.GridSpec(1,1)
fig3 = plt.figure(figsize=(width,height))

ax3 = fig3.add_subplot(grid[0,0])
ax3.plot(qA2e.T,pA2e.T,'k-')
for i in range(4):
  ax3.plot(qfinalA2e[i,:],pfinalA2e[i,:],'-o')
  ax3.fill(qfinalA2e[i,:],pfinalA2e[i,:],'-o',
           label=r'$\tau=%1.2f \tau_0$'%bArray2e[i])
ax3.set_xlabel(r'$z_1$')
#ax3.set_xlim(-np.pi,np.pi)
ax3.set_ylabel(r'$z_2$')
ax3.grid()
ax3.legend()

fig3.tight_layout()
fig3.savefig('PendulumPhaseSpaceUs2e.pdf')

#%% Plot 2f
#width,height=SP.setupPlot(singleColumn=False)
#grid = plt.GridSpec(1,1)
fig4 = plt.figure(figsize=(width,height))

ax4 = fig4.add_subplot(grid[0,0])
ax4.plot(qA2f.T,pA2f.T,'k-')
for i in range(4):
  ax4.plot(qfinalA2f[i,:],pfinalA2f[i,:],'-o')
  ax4.fill(qfinalA2f[i,:],pfinalA2f[i,:],'-o',
           label=r'$\tau=%1.2f \tau_0$'%bArray2f[i])
ax4.set_xlabel(r'$z_1$')
#ax4.set_xlim(-np.pi,np.pi)
ax4.set_ylabel(r'$z_2$')
ax4.grid()
ax4.legend()

fig4.tight_layout()
fig4.savefig('PendulumPhaseSpaceUs2f.pdf')

#%% Plot 2g
#width,height=SP.setupPlot(singleColumn=False)
#grid = plt.GridSpec(1,1)
fig5 = plt.figure(figsize=(width,height))

ax5 = fig5.add_subplot(grid[0,0])
ax5.plot(A2d,'-o',
         label=r'(p,q)=(%1.1f,%1.1f)'%(qp2d[0],qp2d[1]))
ax5.plot(A2e,'-o',
         label=r'(p,q)=(%1.1f,%1.1f)'%(qp2e[0],qp2e[1]))
ax5.plot(A2f,'-o',
         label=r'(p,q)=(%1.1f,%1.1f)'%(qp2f[0],qp2f[1]))
ax5.set_ylabel('Area')
ax5.grid()
ax5.legend()

fig5.tight_layout()
fig5.savefig('PendulumPhaseSpaceAreas.pdf')

#%% Save Data to csv file

colNames=np.array(['$z_1$','$z_2$','Energy'])
row1=np.array([z1[0],z2[0],energy(z1[0],z2[0])])
row2=np.array([z1[1],z2[1],energy(z1[1],z2[1])])
row3=np.array([z1[2],z2[2],energy(z1[2],z2[2])])

rows=[row1,row2,row3]

df = pd.DataFrame(rows,columns=colNames,
                  index=indexNames)

with open('PendulumIV.tex','w') as tf:
    tf.write(df.to_latex(float_format='%2.2f',
                         index=True,
                         escape=False))
    
#%% Save Data to csv file

#colNames2=(['$%1.2f\tau_0$'%bArray2d[0],
#           '$%1.2f\tau_0$'%bArray2d[1],
#           '$%1.2f\tau_0$'%bArray2d[2],
#           '$%1.2f\tau_0$'%bArray2d[3]])
#row1b=qfinalA2d.T
#row2b=A2d
#indexNames2=([np.arange(32),'Areas'])
##row3=np.array([z1[2],z2[2],energy(z1[2],z2[2])])
#
#rows=row1b
#
#df = pd.DataFrame(rows,columns=colNames2)
#
#with open('Pendulum2d.tex','w') as tf:
#    tf.write(df.to_latex(float_format='%2.2f',
#                         index=True,
#                         escape=False))