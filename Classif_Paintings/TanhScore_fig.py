# -*- coding: utf-8 -*-
"""
Created on Sat Sep 12 18:13:05 2020

@author: gonthier
"""

from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
#%matplotlib inline

import seaborn as sns
sns.set_theme(style="whitegrid")
import matplotlib
matplotlib.rcParams['text.usetex'] = True

#ax = fig.add_subplot(3, 4, 1, projection='3d') # nrows, ncols, index,
#ax.set_xlabel("Wx+b")
#ax.set_ylabel("Objectness score")
#ax.set_zlabel("Aggregated Score")
#ax.set_zlim(-1.01, 1.01)
#ax.set_axis_off()
#
#axs[0,0].plot_surface(X, Y, score_with_obj_inside_tanh, cmap=cm.coolwarm,
#                       linewidth=0, antialiased=False)
#
#axs[0,0].gca(projection='3d')

v = np.linspace(-1.0, 1.0, 15, endpoint=True)

fig=plt.figure()
ax = fig.gca(projection='3d')
ax.set_xlabel(r'$W^{T}X+b$')
ax.set_ylabel(r'Objectness score')
ax.set_zlabel(r'Aggregated Score')
ax.set_title(r'$\mathop{Tanh}  \lbrace \left( s \right) \left(W^{T} x + b \right) \rbrace$')

x = np.arange(-4,4,0.01)
obj_score = np.arange(0,1,0.01)
X, Y = np.meshgrid(x, obj_score)
score_with_obj_inside_tanh = np.tanh(X*Y)

surf = ax.plot_surface(X, Y, score_with_obj_inside_tanh, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False,vmin=-1.,vmax=1.,alpha=0.5)
fig.colorbar(surf, ax=ax, orientation='vertical')

cset = ax.contourf(X, Y, score_with_obj_inside_tanh, zdir='z', offset=-1, cmap=cm.coolwarm)
cset = ax.contourf(X, Y, score_with_obj_inside_tanh, zdir='x', offset=-4, cmap=cm.coolwarm)
cset = ax.contourf(X, Y, score_with_obj_inside_tanh, zdir='y', offset=0, cmap=cm.coolwarm)



fig, (ax1, ax2,ax3) = plt.subplots(1, 3)
fig.suptitle(r'$\mathop{Tanh}  \lbrace \left( s \right) \left(W^{T} x + b \right) \rbrace$')
cset = ax1.contourf(X, Y, score_with_obj_inside_tanh, zdir='z', offset=-1, cmap=cm.coolwarm,vmin=-1.,vmax=1.)
cset = ax2.contourf(Y, score_with_obj_inside_tanh,X, zdir='x', offset=-4, cmap=cm.coolwarm,vmin=-1.,vmax=1.)
#cset = ax2.contourf(X, Y, score_with_obj_inside_tanh, zdir='x', offset=0, cmap=cm.coolwarm,vmin=-1.,vmax=1.)
cset = ax3.contourf(X,score_with_obj_inside_tanh,Y, zdir='z', offset=0, cmap=cm.coolwarm,vmin=-1.,vmax=1.)
fig.colorbar(surf, ax=ax3, orientation='vertical')


fig=plt.figure()
ax = fig.gca(projection='3d')
ax.set_xlabel("Wx+b")
ax.set_ylabel("Objectness score")
ax.set_zlabel("Aggregated Score")


x = np.arange(-4,4,0.01)
obj_score = np.arange(0,1,0.01)
X, Y = np.meshgrid(x, obj_score)
score_with_obj_outside_tanh = Y*np.tanh(X)
surf = ax.plot_surface(X, Y, score_with_obj_outside_tanh, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

fig=plt.figure()
ax = fig.gca(projection='3d')
ax.set_xlabel("Wx+b")
ax.set_ylabel("Objectness score")
ax.set_zlabel("Aggregated Score with lambda = 0.5")
ax.set_zlim(-1, 1)
x = np.arange(-4,4,0.01)
obj_score = np.arange(0,1,0.01)
X, Y = np.meshgrid(x, obj_score)
lambda_ = 0.5
score_with_addition = (1-lambda_)*np.tanh(X) + lambda_*Y*np.sign(X)
surf = ax.plot_surface(X, Y, score_with_obj_outside_tanh, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)


fig, axs = plt.subplots(3, 3) # 4 lignes 3 colonnes

cset = ax.contourf(X, Y, score_with_addition, zdir='z', offset=-1, cmap=cm.coolwarm)
cset = ax.contourf(X, Y, score_with_addition, zdir='x', offset=-4, cmap=cm.coolwarm)
cset = ax.contourf(X, Y, score_with_addition, zdir='y', offset=0, cmap=cm.coolwarm)


x = np.arange(-4,4.1,0.1)
#obj_score = np.arange(0,1,0.1/20)
obj_score = np.arange(0,1.1,0.1)
X, Y = np.meshgrid(x, obj_score)
score_with_obj_outside_tanh = Y*np.tanh(X)
score_with_obj_inside_tanh = np.tanh(X*Y)
lambda_ = 0.5
score_with_addition = (1-lambda_)*np.tanh(X) + lambda_*Y*np.sign(X)
fig, (ax1, ax2,ax3) = plt.subplots(1, 3)
cset1 = ax1.contourf(X, Y, score_with_obj_inside_tanh, levels=40,cmap=cm.coolwarm,vmin=-1.,vmax=1.)
cset = ax2.contourf(X, Y, score_with_obj_outside_tanh, levels=40, cmap=cm.coolwarm,vmin=-1.,vmax=1.)
cset = ax3.contourf(X, Y, score_with_addition, levels=40, cmap=cm.coolwarm,vmin=-1.,vmax=1.)
#fig.colorbar(surf, ax=ax3, orientation='horizontal')

#cset1 = ax1.imshow(score_with_obj_inside_tanh, vmin=-1, vmax=1, cmap=cm.coolwarm)
#cset2 = ax2.imshow(score_with_obj_outside_tanh, vmin=-1, vmax=1, cmap=cm.coolwarm)
#cset3 = ax3.imshow(score_with_addition, vmin=-1, vmax=1, cmap=cm.coolwarm)

ax1.set_xlabel(r'$W^{T}X+b$')
ax1.set_ylabel(r'Objectness score')
ax1.set_title(r'$\mathop{Tanh}  \lbrace  s  \left(W^{T} x + b \right) \rbrace$')
ax2.set_xlabel(r'$W^{T}X+b$')
ax2.set_ylabel(r'Objectness score')
ax2.set_title(r'$s * \mathop{Tanh}  \lbrace W^{T} x + b \rbrace$')
ax3.set_xlabel(r'$W^{T}X+b$')
ax3.set_ylabel(r'Objectness score')
ax3.set_title(r'$( 1 - \lambda ) * \mathop{Tanh}  \left( W^{T} x + b \right) + \lambda s  \mathop{sign}  \left( W^{T} x + b \right) ~ with ~ \lambda = 0.5$')

#fig.suptitle(r'
fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
fig.colorbar(cset1, cax=cbar_ax, orientation='vertical')

