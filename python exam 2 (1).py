#!/usr/bin/env python
# coding: utf-8

# 
# <center><h1><font color='blue'> problem 1 <font></h1></center>
# #LIBRARIES
# import os  #interacting with the underlying operating system
# import numpy as np  #ipmorting pandas library
# import math
# import matplotlib.pyplot as plt
# #CONSTANTS
# E = 1#200*10**9    
# A = 1#0.005
# F1 = 400 # in (KN/m)
# F2 = 200 # in (KN/m)

# In[ ]:


#LIBRARIES
import os #interacting with the underlying operating system
import numpy as np #ipmorting pandas library
import math import matplotlib.pyplot as plt
#CONSTANTS
E = 1#200*10**9
A = 1#0.005
F1 = 400 # in (KN/m)
F2 = 200 # in (KN/m)


# ## Elements of node 1 &2

# In[27]:


theta = 45 #rads

#top left quadrants
a11 = math.cos(theta)**2
a12 = math.cos(theta)*math.sin(theta)
a21 = math.cos(theta)*math.sin(theta)
a22 = math.sin(theta)**2

X11_12 = (E*A)*np.array([[a11,a12],[a21,a22]])

#top right quadrants
a11 = -math.cos(theta)**2
a12 = -math.cos(theta)*math.sin(theta)
a21 = -math.cos(theta)*math.sin(theta)
a22 = -math.sin(theta)**2

X12_12 = (E*A)*np.array([[a11,a12],[a21,a22]])

#Bottom left quadrants
a11 = -math.cos(theta)**2
a12 = -math.cos(theta)*math.sin(theta)
a21 = -math.cos(theta)*math.sin(theta)
a22 = -math.sin(theta)**2

X21_12 = (E*A)*np.array([[a11,a12],[a21,a22]])

#Bottom right quadrants
a11 = math.cos(theta)**2
a12 = math.cos(theta)*math.sin(theta)
a21 = math.cos(theta)*math.sin(theta)
a22 = math.sin(theta)**2

X22_12 = (E*A)*np.array([[a11,a12],[a21,a22]])

#const X11 and X12 along vertical axis

top = const((X11_X12,X12_X12),axis = 1)

print(X11_12)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# <center><h1><font color='blue'> problem 2 <font></h1></center>

# ##Gringorten plotting postion

# In[13]:


#LIBRARIES
import os #interacting with the underlying operating system
import numpy as np #ipmorting pandas library
import math
import matplotlib.pyplot as plt


# In[ ]:


#READING THE CSV DATA
a = pd.read_csv('')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# <center><h1><font color='blue'> problem 3 <font></h1></center>

# In[42]:


#Libraries
import numpy as np #ipmorting pandas library
import math
import matplotlib.pyplot as plt


# In[37]:


#Function
f = lambda x:math.log(x)


# In[38]:


def f(x):
    return (x-1)*(x-3)*(x-1)+45

x = np.linspace(0, 10, 200)
y = f(x)


# In[39]:


#Choose a region to integrate over and take only a few points in that region
a, b = 1, 3 # the left and right boundaries
N = 5 # the number of points
xint = np.linspace(a, b, N,)
yint = f(xint)


# In[40]:


#Plot both the function and the area below it in the trapezoid approximation
plt.plot(x, y, lw=2)
plt.axis([0, 9, 0, 140])
plt.fill_between(xint, 0, yint, facecolor='gray', alpha=0.3)
plt.text(0.5 * (a + b), 45,r"$\int_a^b f(x)dx$", horizontalalignment='center', fontsize=20);


# In[41]:


#Compute the integral both at high accuracy and with the trapezoid approximation
from __future__ import print_function
from scipy.integrate import quad
integral, error = quad(f, a, b)
integral_trapezoid = sum( (xint[1:] - xint[:-1]) * (yint[1:] + yint[:-1]) ) / 2
print("The integral is:", integral, "+/-", error)
print("The trapezoid approximation with", len(xint), "points is:", integral_trapezoid)


# In[ ]:




