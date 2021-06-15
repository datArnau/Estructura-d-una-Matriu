#!/usr/bin/env python
# coding: utf-8

# # NIVELL 1

# ## Exercici 1

# In[28]:


import numpy as np


# In[2]:


a = np.arange(12)


# In[3]:


a


# ## Exercici 2

# In[4]:


np.mean(a)


# In[5]:


b=a-np.mean(a)


# In[6]:


b


# In[8]:


a.ndim


# In[9]:


a.shape


# In[17]:


a.dtype = 'int64'


# In[18]:


a.dtype


# In[19]:


b.dtype


# ## Exercici 3

# In[21]:


bidi = np.array([[0, 1, 2, 3, 4], [1, 2, 3, 4, 5], [0, -1, -2, -3, -4], [11, 22, 33, 44, 55], [-11, -22, -33, -44, -55]])


# In[22]:


bidi.ndim


# In[23]:


bidi.shape


# In[25]:


bidi.max()


# In[26]:


np.argmax(bidi)


# In[27]:


bidi.max(1)


# # Nivell 2

# ## Exercici 4

# ### Suma de matrius

# In[29]:


c = np.arange(0, 40, 10)


# In[30]:


c.shape


# In[31]:


c = c[:, np.newaxis]


# In[33]:


c.shape
c


# In[34]:


d = np.array([0,1,2])
d


# In[37]:


e = c+d


# In[38]:


e


# ### Multiplicació de matrius

# In[39]:


f = np.triu(np.ones((3, 3)), 1)
f


# In[40]:


g = np.diag([1, 2, 3])
g


# In[41]:


f.dot(g)


# ### Transposició

# In[42]:


f.T


# ## Exercici 5

# In[43]:


e[0,0], e[0,1], e[0,2]


# In[44]:


e[0,0], e[1,0], e[2,0]


# In[54]:


primertros = e[0,:]


# In[56]:


primertros


# In[63]:


segontros = e[1:,0]


# In[64]:


segontros


# In[65]:


suma = primertros + segontros
suma


# ## Exercici 6

# In[66]:


import numpy.ma as ma


# In[72]:


masksuma = ma.masked_array(suma, mask=[0,1,0])


# In[73]:


masksuma


# # Nivell 3

# In[74]:


import matplotlib as plt


# In[75]:


import matplotlib.pyplot as plt


# In[76]:


import matplotlib.image as mpimg


# ## Exercici 8

# In[81]:


img = mpimg.imread('C:/Users/TREBALL/Pictures/FONDO COLOR.png')
print(img)


# In[82]:


imgplot = plt.imshow(img)


# In[83]:


convertimg = img


# In[92]:


moratimg = img[:, :, 0]
plt.imshow(moratimg)


# In[93]:


turquesaimg = img[:, :, 1]
plt.imshow(turquesaimg)


# In[94]:


grocimg = img[:, :, 2]
plt.imshow(grocimg)


# In[98]:


plt.imshow(moratimg, cmap="hot")


# In[99]:


imgplot = plt.imshow(moratimg)
imgplot.set_cmap('nipy_spectral')


# In[ ]:





# In[ ]:




