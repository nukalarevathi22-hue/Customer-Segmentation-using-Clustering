#!/usr/bin/env python
# coding: utf-8

# In[63]:


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans


# In[64]:


df = pd.read_csv(r"C:\Users\revat\Downloads\archive (14)\Mall_Customers.csv")


# In[65]:


df


# In[6]:


df.head()


# In[7]:


df.tail()


# In[8]:


df.head(5)


# In[9]:


df.tail(10)


# In[10]:


df.describe()


# In[11]:


df.shape


# In[12]:


df.info()


# In[13]:


df.isnull().sum()


# In[14]:


df.duplicated().sum()


# In[17]:


df.columns


# In[19]:


df["Gender"].value_counts().plot(kind="bar")


# In[29]:


plt.hist(df["Age"])
plt.xlabel("Age")
plt.ylabel("Count")
plt.show()


# In[24]:


plt.hist(df["Annual Income (k$)"])


# In[25]:


plt.hist(df["Spending Score (1-100)"])


# In[26]:


X = df[['Annual Income (k$)','Spending Score (1-100)']]


# In[30]:


X


# In[33]:


from sklearn.cluster import KMeans

y = []

for x in range(1, 11):
    model = KMeans(n_clusters=x, random_state=42)
    model.fit(X)
    y.append(model.inertia_)

print(y)


# In[34]:


plt.plot(range(1, 11), y,
         color='blue',
         marker='o',
         markersize=8,
         markerfacecolor='red',
         markeredgecolor='black',
         linewidth=2)

plt.xlabel("Number of Clusters")
plt.ylabel("Inertia")
plt.title("Optimal Number of Clusters")
plt.grid(True)

plt.show()


# In[49]:


kmeans = KMeans(n_clusters=3,random_state=42)
df["Cluster"] = kmeans.fit_predict(X)


# In[50]:


df.head()


# In[51]:


plt.figure(figsize=(8,6))
plt.scatter(
    df["Annual Income (k$)"],
    df["Spending Score (1-100)"],
    c=df["Cluster"]
)
plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
plt.title("Customer Segmentation")


# In[59]:


import matplotlib.pyplot as plt
plt.figure(figsize=(8,6))
colors = ['green', 'red', 'blue']
labels = ['Budget Shoppers','High Spenders','Occasional Buyers']


# In[60]:


for i in range(3):
    plt.scatter(
        df[df["Cluster"] == i]["Annual Income (k$)"],
        df[df["Cluster"] == i]["Spending Score (1-100)"],
        color=colors[i],
        label=labels[i]
    )
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))


# In[55]:


plt.scatter(
    kmeans.cluster_centers_[:,0],
    kmeans.cluster_centers_[:,1],
    color='black',
    marker='X',
    s=200,
    label='Centroids'
)

plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.title("Customer Segmentation using K-Means")

plt.legend()

plt.grid(True)
plt.tight_layout()

plt.show()


# In[56]:


print(kmeans.cluster_centers_)


# In[61]:


import pandas as pd
cluster_table = pd.DataFrame(
    kmeans.cluster_centers_,
    columns=["Annual Income (k$)", "Spending Score (1-100)"])
cluster_table


# In[62]:


cluster_table["Customer Type"] = [
    "Budget Shoppers",
    "High Spenders",
    "Occasional Buyers"]

cluster_table


# In[ ]:




