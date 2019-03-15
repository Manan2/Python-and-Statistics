#!/usr/bin/env python
# coding: utf-8

# # Read File and Show first Values

# In[1]:


import numpy as np
import pandas as pd 
import statsmodels.formula.api as smf
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats


# In[2]:


data1 = pd.read_csv("gpa2.csv") 
# Preview the first 5 lines of the loaded data 
data1.head()


# # Describe Data

# In[2]:


data1.describe()


# # Further Description

# In[3]:


df =(data1.min(),data1.max(),(data1.max()-data1.min()),data1.mean(),data1.median(),data1.std(),data1.isnull().sum(),(data1.isnull().sum()*100)/(len(data1)))
df = pd.concat(df,1)
df.columns = ['min','max','range','mean','median','st dev','null','%_null']
df


# In[4]:


import pandas as pd
import numpy as np

rs = np.random.RandomState(0)
df = pd.DataFrame(data1)
corr = df.corr()
corr.style.background_gradient(cmap='coolwarm')


# #### By looking at above correlation matrix we find that we will have to remove certain variable like female, white, black, hsize as they do not either explain the target variable or increases multicollinearity

# # Drop Values for Regression Model

# In[5]:


data1.drop(["female", "white" , "black" ,"hsizesq"], axis = 1, inplace = True)


# In[6]:


data1


# In[ ]:





# # Check Correlation using Correlation Matrix

# In[7]:


import pandas as pd
import numpy as np

rs = np.random.RandomState(0)
df = pd.DataFrame(data1)
corr = df.corr()
corr.style.background_gradient(cmap='coolwarm')


# # These are scatter plot and histogram of the important variables

# In[28]:


data1.plot(x='colgpa', y='sat', kind="scatter")
data1.plot(x='colgpa', y='athlete', kind="scatter")
data1.plot(x='colgpa', y='hsize', kind="scatter")
data1.plot(x='colgpa', y='hsrank', kind="scatter")

data1.hist()
plt.show()


# In[13]:


plt.figure(figsize=(15,8))
plt.title('College GPA vs High School Size')
plt.xlabel('colgap')
plt.ylabel('hsize')
plt.scatter(data1['colgpa'],data1['hsize'],color='c', alpha=.5)
#plt.plot(cpdata['educ'],cpdata['exper'])
plt.legend()
plt.show()


# # Checking for any null values

# In[5]:


data1.isnull().values.any()


# # Running the Multiple regression on the model

# In[15]:


import statsmodels.formula.api as smf
import statsmodels.api as sm
mod1 = smf.ols('colgpa~sat+tothrs+athlete+verbmath+hsize+hsrank+hsperc', data=data1).fit()
mod1.summary()
#sat	tothrs	colgpa	athlete	verbmath	hsize	hsrank	hsperc


# ## As the regression output now says that there could be multicollinearity in other variables we check the p-value and remove those variables

# In[27]:


mod1 = smf.ols('colgpa~sat+tothrs+athlete+hsrank+hsperc', data=data1).fit()
mod1.summary()


# ## Now we can see that all the variables as significant

# # Group-By

# In[21]:


data1['colgpabins'] = pd.cut(data1['colgpa'],[0,1,2,3,4])
data1.groupby(['athlete','colgpabins']).mean()


# In[22]:


data1.groupby(['female','colgpabins']).mean()


# In[24]:


data1.groupby(['hsize','colgpa']).mean()


# In[26]:


data1.groupby(['hsrank','colgpa']).mean()


# In[19]:


data1.groupby(['hsperc','colgpa']).mean()


# In[ ]:





# In[ ]:




