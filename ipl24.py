#!/usr/bin/env python
# coding: utf-8

# In[359]:


import pandas as pd
from matplotlib import pyplot as plt
v=pd.read_csv("f:/matches.csv")
v


# In[360]:


#unwanted columns doesnot effect on the result excpet mumbai matches
v=v.drop(['id','venue','method','umpire1','umpire2'],axis='columns')
v


# In[361]:


#thrilling matches
v=v.drop(['super_over'],axis='columns')
v


# In[362]:


#team and their total no_of matches win
e=v['winner'].value_counts()


# In[363]:


e1=pd.DataFrame(e).reset_index()


# In[364]:


e1.columns=['team','wins']
e1


# In[365]:


# we reeplace some team name because teams are changed
v=v.replace('Rising Pune Supergiants','Rising Pune Supergiant')
v=v.replace('Royal Challengers Bengaluru','Royal Challengers Bangalore')
v=v.replace('Kings XI Punjab','Punjab Kings')
v=v.replace('Delhi Daredevils','Delhi Capitals')
v


# In[366]:


#total no of teams
v['winner'].unique()


# In[367]:


#droping unwanted null values
v=v.dropna()


# In[368]:


len(v['winner'].unique())


# In[369]:


#data visualization
e2=v['winner'].value_counts()
e3=pd.DataFrame(e2).reset_index()
e3.columns=['team','wins']
e3


# In[370]:


plt.plot(e3['team'],e3['wins'])
plt.scatter(e3['team'],e3['wins'])
plt.xticks(rotation=90)
plt.xlabel('team')
plt.ylabel('total_wins')
plt.title('performance off team')


# In[371]:


v.head()


# In[372]:


#converting categorical values into numerical values because catogrical values not understand by the machine learning
r1=v['match_type'].unique()


# In[373]:


r1


# In[374]:


from sklearn.preprocessing import LabelEncoder
l=LabelEncoder()
v['match_type']=l.fit_transform(v['match_type'])
v.head()


# In[375]:


x1=sorted(r1)


# In[376]:


e3=v['match_type'].unique()
e3


# In[377]:


#converting the matchtype into numerical datatype
e4=pd.DataFrame(e3,r1).reset_index()
e4.columns=['match_tye','number']
e4


# In[378]:


#for cchampions
c1=v[v['match_type']==3]
c1


# In[385]:


red=c1['winner'].value_counts()
red1=pd.DataFrame(red).reset_index()
red1.columns=['team','titles']
red1


# In[384]:


plt.bar(red1['team'],red1['titles'],color='black')
plt.plot(red1['team'],red1['titles'],color='orange')
plt.xticks(rotation=90)


# In[257]:


t1=v['team1'].unique()
t1


# In[258]:


#converting the matchtype into numerical datatype
v['team1']=l.fit_transform(v['team1'])
v.head()


# In[259]:


t11=v['team1'].unique()
t11=sorted(t11)
t11


# In[260]:


t12=sorted(t1)
t12


# In[261]:


t13=pd.DataFrame(t11,t12).reset_index()
t13.columns=['team','shortcut']
t13


# In[262]:


v[v['team1']==14]


# In[263]:


len(t12)


# In[264]:


v[v['team1']==0]


# In[265]:


t2=v['team2'].unique()


# In[266]:


v['team2']=l.fit_transform(v['team2'])
v['team2'].unique()


# In[267]:


v.head()


# In[268]:


v[v['winner']=='Kolkata Knight Riders']


# In[269]:


v['toss_winner']=l.fit_transform(v['toss_winner'])
v.head()


# In[270]:


import numpy as np
c1=v['city'].unique()
c1=np.delete(c1,27)
c1


# In[271]:


len(c1)


# In[272]:


c1=sorted(c1)


# In[273]:


c2=sorted(c1)
v['city']=l.fit_transform(v['city'])
v.head()


# In[274]:


r2=v['city'].unique()
r2=sorted(r2)
r2=np.delete(r2,36)


# In[275]:


r1=pd.DataFrame(r2,c1).reset_index()
r1.columns=['venue','shortcut']
r1


# In[276]:


v.head()


# In[277]:


v['toss_decision']=l.fit_transform(v['toss_decision'])
v.head()


# In[278]:


w12=v['winner'].unique()
w12=np.delete(w12,10)
w12=sorted(w12)


# In[290]:


v['winner']=l.fit_transform(v['winner'])
v.head()


# In[291]:


x9=v['winner'].unique()
x9=np.delete(x9,15)
x9=sorted(x9)


# In[292]:


r34=pd.DataFrame(x9,w12).reset_index()
r34.columns=['winner','shortcut']
r34


# In[293]:


from sklearn.tree import DecisionTreeClassifier
tdc=DecisionTreeClassifier()


# In[294]:


tdc


# In[295]:


v.columns


# In[ ]:


from sklearn.preprocessing import train_test_split


# In[300]:


x=v[['city','match_type','team1','team2','toss_winner','toss_decision']]
y=v['winner']
tdc.fit(x,y)


# In[299]:


len(x)


# In[301]:


len(y)


# In[303]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.8)



# In[305]:


tdc.score(x_test,y_test)


# In[307]:


tdc.predict([[8,4,0,1,0,0]])


# In[308]:


v.head()


# In[325]:


q=tdc.predict([[7,4,10,0,0,0]])
e=q[0]
r34.iloc[e,0]


# In[ ]:




