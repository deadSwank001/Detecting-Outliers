# -*- coding: utf-8 -*-
"""
Created on Sun Jun  4 21:34:38 2023

@author: swank
"""

Considering Outlier Detection
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')

%matplotlib inline
#Self Explanatory
#Mean, Median, Variance

import numpy as np
from scipy.stats.stats import pearsonr
np.random.seed(101)
normal = np.random.normal(loc=0.0, scale= 1.0, size=1000)
#Self Explanatory
print('Mean: %0.3f Median: %0.3f Variance: %0.3f' %
     (np.mean(normal), np.median(normal), np.var(normal)))
outlying = normal.copy()
outlying[0] = 50.0
print('Mean: %0.3f Median: %0.3f Variance: %0.3f' %
                                (np.mean(outlying), 
                                 np.median(outlying),   
                                 np.var(outlying)))
​
print('Pearson''s correlation: %0.3f p-value: %0.3f' % 
                            pearsonr(normal,outlying))


#Examining a Simple Univariate Method
from sklearn.datasets import load_diabetes
diabetes = load_diabetes()
X, y = diabetes.data, diabetes.target
Samples total 442
Dimensionality 10
Features real, -.2 < x < .2
Targets integer 25 - 346

import pandas as pd
pd.options.display.float_format = '{:.2f}'.format
df = pd.DataFrame(X)
df.describe()
oxplot below
#Prints Boxplot below
fig, axes = plt.subplots(nrows=1, ncols=1, 
                         figsize=(10, 5))
df.boxplot(ax=axes);

#Leveraging on the Gaussian distribution
from sklearn.preprocessing import StandardScaler
Xs = StandardScaler().fit_transform(X)
​
# .any(1) method will avoid duplicating 
df[(np.abs(Xs)>3).any(1)]


#Making assumptions and checking out
Xs_capped = Xs.copy()
o_idx = np.where(np.abs(Xs)>3)
Xs_capped[o_idx] = np.sign(Xs[o_idx]) * 3
from scipy.stats.mstats import winsorize
Xs_winsorized = winsorize(Xs, limits=(0.05, 0.95))
from sklearn.preprocessing import RobustScaler
Xs_rescaled = RobustScaler().fit_transform(Xs)


#Developing a Multivariate Approach
#Using principal component analysis
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from pandas.plotting import scatter_matrix
pca = PCA()
Xc = pca.fit_transform(scale(X))
​
first_2 = sum(pca.explained_variance_ratio_[:2]*100)
last_2 = sum(pca.explained_variance_ratio_[-2:]*100)
​
print('variance by the components 1&2: %0.1f%%' % first_2)
print('variance by the last components: %0.1f%%' % last_2)
​
df = pd.DataFrame(Xc, columns=['comp_' + str(j) 
                               for j in range(10)])
​

#Displays Outliers
fig, axes = plt.subplots(nrows=1, ncols=2, 
                         figsize=(15, 5))
first_two = df.plot.scatter(x='comp_0', y='comp_1', 
                            s=50, grid=True, c='Azure', 
                            edgecolors='DarkBlue', 
                            ax=axes[0])
last_two  = df.plot.scatter(x='comp_8', y='comp_9', 
                            s=50, grid=True, c='Azure', 
                            edgecolors='DarkBlue', 
                            ax=axes[1])
​
plt.show()
outlying = (Xc[:,-1] > 0.3) | (Xc[:,-2] > 1.0)
df[outlying]

#Using cluster analysis
from sklearn.cluster import DBSCAN
DB = DBSCAN(eps=2.5, min_samples=25)
DB.fit(Xc)
​
from collections import Counter
print(Counter(DB.labels_))
​
df[DB.labels_==-1]


#Automating outliers detection with Isolation Forests
from sklearn.ensemble import IsolationForest
auto_detection = IsolationForest(max_samples=50, 
                                 contamination=0.05,
                                 random_state=0)
auto_detection.fit(Xc)
​
evaluation = auto_detection.predict(Xc)
df[evaluation==-1]
#Surprisingly all this code works