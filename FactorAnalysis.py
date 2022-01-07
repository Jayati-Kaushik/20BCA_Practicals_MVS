import os
import pandas as pd
from factor_analyzer import FactorAnalyzer
import seaborn as sns
import matplotlib.pyplot as plt
#from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity
from factor_analyzer.factor_analyzer import calculate_kmo

os.chdir("C:/Users/prach/OneDrive/Desktop/Khushi")
df=pd.read_csv('FIFA 2018 Statistics.csv')
#dropping the non-numeric columns 
df.drop(['Date','Team','Opponent','Man of the Match','Round'],axis=1,inplace=True) 

#drop missing values rows
df.dropna(inplace=True)
#df.fillna(0) #df.replace(np.nan,0)

df.info()

# Checking the correlation
x= df.corr(method= 'pearson')
print(x)
sns.heatmap(df.corr(method='pearson'),data=df)
plt.show()

#adequacy test
# Bartlett’s test
#chi_square_value,p_value=calculate_bartlett_sphericity(df)
#print(chi_square_value, p_value) 

# Kaiser-Meyer-Olkin (KMO) Test
kmo_all,kmo_model=calculate_kmo(df)
print(kmo_model)
# KMO values range between 0 and 1. Value of KMO less than 0.5 is considered inadequate.
# The overall KMO for our data is 0.76, which is pretty good
# This value indicates that we can proceed with our planned factor analysis.


# Choosing the Number of Factors
# Create factor analysis object and perform factor analysis
fa = FactorAnalyzer()
fa.fit(df)

#Check Eigenvalues
ev, v = fa.get_eigenvalues() #eigen_values, vectors = fa.get_eigenvalues()
print(ev) #print(vectors) #print(eigen_values)
# 3-factors eigen values are greater than 1
# we choose only 3 factors/unobserved variables


# Create scree plot
plt.scatter(range(1,df.shape[1]+1),ev)
plt.plot(range(1,df.shape[1]+1),ev)
plt.title('Scree Plot')
plt.xlabel('Factors')
plt.ylabel('Eigenvalue')
plt.grid()
plt.show()
# From the scree plot we can see that the number of factors=3 or 4.

# Create factor analysis object and perform factor analysis
fa = FactorAnalyzer()
fa.set_params(n_factors=3, rotation='varimax')
fa.fit(df)
loadings = fa.loadings_
print(loadings)


# Get variance of each factors
print(fa.get_factor_variance())
# Output is in the format:
#                   Factor 1    Factor2    Factor3 
# SS Loadings
# Proportion Var
# Cummulative Var

# Total 52% cumulative Variance is explained by the 3 factors.
