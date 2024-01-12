import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf

df = pd.read_csv("/home/aditya/Downloads/tensorflow/Udemy_Projects/HousePricePrediction_Hypothesis_Testing/real_estate.csv")
df.info()
'''
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 418 entries, 0 to 417
Data columns (total 7 columns):
 #   Column     Non-Null Count  Dtype  
---  ------     --------------  -----  
 0   age        418 non-null    float64
 1   MRT_dist   417 non-null    float64
 2   stores     417 non-null    float64
 3   latitude   418 non-null    float64
 4   longitude  418 non-null    float64
 5   price      418 non-null    float64
 6   year       418 non-null    int64  
dtypes: float64(6), int64(1)
memory usage: 23.0 KB
'''

df[df.isna().any(axis=1)]
'''
      age  MRT_dist  stores  latitude  longitude  price  year
30    7.1       NaN     3.0  24.96942  121.53764   43.2  2012
131  24.3    265.67     NaN  24.87235  121.51564   27.6  2013
'''

df.dropna(how="any", inplace=True)
df.head()
'''
    age   MRT_dist  stores  latitude  longitude  price  year
0  30.4  1735.5950     2.0  24.96464  121.51623   25.9  2012
1  32.7   392.4459     6.0  24.96398  121.54250   30.5  2012
2  15.5   815.9314     4.0  24.97886  121.53464   37.4  2012
3  34.5   623.4731     7.0  24.97933  121.53642   40.3  2012
4  23.0   130.9945     6.0  24.95663  121.53765   37.2  2012
'''

df[df.duplicated(keep=False)]
'''
      age  MRT_dist  stores  latitude  longitude  price  year
243  17.0  1485.097     4.0  24.97073    121.517   30.7  2013
252  17.0  1485.097     4.0  24.97073    121.517   30.7  2013
253  17.0  1485.097     4.0  24.97073    121.517   30.7  2013
'''

df.drop_duplicates(keep="first", inplace=True)

df.reset_index(drop=True, inplace=True)
df.info()
'''
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 414 entries, 0 to 413
Data columns (total 7 columns):
 #   Column     Non-Null Count  Dtype  
---  ------     --------------  -----  
 0   age        414 non-null    float64
 1   MRT_dist   414 non-null    float64
 2   stores     414 non-null    float64
 3   latitude   414 non-null    float64
 4   longitude  414 non-null    float64
 5   price      414 non-null    float64
 6   year       414 non-null    int64  
dtypes: float64(6), int64(1)
memory usage: 22.8 KB
'''

df.describe()

plt.figure(figsize=(12, 9))
sns.heatmap(df.corr(), vmax=0.9, square=True, annot=True, cmap="RdBu")
plt.show()

sns.pairplot(df, height=2)
plt.show()

df["invert_price"] = 1 / df["price"] * 100

sns.pairplot(df[["invert_price", "MRT_dist"]], height=3)
plt.show()

corr_matrix = df.corr()
abs(corr_matrix.loc["invert_price", "MRT_dist"]) > abs(corr_matrix.loc["price", "MRT_dist"])

sns.pairplot(df[["invert_price", "MRT_dist"]], height=3)
plt.show()

utl = df.invert_price.max()
df = df[df.invert_price < utl].copy()

model = smf.ols("invert_price ~ MRT_dist + stores + age + year + latitude + longitude", data=df)
results = model.fit()

print(results.summary())
'''
                            OLS Regression Results                            
==============================================================================
Dep. Variable:           invert_price   R-squared:                       0.743
Model:                            OLS   Adj. R-squared:                  0.739
Method:                 Least Squares   F-statistic:                     195.8
Date:                Fri, 12 Jan 2024   Prob (F-statistic):          1.71e-116
Time:                        00:01:54   Log-Likelihood:                -431.50
No. Observations:                 413   AIC:                             877.0
Df Residuals:                     406   BIC:                             905.2
Df Model:                           6                                         
Covariance Type:            nonrobust                                         
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
Intercept   1642.5058    496.579      3.308      0.001     666.319    2618.692
MRT_dist       0.0006   5.61e-05     10.021      0.000       0.000       0.001
stores        -0.0723      0.015     -4.908      0.000      -0.101      -0.043
age            0.0213      0.003      7.043      0.000       0.015       0.027
year          -0.2731      0.074     -3.674      0.000      -0.419      -0.127
latitude     -28.8408      3.486     -8.273      0.000     -35.694     -21.988
longitude     -3.0465      3.809     -0.800      0.424     -10.535       4.442
==============================================================================
Omnibus:                       71.125   Durbin-Watson:                   1.997
Prob(Omnibus):                  0.000   Jarque-Bera (JB):              544.490
Skew:                           0.457   Prob(JB):                    5.83e-119
Kurtosis:                       8.550   Cond. No.                     3.47e+07
==============================================================================
Notes:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
[2] The condition number is large, 3.47e+07. This might indicate that there are
strong multicollinearity or other numerical problems.
'''

significance_level = 0.05
results.pvalues < significance_level
'''
Intercept     True
MRT_dist      True
stores        True
age           True
year          True
latitude      True
longitude    False
dtype: bool
'''

results.condition_number > 1000
'''
True
'''

df["MRT_dist"] = df["MRT_dist"] / 1000
df["year"] = df["year"] - df["year"].min()
df["latitude"] = (df["latitude"] - df["latitude"].min()) * 100
df["longitude"] = (df["longitude"] - df["longitude"].min()) * 100
df.describe()

model = smf.ols("invert_price ~ MRT_dist + stores + age + year + latitude", data=df)
results = model.fit()
print(results.summary())
'''
                            OLS Regression Results                            
==============================================================================
Dep. Variable:           invert_price   R-squared:                       0.743
Model:                            OLS   Adj. R-squared:                  0.740
Method:                 Least Squares   F-statistic:                     235.0
Date:                Fri, 12 Jan 2024   Prob (F-statistic):          1.46e-117
Time:                        00:03:10   Log-Likelihood:                -431.83
No. Observations:                 413   AIC:                             875.7
Df Residuals:                     407   BIC:                             899.8
Df Model:                           5                                         
Covariance Type:            nonrobust                                         
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
Intercept      3.5631      0.176     20.223      0.000       3.217       3.909
MRT_dist       0.5949      0.038     15.575      0.000       0.520       0.670
stores        -0.0716      0.015     -4.872      0.000      -0.100      -0.043
age            0.0214      0.003      7.078      0.000       0.015       0.027
year          -0.2762      0.074     -3.721      0.000      -0.422      -0.130
latitude      -0.2850      0.035     -8.241      0.000      -0.353      -0.217
==============================================================================
Omnibus:                       73.048   Durbin-Watson:                   2.007
Prob(Omnibus):                  0.000   Jarque-Bera (JB):              560.981
Skew:                           0.480   Prob(JB):                    1.53e-122
Kurtosis:                       8.628   Cond. No.                         115.
==============================================================================
Notes:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
'''

results.condition_number < 1000
'''
True
'''

results.rsquared > 0.7
'''
True
'''

results.pvalues < significance_level
'''
Intercept    True
MRT_dist     True
stores       True
age          True
year         True
latitude     True
dtype: bool
'''

results.params > 0
'''
Intercept     True
MRT_dist      True
stores       False
age           True
year         False
latitude     False
dtype: bool
'''

'''
FINAL SUMMARY
    The distance to the next MRT station significantly influences house prices. 
    A lower distance leads to a higher price (most likely).

    The number of convenience stores in the living circle on foot significantly influences house prices. 
    A higher number leads to a higher price (most likely).

    The house age significantly influences house prices. 
    A lower age leads to a higher price (most likely).

    The transaction year significantly influences house prices. 
    Prices increased from 2012 to 2013. This is in line with rising house prices over time. 

    The latitude significantly influences house prices. The more to the northern part of the city, the higher the price. 
    This means, the closer to the seaside/coast, the higher the price (most likely).

    The longitude does not significantly influence house prices. 
'''