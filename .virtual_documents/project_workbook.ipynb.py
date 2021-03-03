from IPython.display import display
from IPython.display import HTML
import IPython.core.display as di 



# This line will hide code by default when the notebook is exported as HTML
di.display_html('<script>jQuery(function() {if (jQuery("body.notebook_app").length == 0) { jQuery(".input_area").toggle(); jQuery(".prompt").toggle();}});</script>', raw=True)

# This line will add a button to toggle visibility of code blocks, for use with the HTML export version
di.display_html('''<button onclick="jQuery('.input_area').toggle(); jQuery('.prompt').toggle();">Show/hide code</button>''', raw=True)


di.display_html("""

<style>
#customers {
  font-family: "Trebuchet MS", Arial, Helvetica, sans-serif;
  border-collapse: collapse;
  width: 100get_ipython().run_line_magic(";", "")
}

#customers td, #customers th {
  border: 1px solid #ddd;
  padding: 8px;
  text-align: center;
}

.content:nth-child(even){background-color: #f2f2f2;}
.content:hover{background-color:#C7C9C7;}


#customers th {
  padding-top: 12px;
  padding-bottom: 12px;
  text-align: center;
  
  color: white;
}

.first{
    background-color: #4B6D80;
    font-size:20px;
}
.second{
    background-color: #71A4BF;
}

.third{
    background-color: #B1D0E8;
    color: white;
}

#customers a {
    color: black;
    padding: 10px 20px;
    text-align: center;
    text-decoration: none;
        text-decoration-line: none;
        text-decoration-style: solid;
        text-decoration-color: currentcolor;
        text-decoration-thickness: auto;
    display: inline-block;
    font-size: 16px;
    margin-left: 20px;
    
}

</style>

""", raw=True)



di.display_html("""
<table id="customers">
    <thead class="first">
        <th colspan=5>Table of contents</th>
    <tbody>
        <tr>
            <td colspan=5 class="cell"><a href='#Importing-Require'>Importing Require Libraries"</a></td>
        </tr>
          <tr>
            <td colspan=5 class="cell"><a href='#DataLoad'>Load</a></td>
        </tr>
        <tr>
            <td colspan=5 class="cell"><a href='#DataInsights'>Exploration Data - Data Insights</a></td>
        </tr>
         
        <tr>
            <td colspan=5 class="cell"><a href='#SummaryStatistics'>Exploration Data - Summary Statistics</a></td>
        </tr>
        <tr>
            <td colspan=5 class="cell"><a href='#DataLoad'>Data Cleaning</a></td>
        </tr>
        <tr>
            <td colspan=5 class="cell"><a href='#DataVisualization'>Data Visualization</a></td>
        </tr>
        <tr>
            <td class="cell"><a href='#missing-value'>check missing values</a></td>
            <td class="cell"><a href='#correlation'>correlation</a></td>
            <td class="cell"><a href='#'>Correlation Heat Maps - Seaborn</a></td>
            <td class="cell"><a href='#Outliers'>Outliers</a></td>
            <td class="cell"><a href='#distribution-Skewness'>distribution-Skewness</a></td>
        </tr>
        <tr>
            <td colspan=5 class="cell"><a href='#Prediction'>Prediction Age and pay - Linear Regression</a></td>
        </tr>
        <tr>
            <td colspan=5 class="cell"><a href='#Comments-on-results'>Comments on results</a></td>
        </tr>
        <tr>
            <td colspan=5 class="cell"><a href='#References'>References</a></td>
        </tr>
    </tbody>
</table>
""", raw=True)


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm # Predict
import statsmodels.formula.api as smf #Predict 

from sklearn import datasets, linear_model #Learn
from sklearn.metrics import mean_squared_error #Learn


get_ipython().run_line_magic("matplotlib", " inline")


df = pd.read_csv('dataset/credit_cards_dataset.csv',sep=',')
df.head() 


df.shape 


df.columns.values 


df.info() 


df.describe() 


df.AGE.unique() 



df.LIMIT_BAL.unique() 


df.MARRIAGE.value_counts()


# - This tells us count of each MARRIAGE score in descending order.
# - "MARRIAGE" has most values concentrated in the categories 2, 1 .
# - Only a few observations made for the categories 3 & 0 
## DATA CLEANING
### On the Dataset description , we don't have "MARRIAGE Status" = 0, so we need to clean up these values

df = df.loc[df["MARRIAGE"].isin([1,2])]

df



# Data Visualization


sns.heatmap(df.isnull(),cbar=False,yticklabels=False,cmap = 'viridis')


plt.figure(figsize=(6,4))
sns.heatmap(df.corr(),cmap='Blues',annot=False) 


plt.figure(figsize=(6,4))
sns.heatmap(df.corr(),cmap='Blues',annot=True) 


#Quality correlation matrix
k = 12 #number of variables for heatmap
cols = df.corr().nlargest(k, 'LIMIT_BAL')['LIMIT_BAL'].index
cm = df[cols].corr()
plt.figure(figsize=(10,6))
sns.heatmap(cm, annot=True, cmap = 'viridis')


l = df.columns.values
number_of_columns=12
number_of_rows = len(l)-1/number_of_columns
plt.figure(figsize=(number_of_columns,5*number_of_rows))
for i in range(0,len(l)):
    plt.subplot(number_of_rows + 1,number_of_columns,i+1)
    sns.set_style('whitegrid')
    sns.boxplot(df[l[i]],color='green',orient='v')
    plt.tight_layout()



plt.figure(figsize=(2*number_of_columns,5*number_of_rows))
for i in range(0,len(l)):
    plt.subplot(number_of_rows + 1,number_of_columns,i+1)
    sns.distplot(df[l[i]],kde=True) 



from sklearn.model_selection import train_test_split


train, test = train_test_split(df, test_size=0.2, random_state=4) 


results1 = smf.ols('AGE ~ PAY_0 + PAY_2 + PAY_3 + PAY_4 ', data=df).fit()
print(results1.summary())


y = train["AGE"]
cols = ["PAY_0","PAY_2","PAY_3","PAY_4"]

X=train[cols]


regr = linear_model.LinearRegression()
regr.fit(X,y)


ytrain_pred = regr.predict(X)
print("In-sample Mean squared error: get_ipython().run_line_magic(".2f"", "")
      % mean_squared_error(y, ytrain_pred))


ytest = test["AGE"]
cols = ["PAY_0","PAY_2","PAY_3","PAY_4"]

Xtest=test[cols]


ypred = regr.predict(Xtest)
print("Out-of-sample Mean squared error: get_ipython().run_line_magic(".2f"", "")
      % mean_squared_error(ytest, ypred))
