from IPython.display import display
from IPython.display import HTML
import IPython.core.display as di 
import pandas as pd
import numpy as np
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
import os
import plotly.graph_objects as go
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
from skimage import io
from scipy import stats
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV, ParameterGrid, StratifiedKFold
#from google.colab import files
import json
import random
from visualization import *

warnings.simplefilter("ignore")
scaler = MinMaxScaler()

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
            <td colspan=5 class="cell"><a href='#Description-of-the-dataset'>Description of the dataset</a></td>
        </tr>
        <tr>
            <td colspan=5 class="cell"><a href='#Data-cleaning-and-preprocessing'>Data cleaning and preprocessing</a></td>
        </tr>
        <tr>
            <td colspan=5 class="cell"><a href='#Anomaly-detection'>Anomaly detection</a></td>
        </tr>
        <tr>
            <td class="cell"><a href='#Z-score'>Z-score and Boxplot</a></td>
            <td class="cell"><a href='#Isolation-forests'>Isolation Forests</a></td>
            <td class="cell"><a href='#One-class-SVM'>One Class SVM</a></td>
            <td class="cell"><a href='#SVMDD'>SVMDD</a></td>
            <td class="cell"><a href='#Local-Outlier-Factor'>Local Outlier Factor</a></td>
        </tr>
        <tr>
            <td colspan=5 class="cell"><a href='#Distribution-and-Pairplot-matrix'>Distribution and Pairplot matrix</a></td>
        </tr>
        <tr>
            <td colspan=5 class="cell"><a href='#Correlation-among-features'>Correlation among features</a></td>
        </tr>
        <tr>
            <td colspan=5 class="cell"><a href='#Dimensionality-reduction'>Dimensionality reduction</a></td>
        </tr>
        <tr>
            <td colspan=5 class="cell"><a href='#Manage-dataset-imbalancing'>Manage dataset imbalancing</a></td>
        </tr>
        <tr>
            <td td colspan=3 class="cell"><a href='#A-variation:-k-means-SMOTE'>K-means SMOTE</a></td>
            <td td colspan=2 class="cell"><a href='#Undersampling-tecnique:-Cluster-Centroids'>Cluster Centroids</a></td>
        </tr>
        <tr>
            <td colspan=5 class="cell"><a href='#Cross-Validation'>Cross Validation</a></td>
        </tr>
        <tr>
            <td colspan=5 class="cell"><a href='#Algorithms'>Algorithms</a></td>
        </tr>
        <tr>
            <td class="cell"><a href='#Support-Vector-Machine'>Support Vector Machine</a></td>
            <td class="cell"><a href='#Decision-Tree'>Decision Tree and Random Forest</a></td>
            <td class="cell"><a href='#Ensamble-methods-and-boosting'>Ensamble methods and boosting</a></td>
            <td class="cell"><a href='#K-Nearest-neighbor'>K-Nearest neighbor</a></td>
            <td class="cell"><a href='#Logistic-regression'>Logistic regression</a></td>
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


path = 'dataset/cleaned_credit_card.csv'
df = pd.read_csv(path).rename(columns={'PAY_0': 'PAY_1'}) # wrong column name PAY_0 setted to PAY_1
df = df.drop(axis=1, columns='ID')
df.reset_index()
df['default.payment.next.month'] = df['default.payment.next.month'].astype('category')
df.head()


s = ""
for i, n in df.isnull().sum().items():
    s+="<tr><td>" + i + "</td><td>" + str(n) + "</td></tr>"
    
# This line will hide code by default when the notebook is exported as HTML
di.display_html("""

<table>
    <thead>
        <th>Attribute</th>
        <th>Number of null or nan values</th>
    </thead>
    <tbody>
        """ + s + """
    </tbody>

</table>



""", raw=True)



df[['LIMIT_BAL','SEX', 'EDUCATION', 'MARRIAGE', 'AGE']].describe()


summary = df['EDUCATION'].value_counts()
plot_bar(summary, 'EDUCATION', 'Number of rows for each EDUCATION category')

m = (df['EDUCATION'] == 0)|(df['EDUCATION'] == 6)|(df['EDUCATION'] == 5)
df = df.drop(df.EDUCATION[m].index.values, axis=0)

summary = df['EDUCATION'].value_counts()
plot_bar(summary, 'EDUCATION', 'Number of rows for each EDUCATION category with pre-processing')


summary = df['MARRIAGE'].value_counts()
plot_bar(summary, 'MARRIAGE', 'Number of rows for each MARRIAGE category')

m = (df['MARRIAGE'] == 0)
df = df.drop(df.MARRIAGE[m].index.values, axis=0)

summary = df['MARRIAGE'].value_counts()
plot_bar(summary, 'MARRIAGE', 'Number of rows for each MARRIAGE category with pre-processing')


df[['PAY_' + str(n) for n in range(1, 7)]] += 1
df[['PAY_' + str(n) for n in range(1, 7)]].describe()


df['EDUCATION'] = df['EDUCATION'].astype('category')
df['SEX'] = df['SEX'].astype('category')
df['MARRIAGE'] = df['MARRIAGE'].astype('category')

df=pd.concat([pd.get_dummies(df['EDUCATION'], prefix='EDUCATION'), 
                  pd.get_dummies(df['SEX'], prefix='SEX'), 
                  pd.get_dummies(df['MARRIAGE'], prefix='MARRIAGE'),
                  df],axis=1)
df.drop(['EDUCATION'],axis=1, inplace=True)
df.drop(['SEX'],axis=1, inplace=True)
df.drop(['MARRIAGE'],axis=1, inplace=True)
df.head()


scaler = MinMaxScaler()
df['LIMIT_BAL'] = scaler.fit_transform(df['LIMIT_BAL'].values.reshape(-1, 1))
df['AGE'] = scaler.fit_transform(df['AGE'].values.reshape(-1, 1))


for i in range(1,7):
    scaler = MinMaxScaler()
    df['BILL_AMT' + str(i)] = scaler.fit_transform(df['BILL_AMT' + str(i)].values.reshape(-1, 1))

for i in range(1,7):
    scaler = MinMaxScaler()
    df['PAY_AMT' + str(i)] = scaler.fit_transform(df['PAY_AMT' + str(i)].values.reshape(-1, 1))
    
for i in range(1,7):
    scaler = MinMaxScaler()
    df['PAY_' + str(i)] = scaler.fit_transform(df['PAY_' + str(i)].values.reshape(-1, 1))


from sklearn.model_selection import train_test_split

X_train_val, X_test, y_train_val, y_test = train_test_split(df[df.columns[:-1]], df['default.payment.next.month'], test_size=0.25, stratify=df['default.payment.next.month'])



from scipy import stats

figs, axs= plt.subplots(5, 3, figsize=(15, 14))

i, j = 0, 0
d1 = ['BILL_AMT' + str(i) for i in range(1, 7)]
d2 = ['PAY_AMT' + str(i) for i in range(1, 7)]
d =  ['LIMIT_BAL', 'AGE'] + d1 + d2
for attribute in d:
    if j == 3:
        j = 0
        i = i+1
    stats.probplot(
        X_train_val[attribute], 
        dist="norm", 
        sparams = (X_train_val[attribute].mean(), X_train_val[attribute].std()),
        plot=axs[i, j]
    )
    
    axs[i, j].get_lines()[0].set_marker('.')
    axs[i, j].get_lines()[0].set_color('sandybrown')
    axs[i, j].get_lines()[0].set_markersize(1.0)
    axs[i, j].set_title('Probability plot for attribute ' + attribute)
    axs[i, j].grid()
    axs[i, j].get_lines()[1].set_linewidth(3.0)
    axs[i, j].get_lines()[1].set_color('darkseagreen')
    j = j+1
    
figs.tight_layout()
axs[4, 2].set_visible(False)
plt.show()



def detectOutliers(X, outlierConstant=1.5):
    a = np.array(X)
    upper_quartile = np.percentile(a, 75)
    lower_quartile = np.percentile(a, 25)
    IQR = (upper_quartile - lower_quartile) * outlierConstant
    quartileSet = (lower_quartile - IQR, upper_quartile + IQR)
    resultList = []
    for y in a.tolist():
      if y >= quartileSet[0] and y <= quartileSet[1]:
          resultList.append(1)
      else:
          resultList.append(-1)
    return np.array(resultList)


d1 = ['BILL_AMT' + str(i) for i in range(1, 7)]
d2 = ['PAY_AMT' + str(i) for i in range(1, 7)]
d = ['LIMIT_BAL', 'AGE'] + d1 + d2


data = pd.concat([y_train_val, X_train_val[d]], axis=1)
data = pd.melt(
    data,
    id_vars="default.payment.next.month",
    var_name = "features",
    value_name = "value",
)

fig = px.box(
    data,
    x = "features",
    y = "value",
    color = "default.payment.next.month",
    color_discrete_sequence=px.colors.qualitative.Set2,
)
fig.update_layout(
    xaxis_title= "Features",
    yaxis_title= "",
    title='Boxplot for the different attributes'
)
fig.show()


X_train_val['default.payment.next.month'] = y_train_val
#plot_boxplot('LIMIT_BAL', X_train_val)

box_outliers = detectOutliers(X_train_val['LIMIT_BAL'] )

for i in range(1,7):
  #plot_boxplot('BILL_AMT' + str(i), X_train_val)
  new_outliers = detectOutliers(X_train_val['BILL_AMT' + str(i)] )
  mask = np.array((box_outliers == -1)&(new_outliers == -1))
  box_outliers[mask == True] = -1
  box_outliers[mask == False] = 1
  


for i in range(1,7):
  #plot_boxplot('PAY_AMT' + str(i), X_train_val)
  new_outliers = detectOutliers(X_train_val['PAY_AMT' + str(i)] )
  mask = np.array((box_outliers == -1)&(new_outliers == -1))
  box_outliers[mask == True] = -1
  box_outliers[mask == False] = 1

X_train_val.drop('default.payment.next.month', axis=1, inplace=True)


from sklearn.ensemble import IsolationForest

isolation_forest = IsolationForest()
is_outliers = isolation_forest.fit_predict(X_train_val)
score_sample =isolation_forest.score_samples(X_train_val)
offset = isolation_forest.offset_
title = "Score of each data point in Isolation forest"
x_position_outlier_ann = 487
y_position_outlier_ann = -0.57

plot_score_outliers(score_sample, offset, title, x_position_outlier_ann, y_position_outlier_ann)


from sklearn.svm import OneClassSVM
nu = 0.1
one_class_svm = OneClassSVM(nu = nu, kernel='poly')
ocsvm_outliers = one_class_svm.fit_predict(X_train_val)


from sklearn.svm import OneClassSVM
nu = 0.2
one_class_svm = OneClassSVM(nu = nu)
svdd_outliers = one_class_svm.fit_predict(X_train_val)


from sklearn.neighbors import LocalOutlierFactor
lof = LocalOutlierFactor(n_neighbors=200)
lof_outliers = lof.fit_predict(X_train_val)
score_sample = lof.negative_outlier_factor_
offset = lof.offset_

title = "Score of each data point in Local Outlier Factor"
x_position_outlier_ann = 896
y_position_outlier_ann = -1.62

plot_score_outliers(score_sample, offset, title, x_position_outlier_ann, y_position_outlier_ann)


num_is_outliers = len(np.where(is_outliers == -1)[0])
num_oc_outliers = len(np.where(ocsvm_outliers == -1)[0])
num_svdd_outliers = len(np.where(svdd_outliers == -1)[0])
num_lof_outliers = len(np.where(lof_outliers == -1)[0])

s = f'Outliers with Isolation Forest: {num_is_outliers}<br>Outliers with One Class SVM: {num_oc_outliers}'
s += f'<br>Outliers with LOF: {num_lof_outliers}'
s += f'<br>Outliers with SVMDD: {num_svdd_outliers}'

mask = (is_outliers == -1)&(ocsvm_outliers == -1)&(lof_outliers == -1)
#mask = (is_outliers == -1)&(ocsvm_outliers == -1)&(lof_outliers == -1)&(svdd_outliers == -1)

common_outlier = np.sum(mask)
s += f'<br><br>The previous algorithms indentify {common_outlier} outliers in common.'
di.display_html("""

<p style='margin-bottom: 1em;font-size:15px'>
    """ + s + """
</p>
""", raw=True)
X_train_val['default.payment.next.month'] = y_train_val

X_train_val.drop(X_train_val[mask].index, axis=0, inplace=True)
y_train_val = X_train_val['default.payment.next.month']
X_train_val.drop('default.payment.next.month', axis=1, inplace=True)


#d1 = ['PAY_' + str(n) for n in range(1, 7)]
d2 = ['BILL_AMT' + str(n) for n in range(1, 7)]
d3 = ['PAY_AMT' + str(n) for n in range(1, 7)]
dimensions = d2 + d3 +['LIMIT_BAL','AGE','default.payment.next.month']
X_train_val['default.payment.next.month'] = y_train_val
pairpl = sns.pairplot(X_train_val[dimensions], hue='default.payment.next.month', diag_kind='kde', corner=True);
pairpl._legend.remove()


X_train_val.drop('default.payment.next.month', axis=1, inplace=True)


dimensions = ['PAY_' + str(n) for n in range(1, 7)] + dimensions
corr = X_train_val[dimensions[:-1]].corr()
fig, ax = plt.subplots(figsize=(12,12)) 
sns.heatmap(
    corr, 
    vmin = -1, 
    vmax = 1, 
    center = 0,
    cmap = sns.diverging_palette(220, 20, n=200),
    square = True,
    ax = ax,
    annot = True,
    cbar = False

)
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right'
);



from sklearn.decomposition import PCA

pc = 11
explainend_var = []

pca = PCA(n_components=len(X_train_val.columns.values))
pca.fit(X_train_val)

fig = go.Figure()
fig.update_layout(
    title = "Cumulative and variance explained with different number of component",
    xaxis_title= "Number of principal component",
    yaxis_title= "Variance explained",
)


fig.add_annotation(
    x=pc,
    y=0,
    text = 'Principal component '+ str(pc),
    showarrow=True,
    arrowsize=1,
)
fig.add_annotation(
    x=pc,
    y=1,
    text = 'Principal component '+ str(pc),
    showarrow=True,
    arrowsize=1,
    
)


fig.add_trace(
    go.Scatter(
        x=[i for i in range(1, len(X_train_val.columns.values))], 
        y=np.cumsum(pca.explained_variance_ratio_),
        mode="lines+markers+text",
        #text=[round(v*100, 1) for v in np.cumsum(pca.explained_variance_ratio_)],
        name='total variance explained',
        textposition="bottom center",
        hovertemplate = "Cumulative explained varianceget_ipython().run_line_magic("{y:.2f}<br>Principal", " Component %{x})\",")
        textfont=dict(
        size=12,
        ),
        marker=dict(
          color='rgb(229,196,148)',
          size=8,
          ),
        line = dict(
            width=4
            )
        )
    )

fig.add_trace(
    go.Scatter(
        x=[i for i in range(1, len(X_train_val.columns.values))], 
        y=pca.explained_variance_ratio_,
        mode="lines+markers",
        name='variance explained by the single component',
        
        marker=dict(
          color='rgb(102,194,165)',
          size=8,
          ),
        line = dict(
            width=4
            )  
    )
)


fig.update_xaxes(showspikes=True)
fig.update_yaxes(showspikes=True)



fig.show()


pca = PCA(n_components=pc)
pca.fit(X_train_val)
X_train_val.index = pd.RangeIndex(start=0, stop=len(X_train_val), step=1)
X_15d_train_val = pd.DataFrame(pca.transform(X_train_val))
X_15d_test = pd.DataFrame(pca.transform(X_test))
X_15d_train_val.columns = ['PC' + str(i) for i in range(1, pc+1) ]
X_15d_test.columns = ['PC' + str(i) for i in range(1, pc+1) ]
X_15d_train_val.head()


l0 = y_train_val[y_train_val== 0].count()
l1 = y_train_val[y_train_val== 1].count()

s = f"There are: <ul><li>{l0} rows labelled with 0 (<b>{round(l0/(l1+l0)*100)}get_ipython().run_line_magic("</b>);</li><li>", " {l1} rows labelled with 1 (<b>{round(l1/(l1+l0)*100)}%</b>).</li></ul>\"")


di.display_html("""

<p style='margin-bottom: 1em;font-size:15px'>
    """ + s + """
</p>
""", raw=True)


"""if not os.path.isdir('imbalancedlearn'):
  get_ipython().getoutput("git clone https://github.com/scikit-learn-contrib/imbalanced-learn.git")
  get_ipython().getoutput("mv 'imbalanced-learn' 'imbalancedlearn'")
  from imalancedlearn.under_sampling import KMeansSMOTE
  """


from imblearn.over_sampling import SMOTE , KMeansSMOTE

def oversample_dataset(X_train, y_train):

    s = f"<br>Number of instances in the training set before the rebalancing operation: {len(X_train)}"
    #oversample = SMOTE()
    oversample = KMeansSMOTE(cluster_balance_threshold=0.00001)
    X_train_smote, y_train_smote = oversample.fit_resample(X_train, y_train)

    s += f"<br>Number of instances in the training set after the rebalancing operation: {len(X_train_smote)}"
    
    l0 = len(y_train_smote[y_train_smote == 0])
    l1 = len(y_train_smote[y_train_smote == 1])
    
    s += f"<br>There are {l0} rows labelled with 0 ({round(l0/(l1+l0)*100)}get_ipython().run_line_magic("),", " {l1} rows labelled with 1 ({round(l1/(l1+l0)*100)}%)\"")
    return X_train_smote, y_train_smote, s



from imblearn.under_sampling import ClusterCentroids

def undersample_dataset(X_train, y_train):

    s = f"<br>Number of instances in the training set before the rebalancing operation: {len(X_train)}"
    oversample = ClusterCentroids()
    
    X_train_cc, y_train_cc = oversample.fit_resample(X_train, y_train)

    s += f"<br>Number of instances in the training set after the rebalancing operation: {len(X_train_cc)}"
    
    l0 = len(y_train_cc[y_train_cc == 0])
    l1 = len(y_train_cc[y_train_cc == 1])
    
    s += f"<br>There are {l0} rows labelled with 0 ({round(l0/(l1+l0)*100)}get_ipython().run_line_magic("),", " {l1} rows labelled with 1 ({round(l1/(l1+l0)*100)}%)\"")
    return X_train_cc, y_train_cc, s


def train_and_validate(X_train_val, y_train_val, classifier, clf_name, parameter_grid, K = 5, oversampling=True):
    results = []
    parameters = []
    s = ""
    for params in ParameterGrid(parameter_grid):
        fold = 1
        s += f'Training parameters: {params}'

        temp_results = {}

        kfold = StratifiedKFold(n_splits=K)
        for train_index, val_index in kfold.split(X_train_val, y_train_val):
            # define the training, validation and test set
            #print(f'Training on {fold} fold')
            fold +=1
            X_train, X_val = X_train_val.values[train_index], X_train_val.values[val_index]
            y_train, y_val = y_train_val.values[train_index], y_train_val.values[val_index]
            
            if oversampling:
                # oversample the training set only 
                X_train_balanced, y_train_balanced, s_balanced = oversample_dataset(X_train = X_train, y_train = y_train)
            else:
                # undersample the training set only 
                X_train_balanced, y_train_balanced, s_balanced = undersample_dataset(X_train = X_train, y_train = y_train)
            
            s+=s_balanced
            s+= '<br>Partial accuracies: '
            # fit the model
            clf = classifier(**params)
            clf.fit(X_train_balanced, y_train_balanced)

            # evaluate on the validation set
            y_pred = clf.predict(X_val)

            report = classification_report(y_val, y_pred, output_dict=True)
            if fold == 2:
                temp_results['accuracy'] = []
                for label, metrics in report.items():
                    if not isinstance(metrics, float):
                        for name, score in metrics.items():
                            temp_results[str(label) + "_" + name] = []
                    else:
                        temp_results[str(label)] = []

            for label, metrics in report.items():
                if not isinstance(metrics, float):
                    for name, score in metrics.items():
                        temp_results[str(label) + "_" + name].append(score)

            accuracy = accuracy_score(y_val, y_pred)
            s += f'{accuracy} '
            temp_results['accuracy'].append(accuracy)


        mean_score = {}
        
        for name, scores in temp_results.items():
            mean_score[name] = np.mean(scores)
            mean_score['std_' + name] = np.std(scores)
            #print(f'{name}: {np.mean(scores)}')
        
        s += f'<br>Mean accuracy: {mean_score["accuracy"]}'
        s+='<br><br>'
        results.append(mean_score)
        parameters.append(params)

    if oversampling:
        name = './results_oversampling/' + clf_name + '_results_train.json'
    else:
        name = './results_undersampling/' + clf_name + '_results_train.json'
    with open(name, 'w') as f:
        json.dump([results, parameters] , f)

    #files.download(clf_name + '_results_train.json')
    return results, parameters, s


def find_best_configuration(results, parameters, display=True):
    best_f1_1 = 0
    best_f1_0 = 0
    best_accuracy = 0
    best_configuration = parameters[0]

    for i, result in enumerate(results):
        if result['1_f1-score'] > best_f1_1:
            best_f1_1 = result['1_f1-score']
            std_f1_1 = result['std_1_f1-score']
            best_f1_0 = result['0_f1-score']
            std_f1_0 = result['std_0_f1-score']
            best_accuracy = result['accuracy']
            std_accuracy = result['std_accuracy']
            best_configuration = parameters[i]
    s = ""
    if display:
        s = f'Best configuration on validation set: {best_configuration}<br>'
        s += f'f1-score on validation set: {best_f1_0} (0), {best_f1_1} (1)<br>'
        s += f'Accuracy score on validation: {best_accuracy}<br>'
        
    return best_accuracy, best_f1_0, best_f1_1, best_configuration, s, std_f1_1, std_f1_0, std_accuracy


def test(X_train_val, X_test, y_test, classifier, clf_name, results, parameters, oversampling=True):
    best_accuracy, best_f1_0, best_f1_1, best_configuration, s,_, _, _ = find_best_configuration(results, parameters)
    clf = classifier(**best_configuration)
    if oversampling:
        X_train_val_balanced, y_train_val_balanced, _  = oversample_dataset(X_train_val, y_train_val)
    else:
        X_train_val_balanced, y_train_val_balanced, _  = undersample_dataset(X_train_val, y_train_val)
        
    clf.fit(X_train_val_balanced, y_train_val_balanced)

    y_pred = clf.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)

    precision, recall, fscore, _ = precision_recall_fscore_support(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    s += f"<br>Accuracy on the test set: {accuracy}"
    results_test = {
      'accuracy':accuracy, 
      'precision':list(precision), 
      'recall':list(recall), 
      'fscore':list(fscore)
      }
    
    if oversampling:
        name = './results_oversampling/' + clf_name + '_results_test.json'
    else:
        name = './results_undersampling/' + clf_name + '_results_test.json'
    with open(name, 'w') as f:
        json.dump([results_test], f)

    #files.download(clf_name + '_results_test.json')
    plot_confusion_matrix(y_test, y_pred)
    return s, report


from sklearn.svm import SVC
for oversampling in [True, False]:
    parameter_grid = {
        'C': [100 ,0.1, 1, 10],
        'kernel': ['rbf', 'poly'],
        'gamma': [0.0001, 0.001, 0.01]
    }
    K = 5
    classifier = SVC
    results_svm, parameter_svm, s = train_and_validate(X_15d_train_val, y_train_val, classifier, 'svm', parameter_grid, oversampling=oversampling)


    if oversampling:
        di.display_html("""
        <h1>With oversampling - SMOTE</h1>
        """, raw=True)
    else:
        di.display_html("""
        <h1>With undersampling - Cluster Centroid</h1>
        """, raw=True)
        
    #di.display_html("""
    #<p style='margin-bottom: 1em;font-size:15px'>
    #    """ + s + """
    #</p>
    #""", raw=True)
    
    classifier = SVC #{'C': 1, 'gamma': 0.01, 'kernel': 'poly'}7963817277250114
    clf_name = 'svm'
    results = results_svm
    parameters = parameter_svm
    s, report = test(X_15d_train_val, X_15d_test, y_test, classifier, clf_name, results, parameters,  oversampling=oversampling)
    
    print_result(s, report)


di.display_html("""
        
<h2>With SMOTE</h2>
<div>
        
        
    <div id="9a5e763d-1034-4f1e-aec9-c2bd6e400485" class="plotly-graph-div js-plotly-plot" style="height:525px; width:100get_ipython().run_line_magic(";"><div", " class=\"plot-container plotly\"><div class=\"svg-container\" style=\"position: relative; width: 100%; height: 100%;\"><svg class=\"main-svg\" xmlns=\"http://www.w3.org/2000/svg\" xmlns:xlink=\"http://www.w3.org/1999/xlink\" width=\"985.533\" height=\"525\" style=\"background: white none repeat scroll 0% 0%;\"><defs id=\"defs-e207b3\"><g class=\"clips\"><clipPath id=\"clipe207b3xyplot\" class=\"plotclip\"><rect width=\"826\" height=\"385\"></rect></clipPath><clipPath class=\"axesclip\" id=\"clipe207b3x\"><rect x=\"80\" y=\"0\" width=\"826\" height=\"525\"></rect></clipPath><clipPath class=\"axesclip\" id=\"clipe207b3y\"><rect x=\"0\" y=\"60\" width=\"985.533\" height=\"385\"></rect></clipPath><clipPath class=\"axesclip\" id=\"clipe207b3xy\"><rect x=\"80\" y=\"60\" width=\"826\" height=\"385\"></rect></clipPath></g><g class=\"gradients\"></g></defs><g class=\"bglayer\"><rect class=\"bg\" x=\"80\" y=\"60\" width=\"826\" height=\"385\" style=\"fill: rgb(229, 236, 246); fill-opacity: 1; stroke-width: 0px;\"></rect></g><g class=\"draglayer cursor-crosshair\"><g class=\"xy\"><rect class=\"nsewdrag drag\" style=\"fill: transparent; stroke-width: 0px; pointer-events: all;\" data-subplot=\"xy\" x=\"80\" y=\"60\" width=\"826\" height=\"385\"></rect><rect class=\"nwdrag drag cursor-nw-resize\" style=\"fill: transparent; stroke-width: 0px; pointer-events: all;\" data-subplot=\"xy\" x=\"60\" y=\"40\" width=\"20\" height=\"20\"></rect><rect class=\"nedrag drag cursor-ne-resize\" style=\"fill: transparent; stroke-width: 0px; pointer-events: all;\" data-subplot=\"xy\" x=\"906\" y=\"40\" width=\"20\" height=\"20\"></rect><rect class=\"swdrag drag cursor-sw-resize\" style=\"fill: transparent; stroke-width: 0px; pointer-events: all;\" data-subplot=\"xy\" x=\"60\" y=\"445\" width=\"20\" height=\"20\"></rect><rect class=\"sedrag drag cursor-se-resize\" style=\"fill: transparent; stroke-width: 0px; pointer-events: all;\" data-subplot=\"xy\" x=\"906\" y=\"445\" width=\"20\" height=\"20\"></rect><rect class=\"ewdrag drag cursor-ew-resize\" style=\"fill: transparent; stroke-width: 0px; pointer-events: all;\" data-subplot=\"xy\" x=\"162.60000000000002\" y=\"445.5\" width=\"660.8000000000001\" height=\"20\"></rect><rect class=\"wdrag drag cursor-w-resize\" style=\"fill: transparent; stroke-width: 0px; pointer-events: all;\" data-subplot=\"xy\" x=\"80\" y=\"445.5\" width=\"82.60000000000001\" height=\"20\"></rect><rect class=\"edrag drag cursor-e-resize\" style=\"fill: transparent; stroke-width: 0px; pointer-events: all;\" data-subplot=\"xy\" x=\"823.4\" y=\"445.5\" width=\"82.60000000000001\" height=\"20\"></rect><rect class=\"nsdrag drag cursor-ns-resize\" style=\"fill: transparent; stroke-width: 0px; pointer-events: all;\" data-subplot=\"xy\" x=\"59.5\" y=\"98.5\" width=\"20\" height=\"308\"></rect><rect class=\"sdrag drag cursor-s-resize\" style=\"fill: transparent; stroke-width: 0px; pointer-events: all;\" data-subplot=\"xy\" x=\"59.5\" y=\"406.5\" width=\"20\" height=\"38.5\"></rect><rect class=\"ndrag drag cursor-n-resize\" style=\"fill: transparent; stroke-width: 0px; pointer-events: all;\" data-subplot=\"xy\" x=\"59.5\" y=\"60\" width=\"20\" height=\"38.5\"></rect></g></g><g class=\"layer-below\"><g class=\"imagelayer\"></g><g class=\"shapelayer\"></g></g><g class=\"cartesianlayer\"><g class=\"subplot xy\"><g class=\"layer-subplot\"><g class=\"shapelayer\"></g><g class=\"imagelayer\"></g></g><g class=\"gridlayer\"><g class=\"x\"><path class=\"xgrid crisp\" transform=\"translate(117.55,0)\" d=\"M0,60v385\" style=\"stroke: rgb(255, 255, 255); stroke-opacity: 1; stroke-width: 1px;\"></path><path class=\"xgrid crisp\" transform=\"translate(192.64,0)\" d=\"M0,60v385\" style=\"stroke: rgb(255, 255, 255); stroke-opacity: 1; stroke-width: 1px;\"></path><path class=\"xgrid crisp\" transform=\"translate(267.73,0)\" d=\"M0,60v385\" style=\"stroke: rgb(255, 255, 255); stroke-opacity: 1; stroke-width: 1px;\"></path><path class=\"xgrid crisp\" transform=\"translate(342.82,0)\" d=\"M0,60v385\" style=\"stroke: rgb(255, 255, 255); stroke-opacity: 1; stroke-width: 1px;\"></path><path class=\"xgrid crisp\" transform=\"translate(417.91,0)\" d=\"M0,60v385\" style=\"stroke: rgb(255, 255, 255); stroke-opacity: 1; stroke-width: 1px;\"></path><path class=\"xgrid crisp\" transform=\"translate(493,0)\" d=\"M0,60v385\" style=\"stroke: rgb(255, 255, 255); stroke-opacity: 1; stroke-width: 1px;\"></path><path class=\"xgrid crisp\" transform=\"translate(568.0899999999999,0)\" d=\"M0,60v385\" style=\"stroke: rgb(255, 255, 255); stroke-opacity: 1; stroke-width: 1px;\"></path><path class=\"xgrid crisp\" transform=\"translate(643.18,0)\" d=\"M0,60v385\" style=\"stroke: rgb(255, 255, 255); stroke-opacity: 1; stroke-width: 1px;\"></path><path class=\"xgrid crisp\" transform=\"translate(718.27,0)\" d=\"M0,60v385\" style=\"stroke: rgb(255, 255, 255); stroke-opacity: 1; stroke-width: 1px;\"></path><path class=\"xgrid crisp\" transform=\"translate(793.36,0)\" d=\"M0,60v385\" style=\"stroke: rgb(255, 255, 255); stroke-opacity: 1; stroke-width: 1px;\"></path><path class=\"xgrid crisp\" transform=\"translate(868.45,0)\" d=\"M0,60v385\" style=\"stroke: rgb(255, 255, 255); stroke-opacity: 1; stroke-width: 1px;\"></path></g><g class=\"y\"><path class=\"ygrid crisp\" transform=\"translate(0,401.71)\" d=\"M80,0h826\" style=\"stroke: rgb(255, 255, 255); stroke-opacity: 1; stroke-width: 1px;\"></path><path class=\"ygrid crisp\" transform=\"translate(0,358.41)\" d=\"M80,0h826\" style=\"stroke: rgb(255, 255, 255); stroke-opacity: 1; stroke-width: 1px;\"></path><path class=\"ygrid crisp\" transform=\"translate(0,315.12)\" d=\"M80,0h826\" style=\"stroke: rgb(255, 255, 255); stroke-opacity: 1; stroke-width: 1px;\"></path><path class=\"ygrid crisp\" transform=\"translate(0,271.82)\" d=\"M80,0h826\" style=\"stroke: rgb(255, 255, 255); stroke-opacity: 1; stroke-width: 1px;\"></path><path class=\"ygrid crisp\" transform=\"translate(0,228.53)\" d=\"M80,0h826\" style=\"stroke: rgb(255, 255, 255); stroke-opacity: 1; stroke-width: 1px;\"></path><path class=\"ygrid crisp\" transform=\"translate(0,185.23000000000002)\" d=\"M80,0h826\" style=\"stroke: rgb(255, 255, 255); stroke-opacity: 1; stroke-width: 1px;\"></path><path class=\"ygrid crisp\" transform=\"translate(0,141.94)\" d=\"M80,0h826\" style=\"stroke: rgb(255, 255, 255); stroke-opacity: 1; stroke-width: 1px;\"></path><path class=\"ygrid crisp\" transform=\"translate(0,98.65)\" d=\"M80,0h826\" style=\"stroke: rgb(255, 255, 255); stroke-opacity: 1; stroke-width: 1px;\"></path></g></g><g class=\"zerolinelayer\"><path class=\"yzl zl crisp\" transform=\"translate(0,445)\" d=\"M80,0h826\" style=\"stroke: rgb(255, 255, 255); stroke-opacity: 1; stroke-width: 2px;\"></path></g><path class=\"xlines-below\"></path><path class=\"ylines-below\"></path><g class=\"overlines-below\"></g><g class=\"xaxislayer-below\"></g><g class=\"yaxislayer-below\"></g><g class=\"overaxes-below\"></g><g class=\"plot\" transform=\"translate(80, 60)\" clip-path=\"url('#clipe207b3xyplot')\"><g class=\"barlayer mlayer\"><g class=\"trace bars\" style=\"opacity: 1;\"><g class=\"points\"><g class=\"point\"><path style=\"vector-effect: non-scaling-stroke; opacity: 1; stroke-width: 0px; fill: rgb(102, 194, 165); fill-opacity: 1;\" d=\"M7.51,385V19.25H67.58V385Z\"></path></g></g></g><g class=\"trace bars\" style=\"opacity: 1;\"><g class=\"points\"><g class=\"point\"><path style=\"vector-effect: non-scaling-stroke; opacity: 1; stroke-width: 0px; fill: rgb(252, 141, 98); fill-opacity: 1;\" d=\"M82.6,385V160.54H142.67V385Z\"></path></g></g></g><g class=\"trace bars\" style=\"opacity: 1;\"><g class=\"points\"><g class=\"point\"><path style=\"vector-effect: non-scaling-stroke; opacity: 1; stroke-width: 0px; fill: rgb(141, 160, 203); fill-opacity: 1;\" d=\"M157.69,385V181.97H217.76V385Z\"></path></g></g></g><g class=\"trace bars\" style=\"opacity: 1;\"><g class=\"points\"><g class=\"point\"><path style=\"vector-effect: non-scaling-stroke; opacity: 1; stroke-width: 0px; fill: rgb(231, 138, 195); fill-opacity: 1;\" d=\"M232.78,385V184.89H292.85V385Z\"></path></g></g></g><g class=\"trace bars\" style=\"opacity: 1;\"><g class=\"points\"><g class=\"point\"><path style=\"vector-effect: non-scaling-stroke; opacity: 1; stroke-width: 0px; fill: rgb(166, 216, 84); fill-opacity: 1;\" d=\"M307.87,385V195H367.95V385Z\"></path></g></g></g><g class=\"trace bars\" style=\"opacity: 1;\"><g class=\"points\"><g class=\"point\"><path style=\"vector-effect: non-scaling-stroke; opacity: 1; stroke-width: 0px; fill: rgb(255, 217, 47); fill-opacity: 1;\" d=\"M382.96,385V208.03H443.04V385Z\"></path></g></g></g><g class=\"trace bars\" style=\"opacity: 1;\"><g class=\"points\"><g class=\"point\"><path style=\"vector-effect: non-scaling-stroke; opacity: 1; stroke-width: 0px; fill: rgb(229, 196, 148); fill-opacity: 1;\" d=\"M458.05,385V210.91H518.13V385Z\"></path></g></g></g><g class=\"trace bars\" style=\"opacity: 1;\"><g class=\"points\"><g class=\"point\"><path style=\"vector-effect: non-scaling-stroke; opacity: 1; stroke-width: 0px; fill: rgb(179, 179, 179); fill-opacity: 1;\" d=\"M533.15,385V222.15H593.22V385Z\"></path></g></g></g><g class=\"trace bars\" style=\"opacity: 1;\"><g class=\"points\"><g class=\"point\"><path style=\"vector-effect: non-scaling-stroke; opacity: 1; stroke-width: 0px; fill: rgb(102, 194, 165); fill-opacity: 1;\" d=\"M608.24,385V228.29H668.31V385Z\"></path></g></g></g><g class=\"trace bars\" style=\"opacity: 1;\"><g class=\"points\"><g class=\"point\"><path style=\"vector-effect: non-scaling-stroke; opacity: 1; stroke-width: 0px; fill: rgb(252, 141, 98); fill-opacity: 1;\" d=\"M683.33,385V229.45H743.4V385Z\"></path></g></g></g><g class=\"trace bars\" style=\"opacity: 1;\"><g class=\"points\"><g class=\"point\"><path style=\"vector-effect: non-scaling-stroke; opacity: 1; stroke-width: 0px; fill: rgb(141, 160, 203); fill-opacity: 1;\" d=\"M758.42,385V229.81H818.49V385Z\"></path></g></g></g></g></g><g class=\"overplot\"></g><path class=\"xlines-above crisp\" style=\"fill: none; stroke-width: 1px; stroke: rgb(255, 255, 255); stroke-opacity: 1;\" d=\"M79,445.5H906\"></path><path class=\"ylines-above crisp\" style=\"fill: none; stroke-width: 1px; stroke: rgb(255, 255, 255); stroke-opacity: 1;\" d=\"M79.5,60V445\"></path><g class=\"overlines-above\"></g><g class=\"xaxislayer-above\"><g class=\"xtick\"><text text-anchor=\"middle\" x=\"0\" y=\"460.4\" style=\"font-family: &quot;Open Sans&quot;, verdana, arial, sans-serif; font-size: 12px; fill: rgb(42, 63, 95); fill-opacity: 1; white-space: pre;\" data-unformatted=\"PC1\" data-math=\"N\" transform=\"translate(117.55,0)\">PC1</text></g><g class=\"xtick\"><text text-anchor=\"middle\" x=\"0\" y=\"460.4\" style=\"font-family: &quot;Open Sans&quot;, verdana, arial, sans-serif; font-size: 12px; fill: rgb(42, 63, 95); fill-opacity: 1; white-space: pre;\" data-unformatted=\"PC2\" data-math=\"N\" transform=\"translate(192.64,0)\">PC2</text></g><g class=\"xtick\"><text text-anchor=\"middle\" x=\"0\" y=\"460.4\" style=\"font-family: &quot;Open Sans&quot;, verdana, arial, sans-serif; font-size: 12px; fill: rgb(42, 63, 95); fill-opacity: 1; white-space: pre;\" data-unformatted=\"PC8\" data-math=\"N\" transform=\"translate(267.73,0)\">PC8</text></g><g class=\"xtick\"><text text-anchor=\"middle\" x=\"0\" y=\"460.4\" style=\"font-family: &quot;Open Sans&quot;, verdana, arial, sans-serif; font-size: 12px; fill: rgb(42, 63, 95); fill-opacity: 1; white-space: pre;\" data-unformatted=\"PC3\" data-math=\"N\" transform=\"translate(342.82,0)\">PC3</text></g><g class=\"xtick\"><text text-anchor=\"middle\" x=\"0\" y=\"460.4\" style=\"font-family: &quot;Open Sans&quot;, verdana, arial, sans-serif; font-size: 12px; fill: rgb(42, 63, 95); fill-opacity: 1; white-space: pre;\" data-unformatted=\"PC11\" data-math=\"N\" transform=\"translate(417.91,0)\">PC11</text></g><g class=\"xtick\"><text text-anchor=\"middle\" x=\"0\" y=\"460.4\" style=\"font-family: &quot;Open Sans&quot;, verdana, arial, sans-serif; font-size: 12px; fill: rgb(42, 63, 95); fill-opacity: 1; white-space: pre;\" data-unformatted=\"PC10\" data-math=\"N\" transform=\"translate(493,0)\">PC10</text></g><g class=\"xtick\"><text text-anchor=\"middle\" x=\"0\" y=\"460.4\" style=\"font-family: &quot;Open Sans&quot;, verdana, arial, sans-serif; font-size: 12px; fill: rgb(42, 63, 95); fill-opacity: 1; white-space: pre;\" data-unformatted=\"PC7\" data-math=\"N\" transform=\"translate(568.0899999999999,0)\">PC7</text></g><g class=\"xtick\"><text text-anchor=\"middle\" x=\"0\" y=\"460.4\" style=\"font-family: &quot;Open Sans&quot;, verdana, arial, sans-serif; font-size: 12px; fill: rgb(42, 63, 95); fill-opacity: 1; white-space: pre;\" data-unformatted=\"PC9\" data-math=\"N\" transform=\"translate(643.18,0)\">PC9</text></g><g class=\"xtick\"><text text-anchor=\"middle\" x=\"0\" y=\"460.4\" style=\"font-family: &quot;Open Sans&quot;, verdana, arial, sans-serif; font-size: 12px; fill: rgb(42, 63, 95); fill-opacity: 1; white-space: pre;\" data-unformatted=\"PC6\" data-math=\"N\" transform=\"translate(718.27,0)\">PC6</text></g><g class=\"xtick\"><text text-anchor=\"middle\" x=\"0\" y=\"460.4\" style=\"font-family: &quot;Open Sans&quot;, verdana, arial, sans-serif; font-size: 12px; fill: rgb(42, 63, 95); fill-opacity: 1; white-space: pre;\" data-unformatted=\"PC5\" data-math=\"N\" transform=\"translate(793.36,0)\">PC5</text></g><g class=\"xtick\"><text text-anchor=\"middle\" x=\"0\" y=\"460.4\" style=\"font-family: &quot;Open Sans&quot;, verdana, arial, sans-serif; font-size: 12px; fill: rgb(42, 63, 95); fill-opacity: 1; white-space: pre;\" data-unformatted=\"PC4\" data-math=\"N\" transform=\"translate(868.45,0)\">PC4</text></g></g><g class=\"yaxislayer-above\"><g class=\"ytick\"><text text-anchor=\"end\" x=\"76.6\" y=\"4.199999999999999\" style=\"font-family: &quot;Open Sans&quot;, verdana, arial, sans-serif; font-size: 12px; fill: rgb(42, 63, 95); fill-opacity: 1; white-space: pre;\" data-unformatted=\"0\" data-math=\"N\" transform=\"translate(0,445)\">0</text></g><g class=\"ytick\"><text text-anchor=\"end\" x=\"76.6\" y=\"4.199999999999999\" style=\"font-family: &quot;Open Sans&quot;, verdana, arial, sans-serif; font-size: 12px; fill: rgb(42, 63, 95); fill-opacity: 1; white-space: pre;\" data-unformatted=\"0.02\" data-math=\"N\" transform=\"translate(0,401.71)\">0.02</text></g><g class=\"ytick\"><text text-anchor=\"end\" x=\"76.6\" y=\"4.199999999999999\" style=\"font-family: &quot;Open Sans&quot;, verdana, arial, sans-serif; font-size: 12px; fill: rgb(42, 63, 95); fill-opacity: 1; white-space: pre;\" data-unformatted=\"0.04\" data-math=\"N\" transform=\"translate(0,358.41)\">0.04</text></g><g class=\"ytick\"><text text-anchor=\"end\" x=\"76.6\" y=\"4.199999999999999\" style=\"font-family: &quot;Open Sans&quot;, verdana, arial, sans-serif; font-size: 12px; fill: rgb(42, 63, 95); fill-opacity: 1; white-space: pre;\" data-unformatted=\"0.06\" data-math=\"N\" transform=\"translate(0,315.12)\">0.06</text></g><g class=\"ytick\"><text text-anchor=\"end\" x=\"76.6\" y=\"4.199999999999999\" style=\"font-family: &quot;Open Sans&quot;, verdana, arial, sans-serif; font-size: 12px; fill: rgb(42, 63, 95); fill-opacity: 1; white-space: pre;\" data-unformatted=\"0.08\" data-math=\"N\" transform=\"translate(0,271.82)\">0.08</text></g><g class=\"ytick\"><text text-anchor=\"end\" x=\"76.6\" y=\"4.199999999999999\" style=\"font-family: &quot;Open Sans&quot;, verdana, arial, sans-serif; font-size: 12px; fill: rgb(42, 63, 95); fill-opacity: 1; white-space: pre;\" data-unformatted=\"0.1\" data-math=\"N\" transform=\"translate(0,228.53)\">0.1</text></g><g class=\"ytick\"><text text-anchor=\"end\" x=\"76.6\" y=\"4.199999999999999\" style=\"font-family: &quot;Open Sans&quot;, verdana, arial, sans-serif; font-size: 12px; fill: rgb(42, 63, 95); fill-opacity: 1; white-space: pre;\" data-unformatted=\"0.12\" data-math=\"N\" transform=\"translate(0,185.23000000000002)\">0.12</text></g><g class=\"ytick\"><text text-anchor=\"end\" x=\"76.6\" y=\"4.199999999999999\" style=\"font-family: &quot;Open Sans&quot;, verdana, arial, sans-serif; font-size: 12px; fill: rgb(42, 63, 95); fill-opacity: 1; white-space: pre;\" data-unformatted=\"0.14\" data-math=\"N\" transform=\"translate(0,141.94)\">0.14</text></g><g class=\"ytick\"><text text-anchor=\"end\" x=\"76.6\" y=\"4.199999999999999\" style=\"font-family: &quot;Open Sans&quot;, verdana, arial, sans-serif; font-size: 12px; fill: rgb(42, 63, 95); fill-opacity: 1; white-space: pre;\" data-unformatted=\"0.16\" data-math=\"N\" transform=\"translate(0,98.65)\">0.16</text></g></g><g class=\"overaxes-above\"></g></g></g><g class=\"polarlayer\"></g><g class=\"ternarylayer\"></g><g class=\"geolayer\"></g><g class=\"funnelarealayer\"></g><g class=\"pielayer\"></g><g class=\"treemaplayer\"></g><g class=\"sunburstlayer\"></g><g class=\"glimages\"></g></svg><div class=\"gl-container\"></div><svg class=\"main-svg\" xmlns=\"http://www.w3.org/2000/svg\" xmlns:xlink=\"http://www.w3.org/1999/xlink\" width=\"985.533\" height=\"525\"><defs id=\"topdefs-e207b3\"><g class=\"clips\"></g></defs><g class=\"indicatorlayer\"></g><g class=\"layer-above\"><g class=\"imagelayer\"></g><g class=\"shapelayer\"></g></g><g class=\"infolayer\"><g class=\"g-gtitle\"><text class=\"gtitle\" style=\"font-family: &quot;Open Sans&quot;, verdana, arial, sans-serif; font-size: 17px; fill: rgb(42, 63, 95); opacity: 1; font-weight: normal; white-space: pre;\" x=\"49.276650000000004\" y=\"30\" text-anchor=\"start\" dy=\"0em\" data-unformatted=\"Most important features in Random forest classification\" data-math=\"N\">Most important features in Random forest classification</text></g><g class=\"g-xtitle\"><text class=\"xtitle\" style=\"font-family: &quot;Open Sans&quot;, verdana, arial, sans-serif; font-size: 14px; fill: rgb(42, 63, 95); opacity: 1; font-weight: normal; white-space: pre;\" x=\"493\" y=\"488.4505126953125\" text-anchor=\"middle\" data-unformatted=\"Features\" data-math=\"N\">Features</text></g><g class=\"g-ytitle\"><text class=\"ytitle\" transform=\"rotate(-90,23.749804687500003,252.5)\" style=\"font-family: &quot;Open Sans&quot;, verdana, arial, sans-serif; font-size: 14px; fill: rgb(42, 63, 95); opacity: 1; font-weight: normal; white-space: pre;\" x=\"23.749804687500003\" y=\"252.5\" text-anchor=\"middle\" data-unformatted=\"Gini importance\" data-math=\"N\">Gini importance</text></g></g><g class=\"menulayer\"></g><g class=\"zoomlayer\"></g></svg><div class=\"modebar-container\" style=\"position: absolute; top: 0px; right: 0px; width: 100%;\"><div id=\"modebar-e207b3\" class=\"modebar modebar--hover ease-bg\"><div class=\"modebar-group\"><a rel=\"tooltip\" class=\"modebar-btn\" data-title=\"Download plot as a png\" data-toggle=\"false\" data-gravity=\"n\"><svg viewBox=\"0 0 1000 1000\" class=\"icon\" height=\"1em\" width=\"1em\"><path d=\"m500 450c-83 0-150-67-150-150 0-83 67-150 150-150 83 0 150 67 150 150 0 83-67 150-150 150z m400 150h-120c-16 0-34 13-39 29l-31 93c-6 15-23 28-40 28h-340c-16 0-34-13-39-28l-31-94c-6-15-23-28-40-28h-120c-55 0-100-45-100-100v-450c0-55 45-100 100-100h800c55 0 100 45 100 100v450c0 55-45 100-100 100z m-400-550c-138 0-250 112-250 250 0 138 112 250 250 250 138 0 250-112 250-250 0-138-112-250-250-250z m365 380c-19 0-35 16-35 35 0 19 16 35 35 35 19 0 35-16 35-35 0-19-16-35-35-35z\" transform=\"matrix(1 0 0 -1 0 850)\"></path></svg></a></div><div class=\"modebar-group\"><a rel=\"tooltip\" class=\"modebar-btn active\" data-title=\"Zoom\" data-attr=\"dragmode\" data-val=\"zoom\" data-toggle=\"false\" data-gravity=\"n\"><svg viewBox=\"0 0 1000 1000\" class=\"icon\" height=\"1em\" width=\"1em\"><path d=\"m1000-25l-250 251c40 63 63 138 63 218 0 224-182 406-407 406-224 0-406-182-406-406s183-406 407-406c80 0 155 22 218 62l250-250 125 125z m-812 250l0 438 437 0 0-438-437 0z m62 375l313 0 0-312-313 0 0 312z\" transform=\"matrix(1 0 0 -1 0 850)\"></path></svg></a><a rel=\"tooltip\" class=\"modebar-btn\" data-title=\"Pan\" data-attr=\"dragmode\" data-val=\"pan\" data-toggle=\"false\" data-gravity=\"n\"><svg viewBox=\"0 0 1000 1000\" class=\"icon\" height=\"1em\" width=\"1em\"><path d=\"m1000 350l-187 188 0-125-250 0 0 250 125 0-188 187-187-187 125 0 0-250-250 0 0 125-188-188 186-187 0 125 252 0 0-250-125 0 187-188 188 188-125 0 0 250 250 0 0-126 187 188z\" transform=\"matrix(1 0 0 -1 0 850)\"></path></svg></a><a rel=\"tooltip\" class=\"modebar-btn\" data-title=\"Box Select\" data-attr=\"dragmode\" data-val=\"select\" data-toggle=\"false\" data-gravity=\"n\"><svg viewBox=\"0 0 1000 1000\" class=\"icon\" height=\"1em\" width=\"1em\"><path d=\"m0 850l0-143 143 0 0 143-143 0z m286 0l0-143 143 0 0 143-143 0z m285 0l0-143 143 0 0 143-143 0z m286 0l0-143 143 0 0 143-143 0z m-857-286l0-143 143 0 0 143-143 0z m857 0l0-143 143 0 0 143-143 0z m-857-285l0-143 143 0 0 143-143 0z m857 0l0-143 143 0 0 143-143 0z m-857-286l0-143 143 0 0 143-143 0z m286 0l0-143 143 0 0 143-143 0z m285 0l0-143 143 0 0 143-143 0z m286 0l0-143 143 0 0 143-143 0z\" transform=\"matrix(1 0 0 -1 0 850)\"></path></svg></a><a rel=\"tooltip\" class=\"modebar-btn\" data-title=\"Lasso Select\" data-attr=\"dragmode\" data-val=\"lasso\" data-toggle=\"false\" data-gravity=\"n\"><svg viewBox=\"0 0 1031 1000\" class=\"icon\" height=\"1em\" width=\"1em\"><path d=\"m1018 538c-36 207-290 336-568 286-277-48-473-256-436-463 10-57 36-108 76-151-13-66 11-137 68-183 34-28 75-41 114-42l-55-70 0 0c-2-1-3-2-4-3-10-14-8-34 5-45 14-11 34-8 45 4 1 1 2 3 2 5l0 0 113 140c16 11 31 24 45 40 4 3 6 7 8 11 48-3 100 0 151 9 278 48 473 255 436 462z m-624-379c-80 14-149 48-197 96 42 42 109 47 156 9 33-26 47-66 41-105z m-187-74c-19 16-33 37-39 60 50-32 109-55 174-68-42-25-95-24-135 8z m360 75c-34-7-69-9-102-8 8 62-16 128-68 170-73 59-175 54-244-5-9 20-16 40-20 61-28 159 121 317 333 354s407-60 434-217c28-159-121-318-333-355z\" transform=\"matrix(1 0 0 -1 0 850)\"></path></svg></a></div><div class=\"modebar-group\"><a rel=\"tooltip\" class=\"modebar-btn\" data-title=\"Zoom in\" data-attr=\"zoom\" data-val=\"in\" data-toggle=\"false\" data-gravity=\"n\"><svg viewBox=\"0 0 875 1000\" class=\"icon\" height=\"1em\" width=\"1em\"><path d=\"m1 787l0-875 875 0 0 875-875 0z m687-500l-187 0 0-187-125 0 0 187-188 0 0 125 188 0 0 187 125 0 0-187 187 0 0-125z\" transform=\"matrix(1 0 0 -1 0 850)\"></path></svg></a><a rel=\"tooltip\" class=\"modebar-btn\" data-title=\"Zoom out\" data-attr=\"zoom\" data-val=\"out\" data-toggle=\"false\" data-gravity=\"n\"><svg viewBox=\"0 0 875 1000\" class=\"icon\" height=\"1em\" width=\"1em\"><path d=\"m0 788l0-876 875 0 0 876-875 0z m688-500l-500 0 0 125 500 0 0-125z\" transform=\"matrix(1 0 0 -1 0 850)\"></path></svg></a><a rel=\"tooltip\" class=\"modebar-btn\" data-title=\"Autoscale\" data-attr=\"zoom\" data-val=\"auto\" data-toggle=\"false\" data-gravity=\"n\"><svg viewBox=\"0 0 1000 1000\" class=\"icon\" height=\"1em\" width=\"1em\"><path d=\"m250 850l-187 0-63 0 0-62 0-188 63 0 0 188 187 0 0 62z m688 0l-188 0 0-62 188 0 0-188 62 0 0 188 0 62-62 0z m-875-938l0 188-63 0 0-188 0-62 63 0 187 0 0 62-187 0z m875 188l0-188-188 0 0-62 188 0 62 0 0 62 0 188-62 0z m-125 188l-1 0-93-94-156 156 156 156 92-93 2 0 0 250-250 0 0-2 93-92-156-156-156 156 94 92 0 2-250 0 0-250 0 0 93 93 157-156-157-156-93 94 0 0 0-250 250 0 0 0-94 93 156 157 156-157-93-93 0 0 250 0 0 250z\" transform=\"matrix(1 0 0 -1 0 850)\"></path></svg></a><a rel=\"tooltip\" class=\"modebar-btn\" data-title=\"Reset axes\" data-attr=\"zoom\" data-val=\"reset\" data-toggle=\"false\" data-gravity=\"n\"><svg viewBox=\"0 0 928.6 1000\" class=\"icon\" height=\"1em\" width=\"1em\"><path d=\"m786 296v-267q0-15-11-26t-25-10h-214v214h-143v-214h-214q-15 0-25 10t-11 26v267q0 1 0 2t0 2l321 264 321-264q1-1 1-4z m124 39l-34-41q-5-5-12-6h-2q-7 0-12 3l-386 322-386-322q-7-4-13-4-7 2-12 7l-35 41q-4 5-3 13t6 12l401 334q18 15 42 15t43-15l136-114v109q0 8 5 13t13 5h107q8 0 13-5t5-13v-227l122-102q5-5 6-12t-4-13z\" transform=\"matrix(1 0 0 -1 0 850)\"></path></svg></a></div><div class=\"modebar-group\"><a rel=\"tooltip\" class=\"modebar-btn\" data-title=\"Toggle Spike Lines\" data-attr=\"_cartesianSpikesEnabled\" data-val=\"on\" data-toggle=\"false\" data-gravity=\"n\"><svg viewBox=\"0 0 1000 1000\" class=\"icon\" height=\"1em\" width=\"1em\"><path d=\"M512 409c0-57-46-104-103-104-57 0-104 47-104 104 0 57 47 103 104 103 57 0 103-46 103-103z m-327-39l92 0 0 92-92 0z m-185 0l92 0 0 92-92 0z m370-186l92 0 0 93-92 0z m0-184l92 0 0 92-92 0z\" transform=\"matrix(1.5 0 0 -1.5 0 850)\"></path></svg></a><a rel=\"tooltip\" class=\"modebar-btn active\" data-title=\"Show closest data on hover\" data-attr=\"hovermode\" data-val=\"closest\" data-toggle=\"false\" data-gravity=\"ne\"><svg viewBox=\"0 0 1500 1000\" class=\"icon\" height=\"1em\" width=\"1em\"><path d=\"m375 725l0 0-375-375 375-374 0-1 1125 0 0 750-1125 0z\" transform=\"matrix(1 0 0 -1 0 850)\"></path></svg></a><a rel=\"tooltip\" class=\"modebar-btn\" data-title=\"Compare data on hover\" data-attr=\"hovermode\" data-val=\"x\" data-toggle=\"false\" data-gravity=\"ne\"><svg viewBox=\"0 0 1125 1000\" class=\"icon\" height=\"1em\" width=\"1em\"><path d=\"m187 786l0 2-187-188 188-187 0 0 937 0 0 373-938 0z m0-499l0 1-187-188 188-188 0 0 937 0 0 376-938-1z\" transform=\"matrix(1 0 0 -1 0 850)\"></path></svg></a></div><div class=\"modebar-group\"><a href=\"https://plotly.com/\" target=\"_blank\" data-title=\"Produced with Plotly\" class=\"modebar-btn plotlyjsicon modebar-btn--logo\"><svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 132 132\" height=\"1em\" width=\"1em\"><defs><style>.cls-1 {fill: #3f4f75;} .cls-2 {fill: #80cfbe;} .cls-3 {fill: #fff;}</style></defs><title>plotly-logomark</title><g id=\"symbol\"><rect class=\"cls-1\" width=\"132\" height=\"132\" rx=\"6\" ry=\"6\"></rect><circle class=\"cls-2\" cx=\"78\" cy=\"54\" r=\"6\"></circle><circle class=\"cls-2\" cx=\"102\" cy=\"30\" r=\"6\"></circle><circle class=\"cls-2\" cx=\"78\" cy=\"30\" r=\"6\"></circle><circle class=\"cls-2\" cx=\"54\" cy=\"30\" r=\"6\"></circle><circle class=\"cls-2\" cx=\"30\" cy=\"30\" r=\"6\"></circle><circle class=\"cls-2\" cx=\"30\" cy=\"54\" r=\"6\"></circle><path class=\"cls-3\" d=\"M30,72a6,6,0,0,0-6,6v24a6,6,0,0,0,12,0V78A6,6,0,0,0,30,72Z\"></path><path class=\"cls-3\" d=\"M78,72a6,6,0,0,0-6,6v24a6,6,0,0,0,12,0V78A6,6,0,0,0,78,72Z\"></path><path class=\"cls-3\" d=\"M54,48a6,6,0,0,0-6,6v48a6,6,0,0,0,12,0V54A6,6,0,0,0,54,48Z\"></path><path class=\"cls-3\" d=\"M102,48a6,6,0,0,0-6,6v48a6,6,0,0,0,12,0V54A6,6,0,0,0,102,48Z\"></path></g></svg></a></div></div></div><svg class=\"main-svg\" xmlns=\"http://www.w3.org/2000/svg\" xmlns:xlink=\"http://www.w3.org/1999/xlink\" width=\"985.533\" height=\"525\"><g class=\"hoverlayer\"></g></svg></div></div></div>")
    <script type="text/javascript">
        require(["plotly"], function(Plotly) {
            window.PLOTLYENV=window.PLOTLYENV || {};
            
        if (document.getElementById("9a5e763d-1034-4f1e-aec9-c2bd6e400485")) {
            Plotly.newPlot(
                '9a5e763d-1034-4f1e-aec9-c2bd6e400485',
                [{"alignmentgroup": "True", "bingroup": "x", "histfunc": "avg", "hovertemplate": "color=PC1<br>x=get_ipython().run_line_magic("{x}<br>avg", " of y=%{y}<extra></extra>\", \"legendgroup\": \"PC1\", \"marker\": {\"color\": \"rgb(102,194,165)\"}, \"name\": \"PC1\", \"offsetgroup\": \"PC1\", \"orientation\": \"v\", \"showlegend\": true, \"type\": \"histogram\", \"x\": [\"PC1\"], \"xaxis\": \"x\", \"y\": [0.1689598569295432], \"yaxis\": \"y\"}, {\"alignmentgroup\": \"True\", \"bingroup\": \"x\", \"histfunc\": \"avg\", \"hovertemplate\": \"color=PC2<br>x=%{x}<br>avg of y=%{y}<extra></extra>\", \"legendgroup\": \"PC2\", \"marker\": {\"color\": \"rgb(252,141,98)\"}, \"name\": \"PC2\", \"offsetgroup\": \"PC2\", \"orientation\": \"v\", \"showlegend\": true, \"type\": \"histogram\", \"x\": [\"PC2\"], \"xaxis\": \"x\", \"y\": [0.1036894047485998], \"yaxis\": \"y\"}, {\"alignmentgroup\": \"True\", \"bingroup\": \"x\", \"histfunc\": \"avg\", \"hovertemplate\": \"color=PC8<br>x=%{x}<br>avg of y=%{y}<extra></extra>\", \"legendgroup\": \"PC8\", \"marker\": {\"color\": \"rgb(141,160,203)\"}, \"name\": \"PC8\", \"offsetgroup\": \"PC8\", \"orientation\": \"v\", \"showlegend\": true, \"type\": \"histogram\", \"x\": [\"PC8\"], \"xaxis\": \"x\", \"y\": [0.09378873255588618], \"yaxis\": \"y\"}, {\"alignmentgroup\": \"True\", \"bingroup\": \"x\", \"histfunc\": \"avg\", \"hovertemplate\": \"color=PC3<br>x=%{x}<br>avg of y=%{y}<extra></extra>\", \"legendgroup\": \"PC3\", \"marker\": {\"color\": \"rgb(231,138,195)\"}, \"name\": \"PC3\", \"offsetgroup\": \"PC3\", \"orientation\": \"v\", \"showlegend\": true, \"type\": \"histogram\", \"x\": [\"PC3\"], \"xaxis\": \"x\", \"y\": [0.0924419393483735], \"yaxis\": \"y\"}, {\"alignmentgroup\": \"True\", \"bingroup\": \"x\", \"histfunc\": \"avg\", \"hovertemplate\": \"color=PC11<br>x=%{x}<br>avg of y=%{y}<extra></extra>\", \"legendgroup\": \"PC11\", \"marker\": {\"color\": \"rgb(166,216,84)\"}, \"name\": \"PC11\", \"offsetgroup\": \"PC11\", \"orientation\": \"v\", \"showlegend\": true, \"type\": \"histogram\", \"x\": [\"PC11\"], \"xaxis\": \"x\", \"y\": [0.08777132947709518], \"yaxis\": \"y\"}, {\"alignmentgroup\": \"True\", \"bingroup\": \"x\", \"histfunc\": \"avg\", \"hovertemplate\": \"color=PC10<br>x=%{x}<br>avg of y=%{y}<extra></extra>\", \"legendgroup\": \"PC10\", \"marker\": {\"color\": \"rgb(255,217,47)\"}, \"name\": \"PC10\", \"offsetgroup\": \"PC10\", \"orientation\": \"v\", \"showlegend\": true, \"type\": \"histogram\", \"x\": [\"PC10\"], \"xaxis\": \"x\", \"y\": [0.08175351267267282], \"yaxis\": \"y\"}, {\"alignmentgroup\": \"True\", \"bingroup\": \"x\", \"histfunc\": \"avg\", \"hovertemplate\": \"color=PC7<br>x=%{x}<br>avg of y=%{y}<extra></extra>\", \"legendgroup\": \"PC7\", \"marker\": {\"color\": \"rgb(229,196,148)\"}, \"name\": \"PC7\", \"offsetgroup\": \"PC7\", \"orientation\": \"v\", \"showlegend\": true, \"type\": \"histogram\", \"x\": [\"PC7\"], \"xaxis\": \"x\", \"y\": [0.08042301481562293], \"yaxis\": \"y\"}, {\"alignmentgroup\": \"True\", \"bingroup\": \"x\", \"histfunc\": \"avg\", \"hovertemplate\": \"color=PC9<br>x=%{x}<br>avg of y=%{y}<extra></extra>\", \"legendgroup\": \"PC9\", \"marker\": {\"color\": \"rgb(179,179,179)\"}, \"name\": \"PC9\", \"offsetgroup\": \"PC9\", \"orientation\": \"v\", \"showlegend\": true, \"type\": \"histogram\", \"x\": [\"PC9\"], \"xaxis\": \"x\", \"y\": [0.07522895083292785], \"yaxis\": \"y\"}, {\"alignmentgroup\": \"True\", \"bingroup\": \"x\", \"histfunc\": \"avg\", \"hovertemplate\": \"color=PC6<br>x=%{x}<br>avg of y=%{y}<extra></extra>\", \"legendgroup\": \"PC6\", \"marker\": {\"color\": \"rgb(102,194,165)\"}, \"name\": \"PC6\", \"offsetgroup\": \"PC6\", \"orientation\": \"v\", \"showlegend\": true, \"type\": \"histogram\", \"x\": [\"PC6\"], \"xaxis\": \"x\", \"y\": [0.07239145850095174], \"yaxis\": \"y\"}, {\"alignmentgroup\": \"True\", \"bingroup\": \"x\", \"histfunc\": \"avg\", \"hovertemplate\": \"color=PC5<br>x=%{x}<br>avg of y=%{y}<extra></extra>\", \"legendgroup\": \"PC5\", \"marker\": {\"color\": \"rgb(252,141,98)\"}, \"name\": \"PC5\", \"offsetgroup\": \"PC5\", \"orientation\": \"v\", \"showlegend\": true, \"type\": \"histogram\", \"x\": [\"PC5\"], \"xaxis\": \"x\", \"y\": [0.0718588897445081], \"yaxis\": \"y\"}, {\"alignmentgroup\": \"True\", \"bingroup\": \"x\", \"histfunc\": \"avg\", \"hovertemplate\": \"color=PC4<br>x=%{x}<br>avg of y=%{y}<extra></extra>\", \"legendgroup\": \"PC4\", \"marker\": {\"color\": \"rgb(141,160,203)\"}, \"name\": \"PC4\", \"offsetgroup\": \"PC4\", \"orientation\": \"v\", \"showlegend\": true, \"type\": \"histogram\", \"x\": [\"PC4\"], \"xaxis\": \"x\", \"y\": [0.07169291037381866], \"yaxis\": \"y\"}],")
                {"barmode": "relative", "legend": {"title": {"text": "color"}, "tracegroupgap": 0}, "margin": {"t": 60}, "showlegend": false, "template": {"data": {"bar": [{"error_x": {"color": "#2a3f5f"}, "error_y": {"color": "#2a3f5f"}, "marker": {"line": {"color": "#E5ECF6", "width": 0.5}}, "type": "bar"}], "barpolar": [{"marker": {"line": {"color": "#E5ECF6", "width": 0.5}}, "type": "barpolar"}], "carpet": [{"aaxis": {"endlinecolor": "#2a3f5f", "gridcolor": "white", "linecolor": "white", "minorgridcolor": "white", "startlinecolor": "#2a3f5f"}, "baxis": {"endlinecolor": "#2a3f5f", "gridcolor": "white", "linecolor": "white", "minorgridcolor": "white", "startlinecolor": "#2a3f5f"}, "type": "carpet"}], "choropleth": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "choropleth"}], "contour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "contour"}], "contourcarpet": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "contourcarpet"}], "heatmap": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmap"}], "heatmapgl": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmapgl"}], "histogram": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "histogram"}], "histogram2d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2d"}], "histogram2dcontour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2dcontour"}], "mesh3d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "mesh3d"}], "parcoords": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "parcoords"}], "pie": [{"automargin": true, "type": "pie"}], "scatter": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatter"}], "scatter3d": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatter3d"}], "scattercarpet": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattercarpet"}], "scattergeo": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattergeo"}], "scattergl": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattergl"}], "scattermapbox": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattermapbox"}], "scatterpolar": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolar"}], "scatterpolargl": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolargl"}], "scatterternary": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterternary"}], "surface": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "surface"}], "table": [{"cells": {"fill": {"color": "#EBF0F8"}, "line": {"color": "white"}}, "header": {"fill": {"color": "#C8D4E3"}, "line": {"color": "white"}}, "type": "table"}]}, "layout": {"annotationdefaults": {"arrowcolor": "#2a3f5f", "arrowhead": 0, "arrowwidth": 1}, "coloraxis": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "colorscale": {"diverging": [[0, "#8e0152"], [0.1, "#c51b7d"], [0.2, "#de77ae"], [0.3, "#f1b6da"], [0.4, "#fde0ef"], [0.5, "#f7f7f7"], [0.6, "#e6f5d0"], [0.7, "#b8e186"], [0.8, "#7fbc41"], [0.9, "#4d9221"], [1, "#276419"]], "sequential": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "sequentialminus": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]]}, "colorway": ["#636efa", "#EF553B", "#00cc96", "#ab63fa", "#FFA15A", "#19d3f3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52"], "font": {"color": "#2a3f5f"}, "geo": {"bgcolor": "white", "lakecolor": "white", "landcolor": "#E5ECF6", "showlakes": true, "showland": true, "subunitcolor": "white"}, "hoverlabel": {"align": "left"}, "hovermode": "closest", "mapbox": {"style": "light"}, "paper_bgcolor": "white", "plot_bgcolor": "#E5ECF6", "polar": {"angularaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "bgcolor": "#E5ECF6", "radialaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}}, "scene": {"xaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}, "yaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}, "zaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}}, "shapedefaults": {"line": {"color": "#2a3f5f"}}, "ternary": {"aaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "baxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "bgcolor": "#E5ECF6", "caxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}}, "title": {"x": 0.05}, "xaxis": {"automargin": true, "gridcolor": "white", "linecolor": "white", "ticks": "", "title": {"standoff": 15}, "zerolinecolor": "white", "zerolinewidth": 2}, "yaxis": {"automargin": true, "gridcolor": "white", "linecolor": "white", "ticks": "", "title": {"standoff": 15}, "zerolinecolor": "white", "zerolinewidth": 2}}}, "title": {"text": "Most important features in Random forest classification"}, "xaxis": {"anchor": "y", "domain": [0.0, 1.0], "title": {"text": "Features"}}, "yaxis": {"anchor": "x", "domain": [0.0, 1.0], "title": {"text": "Gini importance"}}},
                {"responsive": true}
            ).then(function(){
                    
var gd = document.getElementById('9a5e763d-1034-4f1e-aec9-c2bd6e400485');
var x = new MutationObserver(function (mutations, observer) {{
var display = window.getComputedStyle(gd).display;
if (get_ipython().getoutput("display || display === 'none') {{")
    console.log([gd, 'removedget_ipython().getoutput("']);")
    Plotly.purge(gd);
    observer.disconnect();
}}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
x.observe(outputEl, {childList: true});
}}

                })
        };
        });
    </script>
</div>

<h2>With Cluster Centroid</h2>

<div>
        
        
    <div id="c13947be-83de-4e12-b5cd-4b95eda7c1b7" class="plotly-graph-div js-plotly-plot" style="height:525px; width:100get_ipython().run_line_magic(";"><div", " class=\"plot-container plotly\"><div class=\"svg-container\" style=\"position: relative; width: 100%; height: 100%;\"><svg class=\"main-svg\" xmlns=\"http://www.w3.org/2000/svg\" xmlns:xlink=\"http://www.w3.org/1999/xlink\" width=\"985.533\" height=\"525\" style=\"background: white none repeat scroll 0% 0%;\"><defs id=\"defs-7240ae\"><g class=\"clips\"><clipPath id=\"clip7240aexyplot\" class=\"plotclip\"><rect width=\"826\" height=\"385\"></rect></clipPath><clipPath class=\"axesclip\" id=\"clip7240aex\"><rect x=\"80\" y=\"0\" width=\"826\" height=\"525\"></rect></clipPath><clipPath class=\"axesclip\" id=\"clip7240aey\"><rect x=\"0\" y=\"60\" width=\"985.533\" height=\"385\"></rect></clipPath><clipPath class=\"axesclip\" id=\"clip7240aexy\"><rect x=\"80\" y=\"60\" width=\"826\" height=\"385\"></rect></clipPath></g><g class=\"gradients\"></g></defs><g class=\"bglayer\"><rect class=\"bg\" x=\"80\" y=\"60\" width=\"826\" height=\"385\" style=\"fill: rgb(229, 236, 246); fill-opacity: 1; stroke-width: 0px;\"></rect></g><g class=\"draglayer cursor-crosshair\"><g class=\"xy\"><rect class=\"nsewdrag drag\" style=\"fill: transparent; stroke-width: 0px; pointer-events: all;\" data-subplot=\"xy\" x=\"80\" y=\"60\" width=\"826\" height=\"385\"></rect><rect class=\"nwdrag drag cursor-nw-resize\" style=\"fill: transparent; stroke-width: 0px; pointer-events: all;\" data-subplot=\"xy\" x=\"60\" y=\"40\" width=\"20\" height=\"20\"></rect><rect class=\"nedrag drag cursor-ne-resize\" style=\"fill: transparent; stroke-width: 0px; pointer-events: all;\" data-subplot=\"xy\" x=\"906\" y=\"40\" width=\"20\" height=\"20\"></rect><rect class=\"swdrag drag cursor-sw-resize\" style=\"fill: transparent; stroke-width: 0px; pointer-events: all;\" data-subplot=\"xy\" x=\"60\" y=\"445\" width=\"20\" height=\"20\"></rect><rect class=\"sedrag drag cursor-se-resize\" style=\"fill: transparent; stroke-width: 0px; pointer-events: all;\" data-subplot=\"xy\" x=\"906\" y=\"445\" width=\"20\" height=\"20\"></rect><rect class=\"ewdrag drag cursor-ew-resize\" style=\"fill: transparent; stroke-width: 0px; pointer-events: all;\" data-subplot=\"xy\" x=\"162.60000000000002\" y=\"445.5\" width=\"660.8000000000001\" height=\"20\"></rect><rect class=\"wdrag drag cursor-w-resize\" style=\"fill: transparent; stroke-width: 0px; pointer-events: all;\" data-subplot=\"xy\" x=\"80\" y=\"445.5\" width=\"82.60000000000001\" height=\"20\"></rect><rect class=\"edrag drag cursor-e-resize\" style=\"fill: transparent; stroke-width: 0px; pointer-events: all;\" data-subplot=\"xy\" x=\"823.4\" y=\"445.5\" width=\"82.60000000000001\" height=\"20\"></rect><rect class=\"nsdrag drag cursor-ns-resize\" style=\"fill: transparent; stroke-width: 0px; pointer-events: all;\" data-subplot=\"xy\" x=\"59.5\" y=\"98.5\" width=\"20\" height=\"308\"></rect><rect class=\"sdrag drag cursor-s-resize\" style=\"fill: transparent; stroke-width: 0px; pointer-events: all;\" data-subplot=\"xy\" x=\"59.5\" y=\"406.5\" width=\"20\" height=\"38.5\"></rect><rect class=\"ndrag drag cursor-n-resize\" style=\"fill: transparent; stroke-width: 0px; pointer-events: all;\" data-subplot=\"xy\" x=\"59.5\" y=\"60\" width=\"20\" height=\"38.5\"></rect></g></g><g class=\"layer-below\"><g class=\"imagelayer\"></g><g class=\"shapelayer\"></g></g><g class=\"cartesianlayer\"><g class=\"subplot xy\"><g class=\"layer-subplot\"><g class=\"shapelayer\"></g><g class=\"imagelayer\"></g></g><g class=\"gridlayer\"><g class=\"x\"><path class=\"xgrid crisp\" transform=\"translate(117.55,0)\" d=\"M0,60v385\" style=\"stroke: rgb(255, 255, 255); stroke-opacity: 1; stroke-width: 1px;\"></path><path class=\"xgrid crisp\" transform=\"translate(192.64,0)\" d=\"M0,60v385\" style=\"stroke: rgb(255, 255, 255); stroke-opacity: 1; stroke-width: 1px;\"></path><path class=\"xgrid crisp\" transform=\"translate(267.73,0)\" d=\"M0,60v385\" style=\"stroke: rgb(255, 255, 255); stroke-opacity: 1; stroke-width: 1px;\"></path><path class=\"xgrid crisp\" transform=\"translate(342.82,0)\" d=\"M0,60v385\" style=\"stroke: rgb(255, 255, 255); stroke-opacity: 1; stroke-width: 1px;\"></path><path class=\"xgrid crisp\" transform=\"translate(417.91,0)\" d=\"M0,60v385\" style=\"stroke: rgb(255, 255, 255); stroke-opacity: 1; stroke-width: 1px;\"></path><path class=\"xgrid crisp\" transform=\"translate(493,0)\" d=\"M0,60v385\" style=\"stroke: rgb(255, 255, 255); stroke-opacity: 1; stroke-width: 1px;\"></path><path class=\"xgrid crisp\" transform=\"translate(568.0899999999999,0)\" d=\"M0,60v385\" style=\"stroke: rgb(255, 255, 255); stroke-opacity: 1; stroke-width: 1px;\"></path><path class=\"xgrid crisp\" transform=\"translate(643.18,0)\" d=\"M0,60v385\" style=\"stroke: rgb(255, 255, 255); stroke-opacity: 1; stroke-width: 1px;\"></path><path class=\"xgrid crisp\" transform=\"translate(718.27,0)\" d=\"M0,60v385\" style=\"stroke: rgb(255, 255, 255); stroke-opacity: 1; stroke-width: 1px;\"></path><path class=\"xgrid crisp\" transform=\"translate(793.36,0)\" d=\"M0,60v385\" style=\"stroke: rgb(255, 255, 255); stroke-opacity: 1; stroke-width: 1px;\"></path><path class=\"xgrid crisp\" transform=\"translate(868.45,0)\" d=\"M0,60v385\" style=\"stroke: rgb(255, 255, 255); stroke-opacity: 1; stroke-width: 1px;\"></path></g><g class=\"y\"><path class=\"ygrid crisp\" transform=\"translate(0,405.11)\" d=\"M80,0h826\" style=\"stroke: rgb(255, 255, 255); stroke-opacity: 1; stroke-width: 1px;\"></path><path class=\"ygrid crisp\" transform=\"translate(0,365.22)\" d=\"M80,0h826\" style=\"stroke: rgb(255, 255, 255); stroke-opacity: 1; stroke-width: 1px;\"></path><path class=\"ygrid crisp\" transform=\"translate(0,325.33)\" d=\"M80,0h826\" style=\"stroke: rgb(255, 255, 255); stroke-opacity: 1; stroke-width: 1px;\"></path><path class=\"ygrid crisp\" transform=\"translate(0,285.44)\" d=\"M80,0h826\" style=\"stroke: rgb(255, 255, 255); stroke-opacity: 1; stroke-width: 1px;\"></path><path class=\"ygrid crisp\" transform=\"translate(0,245.55)\" d=\"M80,0h826\" style=\"stroke: rgb(255, 255, 255); stroke-opacity: 1; stroke-width: 1px;\"></path><path class=\"ygrid crisp\" transform=\"translate(0,205.66)\" d=\"M80,0h826\" style=\"stroke: rgb(255, 255, 255); stroke-opacity: 1; stroke-width: 1px;\"></path><path class=\"ygrid crisp\" transform=\"translate(0,165.76999999999998)\" d=\"M80,0h826\" style=\"stroke: rgb(255, 255, 255); stroke-opacity: 1; stroke-width: 1px;\"></path><path class=\"ygrid crisp\" transform=\"translate(0,125.88)\" d=\"M80,0h826\" style=\"stroke: rgb(255, 255, 255); stroke-opacity: 1; stroke-width: 1px;\"></path><path class=\"ygrid crisp\" transform=\"translate(0,85.99)\" d=\"M80,0h826\" style=\"stroke: rgb(255, 255, 255); stroke-opacity: 1; stroke-width: 1px;\"></path></g></g><g class=\"zerolinelayer\"><path class=\"yzl zl crisp\" transform=\"translate(0,445)\" d=\"M80,0h826\" style=\"stroke: rgb(255, 255, 255); stroke-opacity: 1; stroke-width: 2px;\"></path></g><path class=\"xlines-below\"></path><path class=\"ylines-below\"></path><g class=\"overlines-below\"></g><g class=\"xaxislayer-below\"></g><g class=\"yaxislayer-below\"></g><g class=\"overaxes-below\"></g><g class=\"plot\" transform=\"translate(80, 60)\" clip-path=\"url('#clip7240aexyplot')\"><g class=\"barlayer mlayer\"><g class=\"trace bars\" style=\"opacity: 1;\"><g class=\"points\"><g class=\"point\"><path style=\"vector-effect: non-scaling-stroke; opacity: 1; stroke-width: 0px; fill: rgb(102, 194, 165); fill-opacity: 1;\" d=\"M7.51,385V19.25H67.58V385Z\"></path></g></g></g><g class=\"trace bars\" style=\"opacity: 1;\"><g class=\"points\"><g class=\"point\"><path style=\"vector-effect: non-scaling-stroke; opacity: 1; stroke-width: 0px; fill: rgb(252, 141, 98); fill-opacity: 1;\" d=\"M82.6,385V191.21H142.67V385Z\"></path></g></g></g><g class=\"trace bars\" style=\"opacity: 1;\"><g class=\"points\"><g class=\"point\"><path style=\"vector-effect: non-scaling-stroke; opacity: 1; stroke-width: 0px; fill: rgb(141, 160, 203); fill-opacity: 1;\" d=\"M157.69,385V194.65H217.76V385Z\"></path></g></g></g><g class=\"trace bars\" style=\"opacity: 1;\"><g class=\"points\"><g class=\"point\"><path style=\"vector-effect: non-scaling-stroke; opacity: 1; stroke-width: 0px; fill: rgb(231, 138, 195); fill-opacity: 1;\" d=\"M232.78,385V197.41H292.85V385Z\"></path></g></g></g><g class=\"trace bars\" style=\"opacity: 1;\"><g class=\"points\"><g class=\"point\"><path style=\"vector-effect: non-scaling-stroke; opacity: 1; stroke-width: 0px; fill: rgb(166, 216, 84); fill-opacity: 1;\" d=\"M307.87,385V208.59H367.95V385Z\"></path></g></g></g><g class=\"trace bars\" style=\"opacity: 1;\"><g class=\"points\"><g class=\"point\"><path style=\"vector-effect: non-scaling-stroke; opacity: 1; stroke-width: 0px; fill: rgb(255, 217, 47); fill-opacity: 1;\" d=\"M382.96,385V216.17H443.04V385Z\"></path></g></g></g><g class=\"trace bars\" style=\"opacity: 1;\"><g class=\"points\"><g class=\"point\"><path style=\"vector-effect: non-scaling-stroke; opacity: 1; stroke-width: 0px; fill: rgb(229, 196, 148); fill-opacity: 1;\" d=\"M458.05,385V230.98H518.13V385Z\"></path></g></g></g><g class=\"trace bars\" style=\"opacity: 1;\"><g class=\"points\"><g class=\"point\"><path style=\"vector-effect: non-scaling-stroke; opacity: 1; stroke-width: 0px; fill: rgb(179, 179, 179); fill-opacity: 1;\" d=\"M533.15,385V238.47H593.22V385Z\"></path></g></g></g><g class=\"trace bars\" style=\"opacity: 1;\"><g class=\"points\"><g class=\"point\"><path style=\"vector-effect: non-scaling-stroke; opacity: 1; stroke-width: 0px; fill: rgb(102, 194, 165); fill-opacity: 1;\" d=\"M608.24,385V244.82H668.31V385Z\"></path></g></g></g><g class=\"trace bars\" style=\"opacity: 1;\"><g class=\"points\"><g class=\"point\"><path style=\"vector-effect: non-scaling-stroke; opacity: 1; stroke-width: 0px; fill: rgb(252, 141, 98); fill-opacity: 1;\" d=\"M683.33,385V246.88H743.4V385Z\"></path></g></g></g><g class=\"trace bars\" style=\"opacity: 1;\"><g class=\"points\"><g class=\"point\"><path style=\"vector-effect: non-scaling-stroke; opacity: 1; stroke-width: 0px; fill: rgb(141, 160, 203); fill-opacity: 1;\" d=\"M758.42,385V252.09H818.49V385Z\"></path></g></g></g></g></g><g class=\"overplot\"></g><path class=\"xlines-above crisp\" style=\"fill: none; stroke-width: 1px; stroke: rgb(255, 255, 255); stroke-opacity: 1;\" d=\"M79,445.5H906\"></path><path class=\"ylines-above crisp\" style=\"fill: none; stroke-width: 1px; stroke: rgb(255, 255, 255); stroke-opacity: 1;\" d=\"M79.5,60V445\"></path><g class=\"overlines-above\"></g><g class=\"xaxislayer-above\"><g class=\"xtick\"><text text-anchor=\"middle\" x=\"0\" y=\"460.4\" style=\"font-family: &quot;Open Sans&quot;, verdana, arial, sans-serif; font-size: 12px; fill: rgb(42, 63, 95); fill-opacity: 1; white-space: pre;\" data-unformatted=\"PC1\" data-math=\"N\" transform=\"translate(117.55,0)\">PC1</text></g><g class=\"xtick\"><text text-anchor=\"middle\" x=\"0\" y=\"460.4\" style=\"font-family: &quot;Open Sans&quot;, verdana, arial, sans-serif; font-size: 12px; fill: rgb(42, 63, 95); fill-opacity: 1; white-space: pre;\" data-unformatted=\"PC2\" data-math=\"N\" transform=\"translate(192.64,0)\">PC2</text></g><g class=\"xtick\"><text text-anchor=\"middle\" x=\"0\" y=\"460.4\" style=\"font-family: &quot;Open Sans&quot;, verdana, arial, sans-serif; font-size: 12px; fill: rgb(42, 63, 95); fill-opacity: 1; white-space: pre;\" data-unformatted=\"PC10\" data-math=\"N\" transform=\"translate(267.73,0)\">PC10</text></g><g class=\"xtick\"><text text-anchor=\"middle\" x=\"0\" y=\"460.4\" style=\"font-family: &quot;Open Sans&quot;, verdana, arial, sans-serif; font-size: 12px; fill: rgb(42, 63, 95); fill-opacity: 1; white-space: pre;\" data-unformatted=\"PC11\" data-math=\"N\" transform=\"translate(342.82,0)\">PC11</text></g><g class=\"xtick\"><text text-anchor=\"middle\" x=\"0\" y=\"460.4\" style=\"font-family: &quot;Open Sans&quot;, verdana, arial, sans-serif; font-size: 12px; fill: rgb(42, 63, 95); fill-opacity: 1; white-space: pre;\" data-unformatted=\"PC3\" data-math=\"N\" transform=\"translate(417.91,0)\">PC3</text></g><g class=\"xtick\"><text text-anchor=\"middle\" x=\"0\" y=\"460.4\" style=\"font-family: &quot;Open Sans&quot;, verdana, arial, sans-serif; font-size: 12px; fill: rgb(42, 63, 95); fill-opacity: 1; white-space: pre;\" data-unformatted=\"PC8\" data-math=\"N\" transform=\"translate(493,0)\">PC8</text></g><g class=\"xtick\"><text text-anchor=\"middle\" x=\"0\" y=\"460.4\" style=\"font-family: &quot;Open Sans&quot;, verdana, arial, sans-serif; font-size: 12px; fill: rgb(42, 63, 95); fill-opacity: 1; white-space: pre;\" data-unformatted=\"PC9\" data-math=\"N\" transform=\"translate(568.0899999999999,0)\">PC9</text></g><g class=\"xtick\"><text text-anchor=\"middle\" x=\"0\" y=\"460.4\" style=\"font-family: &quot;Open Sans&quot;, verdana, arial, sans-serif; font-size: 12px; fill: rgb(42, 63, 95); fill-opacity: 1; white-space: pre;\" data-unformatted=\"PC7\" data-math=\"N\" transform=\"translate(643.18,0)\">PC7</text></g><g class=\"xtick\"><text text-anchor=\"middle\" x=\"0\" y=\"460.4\" style=\"font-family: &quot;Open Sans&quot;, verdana, arial, sans-serif; font-size: 12px; fill: rgb(42, 63, 95); fill-opacity: 1; white-space: pre;\" data-unformatted=\"PC5\" data-math=\"N\" transform=\"translate(718.27,0)\">PC5</text></g><g class=\"xtick\"><text text-anchor=\"middle\" x=\"0\" y=\"460.4\" style=\"font-family: &quot;Open Sans&quot;, verdana, arial, sans-serif; font-size: 12px; fill: rgb(42, 63, 95); fill-opacity: 1; white-space: pre;\" data-unformatted=\"PC4\" data-math=\"N\" transform=\"translate(793.36,0)\">PC4</text></g><g class=\"xtick\"><text text-anchor=\"middle\" x=\"0\" y=\"460.4\" style=\"font-family: &quot;Open Sans&quot;, verdana, arial, sans-serif; font-size: 12px; fill: rgb(42, 63, 95); fill-opacity: 1; white-space: pre;\" data-unformatted=\"PC6\" data-math=\"N\" transform=\"translate(868.45,0)\">PC6</text></g></g><g class=\"yaxislayer-above\"><g class=\"ytick\"><text text-anchor=\"end\" x=\"76.6\" y=\"4.199999999999999\" style=\"font-family: &quot;Open Sans&quot;, verdana, arial, sans-serif; font-size: 12px; fill: rgb(42, 63, 95); fill-opacity: 1; white-space: pre;\" data-unformatted=\"0\" data-math=\"N\" transform=\"translate(0,445)\">0</text></g><g class=\"ytick\"><text text-anchor=\"end\" x=\"76.6\" y=\"4.199999999999999\" style=\"font-family: &quot;Open Sans&quot;, verdana, arial, sans-serif; font-size: 12px; fill: rgb(42, 63, 95); fill-opacity: 1; white-space: pre;\" data-unformatted=\"0.02\" data-math=\"N\" transform=\"translate(0,405.11)\">0.02</text></g><g class=\"ytick\"><text text-anchor=\"end\" x=\"76.6\" y=\"4.199999999999999\" style=\"font-family: &quot;Open Sans&quot;, verdana, arial, sans-serif; font-size: 12px; fill: rgb(42, 63, 95); fill-opacity: 1; white-space: pre;\" data-unformatted=\"0.04\" data-math=\"N\" transform=\"translate(0,365.22)\">0.04</text></g><g class=\"ytick\"><text text-anchor=\"end\" x=\"76.6\" y=\"4.199999999999999\" style=\"font-family: &quot;Open Sans&quot;, verdana, arial, sans-serif; font-size: 12px; fill: rgb(42, 63, 95); fill-opacity: 1; white-space: pre;\" data-unformatted=\"0.06\" data-math=\"N\" transform=\"translate(0,325.33)\">0.06</text></g><g class=\"ytick\"><text text-anchor=\"end\" x=\"76.6\" y=\"4.199999999999999\" style=\"font-family: &quot;Open Sans&quot;, verdana, arial, sans-serif; font-size: 12px; fill: rgb(42, 63, 95); fill-opacity: 1; white-space: pre;\" data-unformatted=\"0.08\" data-math=\"N\" transform=\"translate(0,285.44)\">0.08</text></g><g class=\"ytick\"><text text-anchor=\"end\" x=\"76.6\" y=\"4.199999999999999\" style=\"font-family: &quot;Open Sans&quot;, verdana, arial, sans-serif; font-size: 12px; fill: rgb(42, 63, 95); fill-opacity: 1; white-space: pre;\" data-unformatted=\"0.1\" data-math=\"N\" transform=\"translate(0,245.55)\">0.1</text></g><g class=\"ytick\"><text text-anchor=\"end\" x=\"76.6\" y=\"4.199999999999999\" style=\"font-family: &quot;Open Sans&quot;, verdana, arial, sans-serif; font-size: 12px; fill: rgb(42, 63, 95); fill-opacity: 1; white-space: pre;\" data-unformatted=\"0.12\" data-math=\"N\" transform=\"translate(0,205.66)\">0.12</text></g><g class=\"ytick\"><text text-anchor=\"end\" x=\"76.6\" y=\"4.199999999999999\" style=\"font-family: &quot;Open Sans&quot;, verdana, arial, sans-serif; font-size: 12px; fill: rgb(42, 63, 95); fill-opacity: 1; white-space: pre;\" data-unformatted=\"0.14\" data-math=\"N\" transform=\"translate(0,165.76999999999998)\">0.14</text></g><g class=\"ytick\"><text text-anchor=\"end\" x=\"76.6\" y=\"4.199999999999999\" style=\"font-family: &quot;Open Sans&quot;, verdana, arial, sans-serif; font-size: 12px; fill: rgb(42, 63, 95); fill-opacity: 1; white-space: pre;\" data-unformatted=\"0.16\" data-math=\"N\" transform=\"translate(0,125.88)\">0.16</text></g><g class=\"ytick\"><text text-anchor=\"end\" x=\"76.6\" y=\"4.199999999999999\" style=\"font-family: &quot;Open Sans&quot;, verdana, arial, sans-serif; font-size: 12px; fill: rgb(42, 63, 95); fill-opacity: 1; white-space: pre;\" data-unformatted=\"0.18\" data-math=\"N\" transform=\"translate(0,85.99)\">0.18</text></g></g><g class=\"overaxes-above\"></g></g></g><g class=\"polarlayer\"></g><g class=\"ternarylayer\"></g><g class=\"geolayer\"></g><g class=\"funnelarealayer\"></g><g class=\"pielayer\"></g><g class=\"treemaplayer\"></g><g class=\"sunburstlayer\"></g><g class=\"glimages\"></g></svg><div class=\"gl-container\"></div><svg class=\"main-svg\" xmlns=\"http://www.w3.org/2000/svg\" xmlns:xlink=\"http://www.w3.org/1999/xlink\" width=\"985.533\" height=\"525\"><defs id=\"topdefs-7240ae\"><g class=\"clips\"></g></defs><g class=\"indicatorlayer\"></g><g class=\"layer-above\"><g class=\"imagelayer\"></g><g class=\"shapelayer\"></g></g><g class=\"infolayer\"><g class=\"g-gtitle\"><text class=\"gtitle\" style=\"font-family: &quot;Open Sans&quot;, verdana, arial, sans-serif; font-size: 17px; fill: rgb(42, 63, 95); opacity: 1; font-weight: normal; white-space: pre;\" x=\"49.276650000000004\" y=\"30\" text-anchor=\"start\" dy=\"0em\" data-unformatted=\"Most important features in Random forest classification\" data-math=\"N\">Most important features in Random forest classification</text></g><g class=\"g-xtitle\"><text class=\"xtitle\" style=\"font-family: &quot;Open Sans&quot;, verdana, arial, sans-serif; font-size: 14px; fill: rgb(42, 63, 95); opacity: 1; font-weight: normal; white-space: pre;\" x=\"493\" y=\"488.4505126953125\" text-anchor=\"middle\" data-unformatted=\"Features\" data-math=\"N\">Features</text></g><g class=\"g-ytitle\"><text class=\"ytitle\" transform=\"rotate(-90,23.749804687500003,252.5)\" style=\"font-family: &quot;Open Sans&quot;, verdana, arial, sans-serif; font-size: 14px; fill: rgb(42, 63, 95); opacity: 1; font-weight: normal; white-space: pre;\" x=\"23.749804687500003\" y=\"252.5\" text-anchor=\"middle\" data-unformatted=\"Gini importance\" data-math=\"N\">Gini importance</text></g></g><g class=\"menulayer\"></g><g class=\"zoomlayer\"></g></svg><div class=\"modebar-container\" style=\"position: absolute; top: 0px; right: 0px; width: 100%;\"><div id=\"modebar-7240ae\" class=\"modebar modebar--hover ease-bg\"><div class=\"modebar-group\"><a rel=\"tooltip\" class=\"modebar-btn\" data-title=\"Download plot as a png\" data-toggle=\"false\" data-gravity=\"n\"><svg viewBox=\"0 0 1000 1000\" class=\"icon\" height=\"1em\" width=\"1em\"><path d=\"m500 450c-83 0-150-67-150-150 0-83 67-150 150-150 83 0 150 67 150 150 0 83-67 150-150 150z m400 150h-120c-16 0-34 13-39 29l-31 93c-6 15-23 28-40 28h-340c-16 0-34-13-39-28l-31-94c-6-15-23-28-40-28h-120c-55 0-100-45-100-100v-450c0-55 45-100 100-100h800c55 0 100 45 100 100v450c0 55-45 100-100 100z m-400-550c-138 0-250 112-250 250 0 138 112 250 250 250 138 0 250-112 250-250 0-138-112-250-250-250z m365 380c-19 0-35 16-35 35 0 19 16 35 35 35 19 0 35-16 35-35 0-19-16-35-35-35z\" transform=\"matrix(1 0 0 -1 0 850)\"></path></svg></a></div><div class=\"modebar-group\"><a rel=\"tooltip\" class=\"modebar-btn active\" data-title=\"Zoom\" data-attr=\"dragmode\" data-val=\"zoom\" data-toggle=\"false\" data-gravity=\"n\"><svg viewBox=\"0 0 1000 1000\" class=\"icon\" height=\"1em\" width=\"1em\"><path d=\"m1000-25l-250 251c40 63 63 138 63 218 0 224-182 406-407 406-224 0-406-182-406-406s183-406 407-406c80 0 155 22 218 62l250-250 125 125z m-812 250l0 438 437 0 0-438-437 0z m62 375l313 0 0-312-313 0 0 312z\" transform=\"matrix(1 0 0 -1 0 850)\"></path></svg></a><a rel=\"tooltip\" class=\"modebar-btn\" data-title=\"Pan\" data-attr=\"dragmode\" data-val=\"pan\" data-toggle=\"false\" data-gravity=\"n\"><svg viewBox=\"0 0 1000 1000\" class=\"icon\" height=\"1em\" width=\"1em\"><path d=\"m1000 350l-187 188 0-125-250 0 0 250 125 0-188 187-187-187 125 0 0-250-250 0 0 125-188-188 186-187 0 125 252 0 0-250-125 0 187-188 188 188-125 0 0 250 250 0 0-126 187 188z\" transform=\"matrix(1 0 0 -1 0 850)\"></path></svg></a><a rel=\"tooltip\" class=\"modebar-btn\" data-title=\"Box Select\" data-attr=\"dragmode\" data-val=\"select\" data-toggle=\"false\" data-gravity=\"n\"><svg viewBox=\"0 0 1000 1000\" class=\"icon\" height=\"1em\" width=\"1em\"><path d=\"m0 850l0-143 143 0 0 143-143 0z m286 0l0-143 143 0 0 143-143 0z m285 0l0-143 143 0 0 143-143 0z m286 0l0-143 143 0 0 143-143 0z m-857-286l0-143 143 0 0 143-143 0z m857 0l0-143 143 0 0 143-143 0z m-857-285l0-143 143 0 0 143-143 0z m857 0l0-143 143 0 0 143-143 0z m-857-286l0-143 143 0 0 143-143 0z m286 0l0-143 143 0 0 143-143 0z m285 0l0-143 143 0 0 143-143 0z m286 0l0-143 143 0 0 143-143 0z\" transform=\"matrix(1 0 0 -1 0 850)\"></path></svg></a><a rel=\"tooltip\" class=\"modebar-btn\" data-title=\"Lasso Select\" data-attr=\"dragmode\" data-val=\"lasso\" data-toggle=\"false\" data-gravity=\"n\"><svg viewBox=\"0 0 1031 1000\" class=\"icon\" height=\"1em\" width=\"1em\"><path d=\"m1018 538c-36 207-290 336-568 286-277-48-473-256-436-463 10-57 36-108 76-151-13-66 11-137 68-183 34-28 75-41 114-42l-55-70 0 0c-2-1-3-2-4-3-10-14-8-34 5-45 14-11 34-8 45 4 1 1 2 3 2 5l0 0 113 140c16 11 31 24 45 40 4 3 6 7 8 11 48-3 100 0 151 9 278 48 473 255 436 462z m-624-379c-80 14-149 48-197 96 42 42 109 47 156 9 33-26 47-66 41-105z m-187-74c-19 16-33 37-39 60 50-32 109-55 174-68-42-25-95-24-135 8z m360 75c-34-7-69-9-102-8 8 62-16 128-68 170-73 59-175 54-244-5-9 20-16 40-20 61-28 159 121 317 333 354s407-60 434-217c28-159-121-318-333-355z\" transform=\"matrix(1 0 0 -1 0 850)\"></path></svg></a></div><div class=\"modebar-group\"><a rel=\"tooltip\" class=\"modebar-btn\" data-title=\"Zoom in\" data-attr=\"zoom\" data-val=\"in\" data-toggle=\"false\" data-gravity=\"n\"><svg viewBox=\"0 0 875 1000\" class=\"icon\" height=\"1em\" width=\"1em\"><path d=\"m1 787l0-875 875 0 0 875-875 0z m687-500l-187 0 0-187-125 0 0 187-188 0 0 125 188 0 0 187 125 0 0-187 187 0 0-125z\" transform=\"matrix(1 0 0 -1 0 850)\"></path></svg></a><a rel=\"tooltip\" class=\"modebar-btn\" data-title=\"Zoom out\" data-attr=\"zoom\" data-val=\"out\" data-toggle=\"false\" data-gravity=\"n\"><svg viewBox=\"0 0 875 1000\" class=\"icon\" height=\"1em\" width=\"1em\"><path d=\"m0 788l0-876 875 0 0 876-875 0z m688-500l-500 0 0 125 500 0 0-125z\" transform=\"matrix(1 0 0 -1 0 850)\"></path></svg></a><a rel=\"tooltip\" class=\"modebar-btn\" data-title=\"Autoscale\" data-attr=\"zoom\" data-val=\"auto\" data-toggle=\"false\" data-gravity=\"n\"><svg viewBox=\"0 0 1000 1000\" class=\"icon\" height=\"1em\" width=\"1em\"><path d=\"m250 850l-187 0-63 0 0-62 0-188 63 0 0 188 187 0 0 62z m688 0l-188 0 0-62 188 0 0-188 62 0 0 188 0 62-62 0z m-875-938l0 188-63 0 0-188 0-62 63 0 187 0 0 62-187 0z m875 188l0-188-188 0 0-62 188 0 62 0 0 62 0 188-62 0z m-125 188l-1 0-93-94-156 156 156 156 92-93 2 0 0 250-250 0 0-2 93-92-156-156-156 156 94 92 0 2-250 0 0-250 0 0 93 93 157-156-157-156-93 94 0 0 0-250 250 0 0 0-94 93 156 157 156-157-93-93 0 0 250 0 0 250z\" transform=\"matrix(1 0 0 -1 0 850)\"></path></svg></a><a rel=\"tooltip\" class=\"modebar-btn\" data-title=\"Reset axes\" data-attr=\"zoom\" data-val=\"reset\" data-toggle=\"false\" data-gravity=\"n\"><svg viewBox=\"0 0 928.6 1000\" class=\"icon\" height=\"1em\" width=\"1em\"><path d=\"m786 296v-267q0-15-11-26t-25-10h-214v214h-143v-214h-214q-15 0-25 10t-11 26v267q0 1 0 2t0 2l321 264 321-264q1-1 1-4z m124 39l-34-41q-5-5-12-6h-2q-7 0-12 3l-386 322-386-322q-7-4-13-4-7 2-12 7l-35 41q-4 5-3 13t6 12l401 334q18 15 42 15t43-15l136-114v109q0 8 5 13t13 5h107q8 0 13-5t5-13v-227l122-102q5-5 6-12t-4-13z\" transform=\"matrix(1 0 0 -1 0 850)\"></path></svg></a></div><div class=\"modebar-group\"><a rel=\"tooltip\" class=\"modebar-btn\" data-title=\"Toggle Spike Lines\" data-attr=\"_cartesianSpikesEnabled\" data-val=\"on\" data-toggle=\"false\" data-gravity=\"n\"><svg viewBox=\"0 0 1000 1000\" class=\"icon\" height=\"1em\" width=\"1em\"><path d=\"M512 409c0-57-46-104-103-104-57 0-104 47-104 104 0 57 47 103 104 103 57 0 103-46 103-103z m-327-39l92 0 0 92-92 0z m-185 0l92 0 0 92-92 0z m370-186l92 0 0 93-92 0z m0-184l92 0 0 92-92 0z\" transform=\"matrix(1.5 0 0 -1.5 0 850)\"></path></svg></a><a rel=\"tooltip\" class=\"modebar-btn active\" data-title=\"Show closest data on hover\" data-attr=\"hovermode\" data-val=\"closest\" data-toggle=\"false\" data-gravity=\"ne\"><svg viewBox=\"0 0 1500 1000\" class=\"icon\" height=\"1em\" width=\"1em\"><path d=\"m375 725l0 0-375-375 375-374 0-1 1125 0 0 750-1125 0z\" transform=\"matrix(1 0 0 -1 0 850)\"></path></svg></a><a rel=\"tooltip\" class=\"modebar-btn\" data-title=\"Compare data on hover\" data-attr=\"hovermode\" data-val=\"x\" data-toggle=\"false\" data-gravity=\"ne\"><svg viewBox=\"0 0 1125 1000\" class=\"icon\" height=\"1em\" width=\"1em\"><path d=\"m187 786l0 2-187-188 188-187 0 0 937 0 0 373-938 0z m0-499l0 1-187-188 188-188 0 0 937 0 0 376-938-1z\" transform=\"matrix(1 0 0 -1 0 850)\"></path></svg></a></div><div class=\"modebar-group\"><a href=\"https://plotly.com/\" target=\"_blank\" data-title=\"Produced with Plotly\" class=\"modebar-btn plotlyjsicon modebar-btn--logo\"><svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 132 132\" height=\"1em\" width=\"1em\"><defs><style>.cls-1 {fill: #3f4f75;} .cls-2 {fill: #80cfbe;} .cls-3 {fill: #fff;}</style></defs><title>plotly-logomark</title><g id=\"symbol\"><rect class=\"cls-1\" width=\"132\" height=\"132\" rx=\"6\" ry=\"6\"></rect><circle class=\"cls-2\" cx=\"78\" cy=\"54\" r=\"6\"></circle><circle class=\"cls-2\" cx=\"102\" cy=\"30\" r=\"6\"></circle><circle class=\"cls-2\" cx=\"78\" cy=\"30\" r=\"6\"></circle><circle class=\"cls-2\" cx=\"54\" cy=\"30\" r=\"6\"></circle><circle class=\"cls-2\" cx=\"30\" cy=\"30\" r=\"6\"></circle><circle class=\"cls-2\" cx=\"30\" cy=\"54\" r=\"6\"></circle><path class=\"cls-3\" d=\"M30,72a6,6,0,0,0-6,6v24a6,6,0,0,0,12,0V78A6,6,0,0,0,30,72Z\"></path><path class=\"cls-3\" d=\"M78,72a6,6,0,0,0-6,6v24a6,6,0,0,0,12,0V78A6,6,0,0,0,78,72Z\"></path><path class=\"cls-3\" d=\"M54,48a6,6,0,0,0-6,6v48a6,6,0,0,0,12,0V54A6,6,0,0,0,54,48Z\"></path><path class=\"cls-3\" d=\"M102,48a6,6,0,0,0-6,6v48a6,6,0,0,0,12,0V54A6,6,0,0,0,102,48Z\"></path></g></svg></a></div></div></div><svg class=\"main-svg\" xmlns=\"http://www.w3.org/2000/svg\" xmlns:xlink=\"http://www.w3.org/1999/xlink\" width=\"985.533\" height=\"525\"><g class=\"hoverlayer\"></g></svg></div></div></div>")
    <script type="text/javascript">
        require(["plotly"], function(Plotly) {
            window.PLOTLYENV=window.PLOTLYENV || {};
            
        if (document.getElementById("c13947be-83de-4e12-b5cd-4b95eda7c1b7")) {
            Plotly.newPlot(
                'c13947be-83de-4e12-b5cd-4b95eda7c1b7',
                [{"alignmentgroup": "True", "bingroup": "x", "histfunc": "avg", "hovertemplate": "color=PC1<br>x=get_ipython().run_line_magic("{x}<br>avg", " of y=%{y}<extra></extra>\", \"legendgroup\": \"PC1\", \"marker\": {\"color\": \"rgb(102,194,165)\"}, \"name\": \"PC1\", \"offsetgroup\": \"PC1\", \"orientation\": \"v\", \"showlegend\": true, \"type\": \"histogram\", \"x\": [\"PC1\"], \"xaxis\": \"x\", \"y\": [0.18337912351866142], \"yaxis\": \"y\"}, {\"alignmentgroup\": \"True\", \"bingroup\": \"x\", \"histfunc\": \"avg\", \"hovertemplate\": \"color=PC2<br>x=%{x}<br>avg of y=%{y}<extra></extra>\", \"legendgroup\": \"PC2\", \"marker\": {\"color\": \"rgb(252,141,98)\"}, \"name\": \"PC2\", \"offsetgroup\": \"PC2\", \"orientation\": \"v\", \"showlegend\": true, \"type\": \"histogram\", \"x\": [\"PC2\"], \"xaxis\": \"x\", \"y\": [0.09716321269554316], \"yaxis\": \"y\"}, {\"alignmentgroup\": \"True\", \"bingroup\": \"x\", \"histfunc\": \"avg\", \"hovertemplate\": \"color=PC10<br>x=%{x}<br>avg of y=%{y}<extra></extra>\", \"legendgroup\": \"PC10\", \"marker\": {\"color\": \"rgb(141,160,203)\"}, \"name\": \"PC10\", \"offsetgroup\": \"PC10\", \"orientation\": \"v\", \"showlegend\": true, \"type\": \"histogram\", \"x\": [\"PC10\"], \"xaxis\": \"x\", \"y\": [0.09543751525875446], \"yaxis\": \"y\"}, {\"alignmentgroup\": \"True\", \"bingroup\": \"x\", \"histfunc\": \"avg\", \"hovertemplate\": \"color=PC11<br>x=%{x}<br>avg of y=%{y}<extra></extra>\", \"legendgroup\": \"PC11\", \"marker\": {\"color\": \"rgb(231,138,195)\"}, \"name\": \"PC11\", \"offsetgroup\": \"PC11\", \"orientation\": \"v\", \"showlegend\": true, \"type\": \"histogram\", \"x\": [\"PC11\"], \"xaxis\": \"x\", \"y\": [0.09405545095230985], \"yaxis\": \"y\"}, {\"alignmentgroup\": \"True\", \"bingroup\": \"x\", \"histfunc\": \"avg\", \"hovertemplate\": \"color=PC3<br>x=%{x}<br>avg of y=%{y}<extra></extra>\", \"legendgroup\": \"PC3\", \"marker\": {\"color\": \"rgb(166,216,84)\"}, \"name\": \"PC3\", \"offsetgroup\": \"PC3\", \"orientation\": \"v\", \"showlegend\": true, \"type\": \"histogram\", \"x\": [\"PC3\"], \"xaxis\": \"x\", \"y\": [0.0884487029181083], \"yaxis\": \"y\"}, {\"alignmentgroup\": \"True\", \"bingroup\": \"x\", \"histfunc\": \"avg\", \"hovertemplate\": \"color=PC8<br>x=%{x}<br>avg of y=%{y}<extra></extra>\", \"legendgroup\": \"PC8\", \"marker\": {\"color\": \"rgb(255,217,47)\"}, \"name\": \"PC8\", \"offsetgroup\": \"PC8\", \"orientation\": \"v\", \"showlegend\": true, \"type\": \"histogram\", \"x\": [\"PC8\"], \"xaxis\": \"x\", \"y\": [0.08464905252001484], \"yaxis\": \"y\"}, {\"alignmentgroup\": \"True\", \"bingroup\": \"x\", \"histfunc\": \"avg\", \"hovertemplate\": \"color=PC9<br>x=%{x}<br>avg of y=%{y}<extra></extra>\", \"legendgroup\": \"PC9\", \"marker\": {\"color\": \"rgb(229,196,148)\"}, \"name\": \"PC9\", \"offsetgroup\": \"PC9\", \"orientation\": \"v\", \"showlegend\": true, \"type\": \"histogram\", \"x\": [\"PC9\"], \"xaxis\": \"x\", \"y\": [0.07722466767359072], \"yaxis\": \"y\"}, {\"alignmentgroup\": \"True\", \"bingroup\": \"x\", \"histfunc\": \"avg\", \"hovertemplate\": \"color=PC7<br>x=%{x}<br>avg of y=%{y}<extra></extra>\", \"legendgroup\": \"PC7\", \"marker\": {\"color\": \"rgb(179,179,179)\"}, \"name\": \"PC7\", \"offsetgroup\": \"PC7\", \"orientation\": \"v\", \"showlegend\": true, \"type\": \"histogram\", \"x\": [\"PC7\"], \"xaxis\": \"x\", \"y\": [0.07346918751891697], \"yaxis\": \"y\"}, {\"alignmentgroup\": \"True\", \"bingroup\": \"x\", \"histfunc\": \"avg\", \"hovertemplate\": \"color=PC5<br>x=%{x}<br>avg of y=%{y}<extra></extra>\", \"legendgroup\": \"PC5\", \"marker\": {\"color\": \"rgb(102,194,165)\"}, \"name\": \"PC5\", \"offsetgroup\": \"PC5\", \"orientation\": \"v\", \"showlegend\": true, \"type\": \"histogram\", \"x\": [\"PC5\"], \"xaxis\": \"x\", \"y\": [0.07028305441673129], \"yaxis\": \"y\"}, {\"alignmentgroup\": \"True\", \"bingroup\": \"x\", \"histfunc\": \"avg\", \"hovertemplate\": \"color=PC4<br>x=%{x}<br>avg of y=%{y}<extra></extra>\", \"legendgroup\": \"PC4\", \"marker\": {\"color\": \"rgb(252,141,98)\"}, \"name\": \"PC4\", \"offsetgroup\": \"PC4\", \"orientation\": \"v\", \"showlegend\": true, \"type\": \"histogram\", \"x\": [\"PC4\"], \"xaxis\": \"x\", \"y\": [0.06925201501282743], \"yaxis\": \"y\"}, {\"alignmentgroup\": \"True\", \"bingroup\": \"x\", \"histfunc\": \"avg\", \"hovertemplate\": \"color=PC6<br>x=%{x}<br>avg of y=%{y}<extra></extra>\", \"legendgroup\": \"PC6\", \"marker\": {\"color\": \"rgb(141,160,203)\"}, \"name\": \"PC6\", \"offsetgroup\": \"PC6\", \"orientation\": \"v\", \"showlegend\": true, \"type\": \"histogram\", \"x\": [\"PC6\"], \"xaxis\": \"x\", \"y\": [0.06663801751454172], \"yaxis\": \"y\"}],")
                {"barmode": "relative", "legend": {"title": {"text": "color"}, "tracegroupgap": 0}, "margin": {"t": 60}, "showlegend": false, "template": {"data": {"bar": [{"error_x": {"color": "#2a3f5f"}, "error_y": {"color": "#2a3f5f"}, "marker": {"line": {"color": "#E5ECF6", "width": 0.5}}, "type": "bar"}], "barpolar": [{"marker": {"line": {"color": "#E5ECF6", "width": 0.5}}, "type": "barpolar"}], "carpet": [{"aaxis": {"endlinecolor": "#2a3f5f", "gridcolor": "white", "linecolor": "white", "minorgridcolor": "white", "startlinecolor": "#2a3f5f"}, "baxis": {"endlinecolor": "#2a3f5f", "gridcolor": "white", "linecolor": "white", "minorgridcolor": "white", "startlinecolor": "#2a3f5f"}, "type": "carpet"}], "choropleth": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "choropleth"}], "contour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "contour"}], "contourcarpet": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "contourcarpet"}], "heatmap": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmap"}], "heatmapgl": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmapgl"}], "histogram": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "histogram"}], "histogram2d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2d"}], "histogram2dcontour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2dcontour"}], "mesh3d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "mesh3d"}], "parcoords": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "parcoords"}], "pie": [{"automargin": true, "type": "pie"}], "scatter": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatter"}], "scatter3d": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatter3d"}], "scattercarpet": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattercarpet"}], "scattergeo": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattergeo"}], "scattergl": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattergl"}], "scattermapbox": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattermapbox"}], "scatterpolar": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolar"}], "scatterpolargl": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolargl"}], "scatterternary": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterternary"}], "surface": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "surface"}], "table": [{"cells": {"fill": {"color": "#EBF0F8"}, "line": {"color": "white"}}, "header": {"fill": {"color": "#C8D4E3"}, "line": {"color": "white"}}, "type": "table"}]}, "layout": {"annotationdefaults": {"arrowcolor": "#2a3f5f", "arrowhead": 0, "arrowwidth": 1}, "coloraxis": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "colorscale": {"diverging": [[0, "#8e0152"], [0.1, "#c51b7d"], [0.2, "#de77ae"], [0.3, "#f1b6da"], [0.4, "#fde0ef"], [0.5, "#f7f7f7"], [0.6, "#e6f5d0"], [0.7, "#b8e186"], [0.8, "#7fbc41"], [0.9, "#4d9221"], [1, "#276419"]], "sequential": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "sequentialminus": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]]}, "colorway": ["#636efa", "#EF553B", "#00cc96", "#ab63fa", "#FFA15A", "#19d3f3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52"], "font": {"color": "#2a3f5f"}, "geo": {"bgcolor": "white", "lakecolor": "white", "landcolor": "#E5ECF6", "showlakes": true, "showland": true, "subunitcolor": "white"}, "hoverlabel": {"align": "left"}, "hovermode": "closest", "mapbox": {"style": "light"}, "paper_bgcolor": "white", "plot_bgcolor": "#E5ECF6", "polar": {"angularaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "bgcolor": "#E5ECF6", "radialaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}}, "scene": {"xaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}, "yaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}, "zaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}}, "shapedefaults": {"line": {"color": "#2a3f5f"}}, "ternary": {"aaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "baxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "bgcolor": "#E5ECF6", "caxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}}, "title": {"x": 0.05}, "xaxis": {"automargin": true, "gridcolor": "white", "linecolor": "white", "ticks": "", "title": {"standoff": 15}, "zerolinecolor": "white", "zerolinewidth": 2}, "yaxis": {"automargin": true, "gridcolor": "white", "linecolor": "white", "ticks": "", "title": {"standoff": 15}, "zerolinecolor": "white", "zerolinewidth": 2}}}, "title": {"text": "Most important features in Random forest classification"}, "xaxis": {"anchor": "y", "domain": [0.0, 1.0], "title": {"text": "Features"}}, "yaxis": {"anchor": "x", "domain": [0.0, 1.0], "title": {"text": "Gini importance"}}},
                {"responsive": true}
            ).then(function(){
                    
var gd = document.getElementById('c13947be-83de-4e12-b5cd-4b95eda7c1b7');
var x = new MutationObserver(function (mutations, observer) {{
var display = window.getComputedStyle(gd).display;
if (get_ipython().getoutput("display || display === 'none') {{")
    console.log([gd, 'removedget_ipython().getoutput("']);")
    Plotly.purge(gd);
    observer.disconnect();
}}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
x.observe(outputEl, {childList: true});
}}

                })
        };
        });
    </script>
</div>
        """, raw=True)



from sklearn.ensemble import RandomForestClassifier
for oversampling in [True, False]:
    parameter_grid = {
        "criterion":["gini", "entropy"],
        "max_features":[None, "sqrt"],
        "oob_score":[True],
        "n_estimators":[10, 50, 100, 200]

    }
    classifier = RandomForestClassifier
    K = 5

    results_rf, parameter_rf, s = train_and_validate(X_15d_train_val, y_train_val, classifier, 'rf', parameter_grid, oversampling=oversampling)
    
    if oversampling:
        di.display_html("""
        <h1>With oversampling - SMOTE</h1>
        """, raw=True)
    else:
        di.display_html("""
        <h1>With undersampling - Cluster Centroid</h1>
        """, raw=True)
    #di.display_html("""
    #<p style='margin-bottom: 1em;font-size:15px'>
    #    """ + s + """
    #</p>
    #""", raw=True)
    
    # evaluating the most important features for classification
    _, _, _, best_configuration, s, _, _, _ = find_best_configuration(results_rf, parameter_rf)
    
    if oversampling:
        X_train_val_balanced, y_train_val_balanced, _  = oversample_dataset(X_15d_train_val, y_train_val)
        name = './results_oversampling/'
    else:
        X_train_val_balanced, y_train_val_balanced, _  = undersample_dataset(X_15d_train_val, y_train_val)
        name = './results_undersampling/'

    classifier = RandomForestClassifier(**best_configuration)
    classifier.fit(X_train_val_balanced, y_train_val_balanced)

    fts = classifier.feature_importances_

    di.display_html("""

    <p style='margin-bottom: 1em;font-size:15px'>
        """ + s + """
    </p>
    """, raw=True)
    
    fig = px.histogram(
    x = X_15d_train_val.columns[fts.argsort()[::-1]], 
    y = fts[fts.argsort()[::-1]],
    histfunc = 'avg',
    color = X_15d_train_val.columns[fts.argsort()[::-1]],
    color_discrete_sequence = px.colors.qualitative.Set2,

    )

    fig.update_layout(
        showlegend=False,
        title = "Most important features in Random forest classification",
        xaxis_title= "Features",
        yaxis_title="Gini importance",
    )
    fig.show()

    with open(name + 'important_fts.json', 'w') as f:
          json.dump(list(fts) , f)
            
    classifier = RandomForestClassifier
    clf_name = 'rf'
    results = results_rf
    parameters = parameter_rf

    s, report = test(X_15d_train_val, X_15d_test,y_test, classifier, clf_name, results, parameters, oversampling=oversampling)

    print_result(s, report)
    
    from sklearn.tree import plot_tree
    fig_tree, ax_tree = plt.subplots(figsize=(20,9))

    plot_tree(
        classifier.estimators_[0],
        feature_names=X.columns,
        filled=True,
        rounded=True,
        fontsize=14
    );


from sklearn.ensemble import AdaBoostClassifier

for oversampling in [True, False]:
    parameter_grid = {
        "learning_rate":[0.01, 0.05, 0.1, 0.5],
        "n_estimators":[2, 50, 100]

    }
    K = 5
    classifier = AdaBoostClassifier

    results_ada, parameter_ada, s = train_and_validate(X_15d_train_val, y_train_val,classifier, 'ada', parameter_grid, oversampling=oversampling)
    if oversampling:
        di.display_html("""
        <h1>With oversampling - SMOTE</h1>
        """, raw=True)
    else:
        di.display_html("""
        <h1>With undersampling - Cluster Centroid</h1>
        """, raw=True)
    #di.display_html("""
    #<p style='margin-bottom: 1em;font-size:15px'>
    #    """ + s + """
    #</p>
    #""", raw=True)
    
    classifier = AdaBoostClassifier
    clf_name = 'ada'
    results = results_ada
    parameters = parameter_ada

    s, report = test(X_15d_train_val, X_15d_test,y_test, classifier, clf_name, results, parameters, oversampling=oversampling)

    print_result(s, report)


from sklearn.neighbors import KNeighborsClassifier
for oversampling in [True, False]:
    parameter_grid = {
        "n_neighbors":[500, 800, 1500, 2500, 3500, 4500]
    }
    classifier = KNeighborsClassifier
    K = 5

    results_knn, parameter_knn, s = train_and_validate(X_15d_train_val, y_train_val, classifier, 'knn', parameter_grid, oversampling=oversampling)
    if oversampling:
        di.display_html("""
        <h1>With oversampling - SMOTE</h1>
        """, raw=True)
    else:
        di.display_html("""
        <h1>With undersampling - Cluster Centroid</h1>
        """, raw=True)
    #di.display_html("""
    #<p style='margin-bottom: 1em;font-size:15px'>
    #   """ + s + """
    #</p>
    #""", raw=True)
    
    classifier = KNeighborsClassifier
    clf_name = 'knn'
    results = results_knn
    parameters = parameter_knn

    s, report = test(X_15d_train_val, X_15d_test,y_test, classifier, clf_name, results, parameters, oversampling=oversampling)

    print_result(s, report)


from sklearn.linear_model import LogisticRegression
for oversampling in [True, False]:
    parameter_grid = {
        "C":[0.0001, 0.001, 0.01, 0.1, 1, 10]
    }
    classifier = LogisticRegression
    K = 5

    results_lr, parameter_lr, s = train_and_validate(X_15d_train_val, y_train_val, classifier, 'lr', parameter_grid, oversampling=oversampling)
    if oversampling:
        di.display_html("""
        <h1>With oversampling - SMOTE</h1>
        """, raw=True)
    else:
        di.display_html("""
        <h1>With undersampling - Cluster Centroid</h1>
        """, raw=True)
    #di.display_html("""
    #<p style='margin-bottom: 1em;font-size:15px'>
    #    """ + s + """
    #</p>
    #""", raw=True)
    classifier = LogisticRegression
    clf_name = 'lr'
    results = results_lr
    parameters = parameter_lr

    s, report = test(X_15d_train_val, X_15d_test,y_test, classifier, clf_name, results, parameters, oversampling=oversampling)

    print_result(s, report)


def load_results(algorithms, alg_names, path):
    import urllib.request, json 
    r = ""
    
    list_results_algorithms = []
    for i, algorithm in enumerate(algorithms):
        df_alg_params = pd.DataFrame()
        
        try:
            with urllib.request.urlopen(path + algorithm + '_results_train.json') as url: #results/svm_results_train.json
                data = json.loads(url.read().decode())
                scores_train = data[0]
                params_train = data[1]
        except FileNotFoundError:
            pass
        try:  
            with urllib.request.urlopen(path + algorithm + '_results_test.json') as url:
                data = json.loads(url.read().decode())
                scores_test = data[0]
        except FileNotFoundError:
            pass

        d1 = pd.DataFrame.from_dict(scores_train, orient='columns')
        d2 = pd.DataFrame.from_dict(params_train, orient='columns')
        df_alg_params = pd.concat([df_alg_params, d2])
        df_results = pd.concat([d1, d2], axis=1)
        list_results_algorithms.append(df_results)
        if "over" in path:
            n = "SMOTE"
        else:
            n = "Cluster centroid"
        
        accuracy, f1_0, f1_1, params, _, std_f1_1, std_f1_0, std_accuracy = find_best_configuration(scores_train, params_train, display=False)
        if i == 0:
            
            r += "<tr class = \"content\"><td class=\"third\" style=\"width:8get_ipython().run_line_magic("\"", " rowspan=\" + str(len(alg_names)) + \">\" + n + \"</td><td>\" + alg_names[i]+ \"</td><td>\" + str(round(accuracy, 2)) +\" +/- \" +str(round(std_accuracy, 2)) + \"</td><td>\"+ str(round(f1_1, 2))+\" +/- \" +str(round(std_f1_1, 2)) +\"</td><td>\"+ str(round(f1_0,2)) +\" +/- \" +str(round(std_f1_0, 2)) +\"</td>\"")
        else:
            r += "<tr class = \"content\"><td>" + alg_names[i]+ "</td><td>" + str(round(accuracy, 2)) + " +/- " +str(round(std_accuracy, 2)) +"</td><td>"+ str(round(f1_1, 2))+" +/- " +str(round(std_f1_1, 2)) +"</td><td>"+ str(round(f1_0,2)) +" +/- " +str(round(std_f1_0, 2)) +"</td>"
        r += "<td>" + str(round(scores_test['accuracy'], 2)) + "</td><td>"+ str(round(scores_test['fscore'][1], 2))+"</td><td>"+ str(round(scores_test['fscore'][0], 2)) +"</td></tr>"
    return list_results_algorithms, r

algorithms = ['svm', 'rf', 'knn', 'ada', 'lr']
alg_names = ['Support Vector Machine', 'Random Forest', 'K-Nearest Neighbor', 'AdaBoost', 'Logistic regression']
path = 'https://raw.githubusercontent.com/irenebenedetto/default-of-credit-card-clients/master/results/'
list_results_algorithms_over, r_over = load_results(algorithms, alg_names, path + 'results_oversampling/')
list_results_algorithms_under, r_under = load_results(algorithms, alg_names, path + 'results_undersampling/')


columns = [ '0_precision', '1_precision', '0_recall', '1_recall', 
            'std_0_precision', 'std_1_precision','std_0_recall', 'std_1_recall']

for oversampling in [True, False]:
    if oversampling:
        df_plot = list_results_algorithms_over[0].groupby(['C', 'kernel']).mean()[columns]
        note = 'SMOTE - '
    else:
        df_plot = list_results_algorithms_under[0].groupby(['C', 'kernel']).mean()[columns]
        note = 'CLUSTER CENTROID - '
    h = 1
    j = 1
    subplots = make_subplots(
            rows=1, cols=2,
            subplot_titles=['Kernel: '+str(g) for g in np.unique([j for _, j in df_plot.index.values])],
        )
    subplots.update_layout(
            title = note + "SVM Scores trend across different C",
        height= 350, 
         )
    subplots.update_xaxes( type="log", title='C')
    subplots.update_yaxes(title='Score')
    for g in np.unique([j for _, j in df_plot.index.values]):
        fig = go.Figure()
        if h == 3:
            h = 1
            j = j+1
        
        i = 0
        for col, c in zip(columns[:4], px.colors.qualitative.Set2):
            if h == 1 and j == 1:
                legend = True
            else:
                legend = False
            fig.add_trace(
                go.Scatter(
                    x=np.unique([i for i, _ in df_plot.index.values]), 
                    y=[round(df_plot.loc[(i, g)][col], 2) for i in np.unique([i for i, _ in df_plot.index.values])] ,
                    showlegend = legend,
                    error_y=dict(
                        #type='data', 
                        array=df_plot[columns[i+4]],
                        visible=True,

                        ),
                    mode="lines+markers+text",
                    name = col,
                    text = [round(df_plot.loc[(i, g)][col], 2) for i in np.unique([i for i, _ in df_plot.index.values])],
                    textposition="bottom center",
                    
                    textfont=dict(
                        size=10,
                    ),

                    line = dict(
                        width=2,

                    ),
                    marker = dict(
                        color = c,
                    )
                )
            )
            i = i+1
            fig.update_xaxes(showspikes=True, type="log")
            fig.update_yaxes(showspikes=True, )

        subplots.add_trace(fig.data[0] , row=j, col=h)
        subplots.add_trace(fig.data[1] , row=j, col=h)
        subplots.add_trace(fig.data[2] , row=j, col=h)
        subplots.add_trace(fig.data[3] , row=j, col=h)
        
        h =h+1
    subplots.show()


columns = [ '0_precision', '1_precision', '0_recall', '1_recall', 
            'std_0_precision', 'std_1_precision','std_0_recall', 'std_1_recall']
h = 1
j = 1
subplots = make_subplots(
        rows=1, cols=2, vertical_spacing=0.1, horizontal_spacing=0.05,
        subplot_titles=['SMOTE', 'CLUSTER CENTROID']
    )
subplots.update_layout(
        title = "Random forest scores trend across different number of estimators",
        height= 350, 
     )
subplots.update_xaxes(title='n_estimators')
subplots.update_yaxes(title='Score')
for oversampling in [True, False]:
    if oversampling:
        df_plot = list_results_algorithms_over[1].groupby(['n_estimators']).mean()[columns]
        note = 'SMOTE - '
    else:
        df_plot = list_results_algorithms_under[1].groupby(['n_estimators']).mean()[columns]
        note = 'CLUSTER CENTROID - '

    fig = go.Figure()
    if h == 1 and j ==1:
        legend = True
    else:
        legend = False
    for i, c in enumerate(columns[:4]):
        fig.add_trace(
            go.Scatter(
                x=df_plot.index.values, 
                y=df_plot[c],
                mode="lines+markers+text",
                name = c,
                showlegend=legend,
                text = [round(i, 3) for i in df_plot[c]],
                textposition="bottom center",
                error_y=dict(
                    #type='data', 
                    array=df_plot[columns[i+4]],
                    visible=True,

                ),
                textfont=dict(
                        size=10,
                    ),

                line = dict(
                    width=2,

                ),
                marker = dict(
                    color = px.colors.qualitative.Set2[i],
                )

            )
        )

    fig.update_xaxes(showspikes=True, type="log")
    fig.update_yaxes(showspikes=True)
    subplots.add_trace(fig.data[0] , row=j, col=h)
    subplots.add_trace(fig.data[1] , row=j, col=h)
    subplots.add_trace(fig.data[2] , row=j, col=h)
    subplots.add_trace(fig.data[3] , row=j, col=h)
    h =h+1
subplots.show()


columns = [ '0_precision', '1_precision', '0_recall', '1_recall', 
            'std_0_precision', 'std_1_precision','std_0_recall', 'std_1_recall']
h = 1
j = 1
subplots = make_subplots(
        rows=1, cols=2, 
        subplot_titles=['SMOTE', 'CLUSTER CENTROID']
)
subplots.update_layout(
        title =  "AdaBoost Scores trend across different number of estimators",
        height= 350, 
    )
subplots.update_xaxes(title='n_estimators')
subplots.update_yaxes(title='Score')

for oversampling in [True, False]:
    if oversampling:
        df_plot = list_results_algorithms_over[3].groupby(['n_estimators']).mean()[columns]
        note = 'SMOTE - '
        
    else:
        df_plot = list_results_algorithms_under[3].groupby(['n_estimators']).mean()[columns]
        note = 'CLUSTER CENTROID - '
    fig = go.Figure()
    for i, c in enumerate(columns[:4]):
        if h == 1 and j == 1:
            legend = True
        else:
            legend = False
            
        fig.add_trace(
            go.Scatter(
                x=df_plot.index.values, 
                y=df_plot[c],
                mode="lines+markers+text",
                name = c,
                showlegend=legend,
                text = [round(i, 3) for i in df_plot[c]],
                textposition="bottom center",
                error_y=dict(
                    array=df_plot[columns[i+4]],
                    visible=True,

                ),
                textfont=dict(
                        size=10,
                    ),

                line = dict(
                    width=2,

                ),
                marker = dict(
                    color = px.colors.qualitative.Set2[i],
                )
             )
        )
        
        fig.update_xaxes(showspikes=True, type="log")
        fig.update_yaxes(showspikes=True, )

    subplots.add_trace(fig.data[0] , row=j, col=h)
    subplots.add_trace(fig.data[1] , row=j, col=h)
    subplots.add_trace(fig.data[2] , row=j, col=h)
    subplots.add_trace(fig.data[3] , row=j, col=h)
    h =h+1
subplots.show()


columns = [ '0_precision', '1_precision', '0_recall', '1_recall', 
            'std_0_precision', 'std_1_precision','std_0_recall', 'std_1_recall']
h = 1
j = 1
subplots = make_subplots(
        rows=1, cols=2, 
        subplot_titles=['SMOTE', 'CLUSTER CENTROID']
    )
subplots.update_layout(
        title = "K-NN Scores trend across different number of neighbors",
        height= 350, 
     )
subplots.update_xaxes(title='n_neighbor')
subplots.update_yaxes(title='Score')
for oversampling in [True, False]:
    if oversampling:
        df_plot = list_results_algorithms_over[2].groupby(['n_neighbors']).mean()[columns]
        note = 'SMOTE - '
    else:
        df_plot = list_results_algorithms_under[2].groupby(['n_neighbors']).mean()[columns]
        note = 'CLUSTER CENTROID - '
        
    fig = go.Figure()
    
    
    for i, c in enumerate(columns[:4]):
        if h == 3:
            h = 1
            j = j+1
        if h == 1 and j == 1:
                legend = True
        else:
            legend = False
        fig.add_trace(
            go.Scatter(
                x=df_plot.index.values, 
                y=df_plot[c],
                showlegend=legend,
                error_y=dict(
                    #type='data', 
                    array=df_plot[columns[i+4]],
                    visible=True,

                ),
                mode="lines+markers+text",
                name = c,
                text = [round(i, 2) for i in df_plot[c]],
                textposition="bottom center",

                textfont=dict(
                        size=10,
                    ),

                line = dict(
                    width=2,

                ),
                marker = dict(
                    color = px.colors.qualitative.Set2[i],
                )

            )
        )

    subplots.add_trace(fig.data[0] , row=j, col=h)
    subplots.add_trace(fig.data[1] , row=j, col=h)
    subplots.add_trace(fig.data[2] , row=j, col=h)
    subplots.add_trace(fig.data[3] , row=j, col=h)
    h =h+1
subplots.show()


columns = [ '0_precision', '1_precision', '0_recall', '1_recall', 
            'std_0_precision', 'std_1_precision','std_0_recall', 'std_1_recall']
h = 1
j = 1
subplots = make_subplots(
        rows=1, cols=2, 
        subplot_titles=['SMOTE', 'CLUSTER CENTROID']
    )
subplots.update_layout(
        title = "Logistic regression scores trend across different C",
        height= 350, 
     )
subplots.update_xaxes( type="log", title='C')
subplots.update_yaxes(title='Score')
for oversampling in [True, False]:
    if oversampling:
        df_plot = list_results_algorithms_over[4].groupby(['C']).mean()[columns]
        note = 'SMOTE - '
    else:
        df_plot = list_results_algorithms_under[4].groupby(['C']).mean()[columns]
        note = 'CLUSTER CENTROID - '
        
    fig = go.Figure()

    for i, c in enumerate(columns[:4]):
        if h == 3:
            h = 1
            j = j+1
        if h == 1 and j == 1:
                legend = True
        else:
            legend = False
            
        fig.add_trace(
            go.Scatter(
                x=df_plot.index.values, 
                y=df_plot[c],
                error_y=dict(
                    #type='data', 
                    array=df_plot[columns[i+4]],
                    visible=True,

                ),
                showlegend=legend,
                mode="lines+markers+text",
                name = c,
                text = [round(i, 2) for i in df_plot[c]],
                textposition="bottom center",
                textfont=dict(
                            size=10,
                        ),

                line = dict(
                    width=2,

                ),
                marker = dict(
                    color = px.colors.qualitative.Set2[i],
                )

            )
        )

    fig.update_xaxes(showspikes=True, type="log")
    fig.update_yaxes(showspikes=True)
    subplots.add_trace(fig.data[0] , row=j, col=h)
    subplots.add_trace(fig.data[1] , row=j, col=h)
    subplots.add_trace(fig.data[2] , row=j, col=h)
    subplots.add_trace(fig.data[3] , row=j, col=h)
    h =h+1
subplots.show()


import urllib.request, json 
algorithms= ['svm', 'rf', 'knn', 'ada', 'lr']
alg_names = ['Support Vector Machine', 'Random Forest', 'K-Nearest Neighbor', 'AdaBoost', 'Logistic regression']
paths = ['https://raw.githubusercontent.com/irenebenedetto/default-of-credit-card-clients/master/results/results_oversampling/', 
         'https://raw.githubusercontent.com/irenebenedetto/default-of-credit-card-clients/master/results/results_undersampling/']
d = {'Algorithms':[], 'Method':[], 'F1 on positive class':[], 'F1 on negative class':[], 'Accuracy':[], 'std(F1 on positive class)':[], 'std(F1 on negative class)':[]}
for path in paths:
    r = ""
    list_results_algorithms = []
    for i, algorithm in enumerate(algorithms):
        df_alg_params = pd.DataFrame()
        
        try:
            with urllib.request.urlopen(path + algorithm + '_results_train.json') as url:  #results/svm_results_train.json
                data = json.loads(url.read().decode())
                scores_train = data[0]
                params_train = data[1]
        except FileNotFoundError:
            pass
        try:  
            with urllib.request.urlopen(path + algorithm + '_results_train.json') as url:
                data = json.loads(url.read().decode())
                scores_test = data[0]
        except FileNotFoundError:
            pass

        d1 = pd.DataFrame.from_dict(scores_train, orient='columns')
        d2 = pd.DataFrame.from_dict(params_train, orient='columns')
        df_alg_params = pd.concat([df_alg_params, d2])
        df_results = pd.concat([d1, d2], axis=1)
        list_results_algorithms.append(df_results)
        if "over" in path:
            n = "SMOTE"
        else:
            n = "Cluster centroid"
        
        accuracy, f1_0, f1_1, params, _,std_accuracy, std_f1_1, std_f1_0 = find_best_configuration(scores_train, params_train, display=False)
        d['Algorithms'].append(alg_names[i])
        d['F1 on positive class'].append(f1_1)
        d['F1 on negative class'].append(f1_0)
        d['std(F1 on positive class)'].append(std_f1_1)
        d['std(F1 on negative class)'].append(std_f1_0)
        d['Method'].append(n)
        d['Accuracy'].append(accuracy)
        
df_plot = pd.DataFrame.from_dict(d)
fig = px.bar(
    df_plot, 
    x="Algorithms", 
    y="F1 on positive class",
    error_y="std(F1 on positive class)",
    color='Method', 
    barmode='group',
    color_discrete_sequence=px.colors.qualitative.Set2,
    height=400, 
    title="F1 on positive class over different configuration - mean on validation sets")
fig.show()
fig = px.bar(
    df_plot, 
    x="Algorithms", 
    y="F1 on negative class",
    error_y="std(F1 on negative class)",
    color='Method', 
    barmode='group',
    color_discrete_sequence=px.colors.qualitative.Set2,
    height=400,  
    title="F1 on negative class over different configuration - mean on validation sets")
fig.show()



di.display_html("""


<table id="customers">
    <thead>
        <tr class="first">
            <th style=\"width:8get_ipython().run_line_magic("\"></th>", "")
            <th></th>
            <th colspan="3">Results on validation set</th>
            <th colspan="3">Results on test set</th>
        </tr>
        
        <tr class="second">
            <th style=\"width:8get_ipython().run_line_magic("\"></th>", "")
            <th>Algorithm</th>
            <th>Accuracy</th>
            <th>F1 score on positive class</th>
            <th>F1 score on negative class</th>
        
            <th>Accuracy</th>
            <th>F1 score on positive class</th>
            <th>F1 score on negative class</th>
        </tr>
    </thead>
    <tbody>
""" + r_over + r_under +"""
    </tbody>
</table>

""", raw=True)



