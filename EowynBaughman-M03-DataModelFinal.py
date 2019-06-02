# -*- coding: utf-8 -*-
"""
Created on Sat Mar 10 2018

@author: eowyn baughman

ASSIGNMENT SPECIFICATIONS:
Data Model and Evaluation
Specifically, you’ll need to pick two (or more) classifiers and perform 
each one of the following tasks, in order:

1. Split your dataset into training and testing sets - DONE
2. Train your classifiers, using the training set partition - DONE
3. Apply your (trained) classifiers on the test set - DONE
4. Measure each classifier’s performance using at least 3 of the  
metrics we covered in this course (one of them has to be the ROC-based one). DONE
5. At one point, you’ll need to create a confusion matrix. - DONE
6. Document your results and your conclusions, along with any 
relevant comments about your work - DONE
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import *
from copy import deepcopy
from sklearn.feature_selection import *
import matplotlib.pyplot as plt
from matplotlib.pyplot import *
"""
FEATURE ENGINEERING

Importing the data, cleaning it, removing outliers, 
imputing missing values, decoding categorical values, and 
dropping unnecessary values. 

1. Account for aberrant data (missing and outlier values).
2. Normalize numeric values (at least 1 column).
3. Bin categorical variables (at least 1 column).
4. Construct new categorical variables.
5. Remove obsolete columns.

The Hepatitis dataset contains entirely numeric data, with missing
values denoted by "?" symbol. The cleaned data has first had NaN values
substitued for ?. Then, ignoring NaNs, the upper and lower limits for
valid data are calculated as the mean plus/minus 2 standard deviations,
respectively, which represents the majority of normally distributed
data. Variables that have a small number of unique values are likely
to be ordinal data for which outlier detection is inappropriate. In
that case, no attempt is made to replace outliers with NaN.

13 variables represent "Yes", "No" type data for various medical symptoms
such as, "Is the spleen palpable, yes or no?". For these variables, the
original 1/2 coding is replaced by boolean 0/1 encoding, in place.

Outliers are those values which are less than the lower limit
or greater than the upper limit. All NaN values AND all outliers have
been replaced by the median of the "good" (ie, not NaN and not outlier)
values. These data cleaning steps are all done on a per-column basis
so that the different magnitudes of each metric can be considered
independently. The resulting cleaned data is stored in a dataframe
cleaned_hep_df and the oringal, unclean dataframe is hep_df.

The "Sex" column has been decoded to separate dummy binary columns
"Male" and "Female", and the obsolete "Sex" column is dropped. The "Sex"
column had no missing values to impute.

No variables were suitable for consolidation: LiverBig and LiverFirm was a
a candidate but their mean difference is -0.22 with a std of 0.44 which 
suggested consolidation would obliterate useful information. Similar
indication with Fatigue and Malaise. Based on background reading, those
were the only two variables with the possibility of consolidation.

 According to the data description and   http://www.hepatitiscentral.com/
 BILIRUBIN: 0.39, 0.80, 1.20, 2.00, 3.00, 4.00 -- A breakdown product 
 of hemaglobin; 1.1 mg/deciliter or lower is normal. Over 3mg/dl,
 jaundice is visible. Elevated bilirubin is a diagnostic criteria
, BILIRUBIN is a breakdown product of hemaglobin; 1.1 mg/deciliter
 or lower is normal. Over 3mg/dl, jaundice is visible. Elevated 
 bilirubin is a diagnostic criteria. I therefore binned the 
 variable between min - 1.1, 1- 3, and 3 - max to represent
 normal, elevated, jaundiced.

Reference: http://www.hepatitiscentral.com/ 
"""
#  Given a pandas series, return series with outliers replaced by nan
#  For series with few unique variables, do nothing and return the series
#  The hepatitis dataset has many columns with 2 or 3 unique variables
def replace_outliers_with_na(x):
    if len(x.unique()) < 5:
        return x
    xbar = np.mean(x) # Mean, ignoring NA
    xsd = np.std(x) # Standard deivation, ignoring NA
    LL = xbar - 2*xsd # Lower limit for outlier detection
    UL = xbar + 2*xsd # Upper limit for outlier detection
    return x.map(lambda y: y if y > LL and y < UL else np.nan) # Change outliers to NA

# Given a pandas series x, replace any NA with median of non-NA values
def replace_na_with_median(x):
    return x.fillna(np.nanmedian(x))

# Given a pandas series x, for columns with many non-unique values, ie,
# columns that are unlikely to be nominal, return a Z-score normal version
# For likely binary variables, recode to 0-1 from 1-2. For the binned
# variable(s) with 0-4 bins, do nothing.
def replace_numeric_with_znorm(x):
    if len(x.unique()) == 2 and np.max(x) == 2:
        return x.map({1:0,2:1})
    if len(x.unique()) < 5:
        return x
    return (x-np.mean(x))/np.std(x)

# Bin data x on bins b and return binned data
def bin(x, b): # x = data array, b = boundaries array
    nb = len(b)
    N = len(x)
    y = np.zeros(N, int) # empty integer array to store the bin numbers (output)
    
    for i in range(1, nb): # repeat for each pair of bin boundaries
        y[(x >= bounds[i-1])&(x < bounds[i])] = i
    
    y[x == bounds[-1]] = nb - 1 # ensure that the borderline cases are also binned appropriately
    return y
############# 

#########################
# Import the data and load it into a pandas dataframe with named columns

#URL of  data
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/hepatitis/hepatitis.data"

# Read data contained at URL into pandas dataframe object
hep_df = pd.read_csv(url, header=None)

# Assign list of column names from the published dataset attributes
hep_df.columns = ["Class","Age","Sex","Steroid","Antivirals","Fatigue",
                  "Malaise","Anorexia","LiverBig","LiverFirm","SpleenPalpable",
                  "Spiders","Ascites","Varices","Bilirubin","AlkPhosphate",
                  "SGOT","Albumin","Protime","Histology"]

#########################
# Clean the data, which is entirely numeric

# Replace all '?' entries with NaN & coerce dataframe to float
cleaned_hep_df = hep_df.replace('?',np.nan).astype(float)

# Replace all outliers with NA
cleaned_hep_df = cleaned_hep_df.apply(replace_outliers_with_na)

# Replace the outliers and NA values with median of the good values
cleaned_hep_df = cleaned_hep_df.apply(replace_na_with_median)

# Bin the Bilirubin variable, which is inherently ordered (ordinal)
# In the sense that higher values indicate worse liver function
NB = 4
x = cleaned_hep_df["Bilirubin"]
bounds = [np.min(x), 1.1, 3, np.max(x)]# Bins corresponding to medical research
cleaned_hep_df["Bilirubin-Binned"] = bin(cleaned_hep_df["Bilirubin"],bounds)
cleaned_hep_df.head()


# For all columns that do not appear binary, replace with z-score
# For the binary variables coded 1-2, recode 0-1
norm_hep_df = cleaned_hep_df.apply(replace_numeric_with_znorm)

# The category column Sex is decoded
norm_hep_df.loc[norm_hep_df.loc[:, "Sex"] == 0, "Sex"] = "Male"
norm_hep_df.loc[norm_hep_df.loc[:, "Sex"] == 1, "Sex"] = "Female"

# Create 1 new column, for Male where 0 means Female
norm_hep_df.loc[:, "Male"] = (norm_hep_df.loc[:, "Sex"] == "Male").astype(int)

# Remove obsolete columns
norm_hep_df = norm_hep_df.drop("Sex", axis=1)
norm_hep_df = norm_hep_df.drop("Bilirubin", axis=1)



#################################################
"""
FEATURE REDUCTION,SELECTION on TRAINING dataset

Preparing for classification model development and testing. 
The dataset has 19 non-target columns and only 155 sample points, which 
presents a very real risk of over-fitting the model on the testing dataset.
This would severely compromise its abilty to accurately predict unseen
data. 

It is very important that the training dataset be utterly ignorant of the 
testing dataset. For this reason, the dataset should be split into its
test and train segments BEFORE the feature reduction. In a general sense, 
it is vitally important to train a classifier on a different subset of 
data than is used to TEST the classifer's performance. Otherwise you cannot
possibly determine the skill and accuracy of the model, because those metrics
are defined only in relation to UNSEEN data. If you test the model on the 
same data you trained it on, your tests are MEANINGLESS. 

In reducing the number of features, we wish to preserve the variance
in the dataset which retaining predictive capacity. For this project, 
I used pandas's .corr function in the interpreter to calculate all the
pairwise correlations between the variables. The highest pairwise
correlation was merely 0.36 (R^2) which was not high enough to justify
combining the two variables. 

I also looked at the univariate R^2 with the 
target variable -- these R^2 ranged from 0.0004 (useless) to 0.22. Seven
variables had R^2 > 0.10. I tried to use 7 as the k value in
SelectKBest:  the resulting columns included Fatigue, which had
R^2 = 0.09 on the target variable, and excluded Malaise, which had 
R^2 = 0.12 on the target variable. Malaise and Fatigue are the pair of 
variables most closely correlated (R^2 = 0.36). k = 8 includes both, which
may attribute duplicative variance in some cases. I had to choose, 7 or 8?

I ran the entire algorithm with k = 8 and k = 7, and compared the two sets
of results. Logistic Regression and Naive Bayes were not sensitive to the 
choice of 7 or 8, however, Decision Tree was more accurate with 
k = 7 (AUC = 0.77) than k = 8 (AUC = 0.73). 

Thus I determined that k = 7 would preserve the predictive ability of the 
classifiers, and IMPROVE the ability of one of the classifiers, while also
erring on the side of caution regarding overfitting. This is why
I used k = 7 in SelectKBest. 

The pairwise correlations, univariate correlations on the target variable,
and the metrics for the 3 classifers are included in hep_correlations.xlsx
"""

""" Setting up the data for classification problem """

# First, perform the data splitting. The target is the surival, "Class"
# The features are everything else, which will be reduced in the next step
features = norm_hep_df.drop("Class", axis=1) # all variables except the target
target = np.array(norm_hep_df["Class"]) # Live, or die?

r = 0.2 # ratio of test data over all data (this can be changed to any number between 0.0 and 1.0 (not inclusive)
# X is the training data, XX is the test data, Y is the training-target
# and YY is the testing-target
X, XX, Y, YY = train_test_split(features, target, random_state=0, test_size=r)

# Create a deep copy of X and Y to do the feature reduction
features = deepcopy(X)
target = deepcopy(Y)
k = 7 # number of features, see commentary above for rationale

# Determine the numpy numerical values of the n best predictors
temp_features = SelectKBest(f_classif, k).fit_transform(features, target)

# Get the names of the n best predictors and stash in new_features
mask = SelectKBest(f_classif, k).fit(features,target).get_support()
feature_names = list(features.columns.values)
new_feature_names = [] # The list of your K best features

for bool, feature in zip(mask, feature_names):
    if bool:
        new_feature_names.append(feature)

# Create dataframe with the numerical values and their column names
new_features = pd.DataFrame(temp_features, columns = new_feature_names)

# Subset the TEST dataset to include the columns selected during the feature
# reduction process. 
XX_new = XX[new_feature_names]
XX = deepcopy(XX)
################################################

""" 
CLASSIFICATION MODELS 

Train three different classification models to predict whether a patient
in the dataset survives. The target variable = 1 for survival, 
0 for mortality. The three models, in order, are:
1. Logistic Regression
2. Naive Bayes
3. Decision Tree

These CLASSIFIERS were chosen because:
    (reference: Python Datascience Handbook, Jake Vanderplas )
    1. Logistic Regression is inherently well suited to binary classification
    problems such as this one. These models are very interpretable due to the
    approchability of regression algorithms in general to the student of 
    statistics. 
    2. Naive Bayes are extremely fast and simple classification models
    that are often suitable for high-dimensional datasets, such as this one.
    It is exremely useful as a quick-and-dirty baseline, and in this case it
    happened to outperform the other two.
    3. Decision Trees perform binary splits in the data, which make them both
    extremely efficient and well suited to binary classification problems, 
    such as this one.

Each performed well with the default parameters following a moderate amount
of parameter sensitivity testing. The defaults are hence used for this code. 

For each model, fit the model on a set of training data which is 80%
of the total data, then test it on the remaining 20%. This is a common 
apportionment of testing and training data. 

Print the counts of true and false positives and negatives from the 
confusion matrix as these are important attributes of a model's performance.
The confusion matrix is relevant because its entries contain the counts of 
the number of true positives and true negatives -- that is, the number of
data for which the classifier predicted "yes" or "no" correctly -- and the 
counts of the number of false positives and false negatives -- that is, the 
number of data for which the classifier predicted "yes" or "no" erroneously. 
A classifier's skill can be measured with several different metrics, but 
no matter what, you need to be mindful of how often it is wrong, just not
how often it is right. We want to minimize false positives and false negatives
which can be very detrimental in real life. In this dataset, we would draw
false conclusions from a classifier that too often said a patient would live,
or die, with given symptoms if that was not true.

EVALUATION METRICS: 
Calculate the accuracy score, recall score, precision score, and F1 score:
Accuracy = 1 - Error Rate = (TP+TN)/(TP+TN+FP+FN)
Precision = TP/(TP+FP)
Sensitivity/Recall = TP/(TP+FN)
F1 score = harmonic mean of Precision rate and recall rate
The count of true positives, true negatives, false positives, and false negatives
And, the AUC, which is described next. 

Finally, plot the Return Operator Curve (ROC) which is the True Positive Rate
vs the False Positive Rate at different decision boundaries (thresholds). 
The ROC is an intuitive, graphical measure of the sensitivity for a binary
classifier. Since the target variable in this dataset is binary, the ROC
is a useful figure to examine. The ROCs for each classifier is plotted on 
the same figure for easy comparison. A baseline 1-1 line is also plotted
as a dashed line -- a model has value if it does better than the baseline
which represents a random choice classifer. ROC curves which crest quickly, 
towards the (0,1) northwest corner of the figure, represent the highest skilled
classifiers.

The area under the ROC curve is a numerical measure of skill. This quantity
is called the AUC, area under curve, and is calculated as the integral of 
the ROC from 0..1 inclusive. 

FINAL CONCLUSIONS: 
    
Of the three classifers considered for this problem, the Naive Bayes was the 
classifier with the highest AUC (0.82), Accuracy Score, Precision Score, and 
F1 score. It had the lowest number of false negatives and false positives. 
Naive Bayes had slightly fewer True Positives and 
a lower recall score than Logistic Regression. Decision Tree was the worst
on every single metric. 

The pairwise correlations, univariate correlations on the target variable,
and the metrics for the 3 classifers are included in hep_correlations.xlsx
"""

#####################
# Logistic regression classifier
print ('\n\n\nLogistic regression classifier\n')
#####################

#Training the Model
clf = LogisticRegression()
clf.fit(X, Y) 

# Apply the Model

# make class predictions for the testing set
y_pred_class = clf.predict(XX)
print ('predictions for test set:')
print (clf.predict(XX))
print ('actual class values:')
print (YY)

# IMPORTANT: first argument is true values, second argument is predicted values
# this produces a 2x2 numpy array (matrix)
CM_log = confusion_matrix(YY, y_pred_class)

# View false positives and negatives
tn, fp, fn, tp = CM_log.ravel()
print("TN: ",tn)
print("TP: ",tp)
print("FN: ",fn)
print("FP: ",fp)
# calculate accuracy
print("Accuracy Score: ", accuracy_score(YY, y_pred_class))
#recall 
print("Recall Score: ", recall_score(YY, y_pred_class))
#precision
print("Precision Score: ",precision_score(YY, y_pred_class))
#f1_score
print("F1 Score: ", f1_score(YY, y_pred_class))
#####################

"""
CREATING ROC CURVES
# IMPORTANT: first argument is true values, second argument is predicted probabilities

# we pass y_test (YY) and y_pred_prob
# we do not use y_pred_class, because it will give incorrect results without generating an error
# roc_curve returns 3 objects fpr, tpr, thresholds
# fpr: false positive rate
# tpr: true positive rate

# store the predicted probabilities for class 1

WE ARE TRYING TO PREDICT THE PEOPLE WHO LIVE, NOT THE ONES WHO DIE
SO WE USE THE INDEX-1 VALUE OF THE PREDICTION PROBABILITIES
"""
y_pred_prob = clf.predict(XX)

# CALCULATE ROC CURVE FROM PREDICTIONS AND PROBABILITIES OF THOSE PREDICTIONS
fpr, tpr, thresholds = roc_curve(YY, y_pred_prob)
# Plot the data -- plotting options will be set on last figure
plt.plot(fpr, tpr, label='Logistic Regression')

"""
#AUC Curve
# IMPORTANT: first argument is true values, 
second argument is predicted probabilities
"""
print("AUC Score: ", roc_auc_score(YY, y_pred_prob))

##################################################

# Naive Bayes classifier
print ('\n\nNaive Bayes classifier\n')
nbc = GaussianNB() # default parameters are fine
nbc.fit(X, Y)
print ("predictions for test set:")
print (nbc.predict(XX))
print ('actual class values:')
print (YY)
####################

# Apply the Model

# make class predictions for the testing set
y_pred_class = nbc.predict(XX)

# IMPORTANT: first argument is true values, second argument is predicted values
# this produces a 2x2 numpy array (matrix)
CM_log = confusion_matrix(YY, y_pred_class)

# View false positives and negatives
tn, fp, fn, tp = CM_log.ravel()
print("TN: ",tn)
print("TP: ",tp)
print("FN: ",fn)
print("FP: ",fp)
# calculate accuracy
print("Accuracy Score: ", accuracy_score(YY, y_pred_class))
#recall 
print("Recall Score: ", recall_score(YY, y_pred_class))
#precision
print("Precision Score: ",precision_score(YY, y_pred_class))
#f1_score
print("F1 Score: ", f1_score(YY, y_pred_class))
#####################

"""
CREATING ROC CURVES
See comments on first ROC calculation for additional detail
"""
y_pred_prob = nbc.predict(XX)

# CALCULATE ROC CURVE FROM PREDICTIONS AND PROBABILITIES OF THOSE PREDICTIONS
fpr, tpr, thresholds = roc_curve(YY, y_pred_prob)
# Plot the data -- plotting options will be set on last figure
plt.plot(fpr, tpr, label = "Naive Bayes")

"""
#AUC Curve
# IMPORTANT: first argument is true values, 
second argument is predicted probabilities
"""
print("AUC Score: ", roc_auc_score(YY, y_pred_prob))

##################################################

# Decision Tree classifier
print ('\n\nDecision Tree classifier\n')
dec = DecisionTreeClassifier() # default parameters are fine
dec.fit(X, Y)
print ("predictions for test set:")
print (dec.predict(XX))
print ('actual class values:')
print (YY)
####################

# Apply the Model

# make class predictions for the testing set
y_pred_class = dec.predict(XX)

# IMPORTANT: first argument is true values, second argument is predicted values
# this produces a 2x2 numpy array (matrix)
CM_log = confusion_matrix(YY, y_pred_class)

# View false positives and negatives
tn, fp, fn, tp = CM_log.ravel()
print("TN: ",tn)
print("TP: ",tp)
print("FN: ",fn)
print("FP: ",fp)
# calculate accuracy
print("Accuracy Score: ", accuracy_score(YY, y_pred_class))
#recall 
print("Recall Score: ", recall_score(YY, y_pred_class))
#precision
print("Precision Score: ",precision_score(YY, y_pred_class))
#f1_score
print("F1 Score: ", f1_score(YY, y_pred_class))
#####################

"""
CREATING ROC CURVES
See comments on first ROC calculation for additional detail
"""
y_pred_prob = dec.predict(XX)

# CALCULATE ROC CURVE FROM PREDICTIONS AND PROBABILITIES OF THOSE PREDICTIONS
fpr, tpr, thresholds = roc_curve(YY, y_pred_prob)

plt.plot(fpr, tpr, label='Decision Tree')
"""
#AUC Curve
# IMPORTANT: first argument is true values, 
second argument is predicted probabilities
"""
print("AUC Score: ", roc_auc_score(YY, y_pred_prob))

"""
Specify the figure properties shared by all ROC plots. 
This plot has a legend in the interior SE corner, axes from 0-1,
x and y axis labels, a title, and a grid. A reference 1-1 line is 
also included to show the ROC for a random classifier. A model whose 
ROC curve fell on or below the random-classifier 1-1 line would 
be useless. 
"""

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.rcParams['font.size'] = 12
plt.title('ROC curve for hepatitis survival classifier')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.grid(True)

# reference line for random classifier: 1-1 line
plt.plot([0, 1], [0, 1], color='navy', linestyle='--', label="Random Classifier") 
plt.legend()



