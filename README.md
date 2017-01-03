# Building a Student Intervention System

#### Project Description

 - There is a strong push to help increase the likelihood of students being successful in the classroom without watering down the education or engaging in behaviors that raise the likelihood of passing without improving the actual underlying learning. Graduation rates are often the criteria of choice for this; educators and administrators are after new ways to predict success and failure early enough to stage effective interventions, as well as to identify the effectiveness of different interventions. The objective of this project is to select and develop a model that will predict the likelihood that a given student will pass, thus helping diagnose whether or not an intervention is necessary. The model is developed based on a subset of data that consists of information from randomly sample students, and it will be tested against a subset of the data that is kept hidden from the learning algorithm, in order to test the model’s effectiveness on data outside the training set. 

    - The student datasheet can be found here: [`student-data.csv`](https://github.com/Tsmith5151/Student-Intervention/blob/master/student-data.csv) (Information regarding the attributes is listed in the Appendix located at the bottom on this page)
    - The code for this project can be found here: [`student_intervention.ipynb`](https://github.com/Tsmith5151/Student-Intervention/blob/master/student_intervention.ipynb)
    - Additional analysis was added to: [`student_intervention_add_on.ipynb`](https://github.com/Tsmith5151/Student-Intervention/blob/master/student_intervention_add_on.ipynb). Receiving Operating Characteristic [`ROC`](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html) plot was created for the SVM model to measure the performance of the learning algorithm on the testing set by computing the area under the curve [`AUC`](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.auc.html). The ROC plot is a visual way of inspecting the performance of a binary classifier (pass/fail). This method compares the rate at which the classifier is making a false alarm and making a correct prediction. Also included in the code is a [`Confusion_Matrix`](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html), which allows the accuracy of the model to be calculated. Generating a heatmap using [`seaborn`](https://stanford.edu/~mwaskom/software/seaborn/generated/seaborn.heatmap.html) shows the number of occurances the classifier makes a true positive, false negative, false positive, and true negative. (Updated 3/14/2016)
    
#### Software and Libraries

  - Python 3.5 (Anaconda)
  - Sklearn 0.17
  - Numpy 1.10
  - iPython/Jupyter Notebook

``` python
import numpy as np
import pandas as pd
import csv
import time
from sklearn import cross_validation
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import train_test_split, KFold
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import f1_score

# First, decide how many training vs test samples you want
from sklearn.cross_validation import train_test_split

import pylab as pl
import matplotlib.pyplot as pl
from sklearn.preprocessing import scale
```
### Classification vs Regression

#### Question 1: The goal is to identify students who might need early intervention - which type of supervised machine learning problem is this, classification or regression? Why?

  -The goal of the project is to identify which students would require early intervention from the teacher/administration/parents by assigning a class to the students as either being successful and "passing" the course or "failing". Being that we are not predicting a continuous output, this would not be an example of regression, rather this type of supervised machine learning problem is `classification`.

### Explore The Data
- The dataset used in this project is included as `student-data.csv`. It has the following attributes for each student:
```python
student_data = pd.read_csv("student-data.csv")
```
#### Question 2:
- Compute desired values with an appropriate expression/function call for the following 
  - Total number of students
  - Number of students who passed
  - Number of students who failed
  - Number of features
  - Graduation rate of the class

```python
n_students = student_data.shape[0]
n_features = student_data.shape[1] 
n_passed =  student_data[student_data['passed'] =='yes'].shape[0]
n_failed =  student_data[student_data['passed'] == 'no'].shape[0]
grad_rate = (n_passed*1.0) / (n_students*1.0) * 100
print "Total number of students: {}".format(n_students)
print "Number of students who passed: {}".format(n_passed)
print "Number of students who failed: {}".format(n_failed)
print "Number of features: {}".format(n_features)
print "Graduation rate of the class: {:.2f}%".format(float(grad_rate))
```

| Parameters                       | Value    |
| -------------------------------  | ---------|
| Total number of students         | 395      |
| Number of students "passed"      | 265      |
| Number of students "failed"      | 130      |
| Number of features               | 31       |
| Graduation rate of the class (%) | 67.09    |

#### Preparing the Data

- It is often the case that the data you obtain contains non-numeric features. This can be a problem, as most machine learning algorithms expect numeric data to perform computations with. Lets first separate our data into feature and target columns, and see if any features are non-numeric. Note: For this dataset, the last column ('passed') is the target or label we are trying to predict.

```python 
# Extract feature (X) and target (y) columns
feature_cols = list(student_data.columns[:-1])  # all columns but last are features
target_col = student_data.columns[-1]  # last column is the target/label
print "Feature column(s):\n{}".format(feature_cols)
print "Target column: {}".format(target_col)

X_all = student_data[feature_cols]  # feature values for all students
y_all = student_data[target_col]  # corresponding targets/labels
print "\nFeature values:"
print X_all.head()  # print the first 5 rows
```

#### Preprocess feature columns
- There are several non-numeric columns that need to be converted! Many of them are simply `yes` or `no`, e.g. internet. These can be reasonably converted into 1/0 (binary) values. Other columns, like `Mjob` and `Fjob`, have more than two values, and are known as categorical variables. The recommended way to handle such a column is to create as many columns as possible values `(e.g. Fjob_teacher, Fjob_other, Fjob_services, etc.)`, and assign a `1` to one of them and `0` to all others. These generated columns are sometimes called dummy variables, and we will use the `pandas.get_dummies()` function to perform this transformation.

```python
def preprocess_features(X):
    outX = pd.DataFrame(index=X.index)  # output dataframe, initially empty

    # Check each column
    for col, col_data in X.iteritems():
        # If data type is non-numeric, try to replace all yes/no values with 1/0
        if col_data.dtype == object:
            col_data = col_data.replace(['yes', 'no'], [1, 0])
        # Note: This should change the data type for yes/no columns to int

        # If still non-numeric, convert to one or more dummy variables
        if col_data.dtype == object:
            # Splits Columns Up if non-numeric into 1 or 0; 
            col_data = pd.get_dummies(col_data, prefix=col)  
            
        outX = outX.join(col_data)  # collect column(s) in output dataframe

    return outX

X_all = preprocess_features(X_all)
print "Processed feature columns ({}):-\n{}".format(len(X_all.columns), list(X_all.columns))

#Format Target yes/no values with 1/0
y = pd.DataFrame(y_all)
y = y.replace(['yes', 'no'], [1, 0])
#in the form (X, 1), but the method expects a 1d array and has to be in the form (X, )
y_all = np.ravel(y)

#join dataset
student_data = pd.concat([X_all, y], axis = 1)
print(student_data)
```
#### Split data into training and test sets
- So far, all categorical features are turned into numeric values. Now the data can be split (both features and labels) into training and test sets. Using [`StratifiedShuffleSplit()`](http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html), the features for the 395 sampled students will be randomly stratified and split; The training size is `300` and testing size  is `40`. 

- As the data shows, more students passed than fail, meaning an unbalanced dataset. This is important to address when splitting the data into training and testing. Stratified Shuffle Split helps deal with the imbalance number of labels in the dataset by allowing the training and testing set to have roughly the same ratio of passed vs failed students; the data is also shuffled to remove any bias in the order of the students. StratifiedShuffleSplit will randomly select training and test sets multiple times and average the results over all the tests.

```python
# First, decide how many training vs test samples you want
num_all = student_data.shape[0]  # same as len(student_data)
num_train = 300  # about 75% of the data
num_test = num_all - num_train

y = student_data['passed']
#print y

def Stratified_Shuffle_Split(X,y,num_train):
    sss = StratifiedShuffleSplit(y, 3, train_size=num_train, random_state = 0)
    for train_index, test_index in sss:
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    return X_train, X_test, y_train, y_test
    
X_train, X_test, y_train, y_test = Stratified_Shuffle_Split(X_all, y, num_train)
print "Training Set: {0:.2f} Samples".format(X_train.shape[0])
print "Testing Set: {0:.2f} Samples".format(X_test.shape[0])
```
### Training and Evaluating Models

Choose 3 supervised learning models that are available in scikit-learn, and appropriate for this problem. 

- The three supervised learning models selected for the project are:
  - [`Naive Bayes`](http://scikit-learn.org/stable/modules/naive_bayes.html)
  - [`RandomForestClassifier`](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
  - [`SVM)`](http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)

#### Question 3: What is the theoretical O(n) time & space complexity in terms of input size?

 - `Naive Bayes`
  - Space complexity: O(dc); d = number of dimensions (attributes) ; c = number of classes
  - Training time: O(nd +cd); n = number of instances 
 
 - `Random Forest Classifier`
  - Space Complexity: O(M(mn*log(n)); m = number attributes, M = number of trees
  - Training time: O(n*logn); n = number of elements 
 
- `Support Vector Machine`
 - Space Complexity: O(n^2); n = training size 
 - Training time: O(n^3); n = training size (quadratic)
 
#### Question 4: What are the general applications of this model? What are its strengths and weaknesses?

- `Naive Bayes`
 - Naive Bayes (NB) performs wells in many different domains and is simple to implement. The computational complexity of NB is lower than other methods such as decision trees and therefore is quite fast to run and does not require a lot of CPU memory. Additionally, the classifier is not sensitive to irrelevant features and is robust to noisy and missing data. As it is often stated, Naive Bayes is not so naive as the algorithm assumes independence between every pair of features, which may not hold for all the attributes and can be too constraining. Furthermore, the performance of the model does not significantly improve by adding more data to increase the sample size. 

- `Random Forest Classifier` 
 - The runtimes are relatively fast and is able to handle unbalanced data such as the case from the `student-data.csv`, where only 67% of students pass. The algorithm can be used with large or small datasets and with a larger number of attributes. Compared to the decision tree classifier, Random Forest almost always has a lower classification error and better f-scores. One of the weaknesses is when Random Forest is used for regression as it cannot predict beyond the range in the training data and thus can cause over-fitting especially if the dataset is noisy. 

- `Support Vector Classifier`
 - Support Vector Machines (SVMs) are a set of supervised learning methods that is used for classification, regression, and outliers detection. Being a classification problem, Support Vector Classifiers (SVC) is very effective in high dimensional spaces and when the number of dimensions is greater than the number of samples. This algorithm detects nonlinearity in the data automatically, therefore you do not have to apply complex transformations to the variables. Also SVM's uses a subset of training points in the decision function (called support vectors), so it is also memory efficient. The disadvantages of SVC is when the number of features is much greater than the number of elements, the method is likely to give poor a performance score. Also, SVMs do not directly provide probability estimates, these are calculated using an expensive five-fold cross-validation. 
 
#### Question 5: Given what you know about the data so far, why did you choose this model to apply?

The following details highlight the functionality of each model and why it could be applicable for this particular problem: 

- `Naive Bayes` is a classification algorithm which is based on Bayes' rule and the idea of probability conditioned by evidence. Being the goal of learning probability of Y (target) given X (feature(s)) `P(Y|X)` where X = ⟨X1 ...,Xn⟩, NB makes the assumption that each feature is conditionally independent from the other given Y. NB takes all of the evidence available (attributes) in order to modify the prior probability of the prediction. The training data is utilized to learn estimates of P(X|Y) and P(Y) in order to learn some target P(Y|X). Naive Bayes (Gaussian) was chosen primarily for it's quick computational run time, however one drawback when using NB is the model may be too simplistic and under-fit the data given smaller datasets.

- `Random Forest Classifier` uses a large number of decision tree models from bootstrapped aggregated datasets which improves the predictions by reducing the variances and bias of the estimates, especially for unbalanced data. The algorithm is similar to that of [`BaggingClassifier`](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingClassifier.html), however the `Random Forest` provides improvement to the model by de-correlating the decision trees. The algorithm first begins boostrapping the dataset of observations and generating a large number of trees, with each one being different. The model entails that each time a tree is split, a random selection of `m` observations per feature is chosen as split candidates from the full dataset. The number of observations considered at each split is roughly equal to the `square_root` of the total number of observations. For instance there is 395 samples is the `student_data.csv` file, thus roughly 20 observations will be randomly selected per feature. As a note, when predicting the target, the terminal node of the decision tree is the most commonly occurring class in the corresponding region, unlike regression which the average of the regression estimates. 

- `Support Vector Machine` is primarily a classier method that performs classification tasks by constructing hyperplanes in a multidimensional space that separates cases of different class labels. Given a labeled training data set, the algorithm outputs an optimal hyperplane, which categorizes new examples. The objective of the SVM algorithm is to maximize the margin around the separating hyperplane (higher dimensions). In other words, we want a classifier with as big of a margin as possible when separating the data. Hence, the optimal separating hyperplane maximizes the margin of the training data. Based on the larger number of features in the student dataset, a linear relationship is not likely and thus SVM's would be suitable for this dataset for constructing decision boundaries.

- As mentioned, the data is unbalanced and that is why it is required to use precision and recall (or F1, harmonic mean of them) instead of accuracy as our evaluation metric. The "true positives" are the intersection of the two lists, false positives are predicted items that aren't real, and false negatives are real items that aren't predicted. F1 score is calculated by `2(precision * recall) / (precision + recall)`. Where the precision is out of those selected for intervention, how many really need intervention, and recall is out of those who need intervention, how many of them were identified?). The following table shows the results for all three models: (only 1 iteration):

| Model                |      Train -F1      |      Test - F1     |  Training Time  | Prediction Time |
| ---------------------| --------------------|--------------------|-----------------|-----------------|
| Random Forest        |    0.989            |      0.723         |     0.002       |      0.001      |          
| Naive Bayes          |    0.810            |      0.742         |     0.001       |      0.001      |         
| SVC                  |    0.870            |      0.805         |     0.008       |      0.002      |    

#### Question 6: Fit this model to the training data, try to predict labels (for both training and test sets), and measure the F1 score. Repeat this process with different training set sizes (100, 200, 300).

- The objective of running the models of various training sizes (100, 200, and 300) is to see how much each model's F1 score improves as the computation time increases. The training time, prediction time, F1 score on training set and F1 score on test set, for each respected training size is shown in the table below. As a note, the performance metric for this project is the [`F1 score`](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html), which is just the weighted average between precision and recall, and thus can allow you to understand how well your model performs on a testing set. Furthermore, to address any biased sampling, the model was run multiple times with `StratifiedShuffleSplit()` applied to each iteration. As the size of the training sizes increases, so does the computational time; since our dataset is relatively small, all three models run substantially quick. The results are based on the average of `10` iterations and the model which generalized the best to the unseen test data is `SVC`, F1 = 0.810 (training size = 300).  The second best model would be Random Forest (0.782), however the performance of the model could be marginally improved by increasing the number of trees to n_estimators = 100, but would require a slightly longer training/prediction time. 

> F1 Training Score:

| Model                       | F1  (Size = 100)  | F1 (size = 200) | F1 (size = 300) |
| ----------------------------| ------------------|-----------------|-----------------|
| Random Forest               |       0.991       |     0.994       |     0.990       |
| Naive Bayes                 |       0.579       |     0.793       |     0.785       |
| SVC                         |       0.867       |     0.848       |     0.871       |

> F1 Testing Score:

| Model                       | F1  (Size = 100)  | F1 (size = 200) | F1 (size = 300) |
| ----------------------------| ------------------|-----------------|-----------------|
| Random Forest               |       0.775       |     0.752       |      0.732      |
| Naive Bayes                 |       0.466       |     0.707       |      0.764      |
| SVC                         |       0.792       |     0.801       |      0.810      |

> Training Time

| Model                       | Time  (Size = 100)| Time (size = 200) | Time (size = 300) |
| ----------------------------| ------------------|-------------------|-------------------|
| Random Forest               |       0.092       |      0.126        |     0.134         |
| Naive Bayes                 |       0.003       |      0.003        |     0.008         |
| SVC                         |       0.003       |      0.015        |     0.049         |

> Prediction Time:

| Model                       | Time  (Size = 100)| Time (size = 200) | Time (size = 300) |
| ----------------------------| ------------------|-------------------|-------------------|
| Random Forest               |       0.031       |      0.006        |      0.005        |
| Naive Bayes                 |       0.001       |      0.001        |      0.002        |
| SVC                         |       0.001       |      0.002        |      0.005        |

##### Predict on Training Set and Compute F1 Score

```python
#Train Model
def train_classifier(clf, X_train, y_train):
    print "Training {}:".format(clf.__class__.__name__)
    start = time.time()
    clf.fit(X_train, y_train)
    end = time.time()
    train_clf_time = end - start
    print "Training Time (secs): {:.3f}".format(train_clf_time)
    return train_clf_time
```
```python
# Predict on Training Set and Compute F1 Score
def predict_labels(clf, features, target):
    print "Predicting labels using {}:".format(clf.__class__.__name__)
    start = time.time()
    y_pred = clf.predict(features)
    end = time.time()
    prediction_time = end - start
    print "Prediction Time (secs): {:.3f}".format(prediction_time)
    return (f1_score(target.values, y_pred, pos_label='yes'), prediction_time)
```

##### Naive Bayes:

```python
clf_NB = GaussianNB()

# Fit model to training data
train_classifier(clf_NB, X_train, y_train)

# Predict on training set and compute F1 score
train_f1_score = predict_labels(clf_NB, X_train, y_train)
print "F1 score for training set: {}".format(train_f1_score[0])

#Predict on Test Data
print "**********************************************************"
print "Testing Data Size:",len(X_test)
print "F1 score for test set:",(predict_labels(clf_NB, X_test, y_test))
```

##### Random Forest Classifier:
 
```python
clf_RF = RandomForestClassifier(n_estimators=10)

# Fit model to training data
train_classifier(clf_rf, X_train, y_train)

# Predict on training set and compute F1 score
train_f1_score = predict_labels(clf_rf, X_train, y_train)
print "F1 score for training set: {}".format(train_f1_score)

#Predict on Test Data:
print "**********************************************************"
print "Testing Data Size:",len(X_test)
print "F1 score for test set: {}".format(predict_labels(clf_rf, X_test, y_test))
```

##### Support Vector Classifier:
```python
clf_SVC = SVC()

# Fit model to training data
train_classifier(clf_SVC, X_train, y_train)

# Fit model to training data
train_classifier(clf_SVC, X_train, y_train)

# Predict on training set and compute F1 score
train_f1_score = predict_labels(clf_SVC, X_train, y_train)
print "F1 score for training set: {}".format(train_f1_score)

#Predict on Test Data
print "**********************************************************"
print "Testing Data Size:",len(X_test)
print "F1 score for test set: {}".format(predict_labels(clf_SVC, X_test, y_test))
```

##### Run All Models (10 Iterations)

```python
def run_all_models(classifiers, sizes):
    for clf in classifiers:
        for size in sizes:
            df = pd.DataFrame(columns = ['Training_Size',
                'Testing_Size',
                'Training_Time',
                'Prediction_Time',
                'F1_Training_Score',
                'F1_Testing_Score'])
            y = student_data['passed']
            y = student_data['passed']
            X_train, X_test, y_train, y_test = Stratified_Shuffle_Split(X_all, y, size)
        
            num_times_to_run = 10
            for x in range(0, num_times_to_run): 
                
                f1_score_train, f1_score_test, train_time, pred_time_test = train_predict(clf, X_train, y_train, X_test, y_test)
                df = df.append({'Training_Size': X_train.shape[0],
                        'Testing_Size': X_test.shape[0],
                        'Training_Time': train_time,
                        'Prediction_Time': pred_time_test,
                        'F1_Training_Score': f1_score_train,
                        'F1_Testing_Score': f1_score_test}, 
                        ignore_index= True)
            
            df_100 = df[(df.Training_Size == size)]
            df_100_mean = df_100.mean()
            print "**********************************************************"
            print "Mean Statistics:"
            print df_100_mean
            print "**********************************************************"
 ```
 
 ```python
 run_all_models([clf_RF], [100])
 run_all_models([clf_SVC], [100])
 run_all_models([clf_NB], [100])
 ```

### Choosing the Best Model

#### Question 7: Which model is generally the most appropriate based on the available data, limited resources, cost, and performance? Also explain to the board of supervisors in layman's terms how the final model chosen is supposed to work. 

- The board of supervisors wants to find the most effective model with the least amount of computation costs due to the limited resources and budget. Implementing a student intervention system using concepts of supervised machine learning which will predict the likelihood that a given student will pass, thus helping diagnose whether or not an intervention is necessary. The model that would be best for the student intervention system is Support Vector Machines (SVM), a classification algorithm that constructs hyperplanes in a high dimensional space and separates two different class labels ("pass" or "fail"). As a visual illustration in the 2D plot shown below, let the students who passed is denoted by "green" and those who failed as "red". There are many possible solutions on where to draw the decision boundary, such as the solid line on the left or to the right. Both of these lines are too close to the existing observations and if new observations are added, it would be unlikely they behave precisely like the initial data. Therefore, SVM's reduces the risk of selecting the wrong decision boundary by choosing the line that has the largest distance from the bordering data points of the two classes. Having the additional space between the groups reduces the chance of selecting the wrong class. As shown in the figure, the largest distance between the two groups of students is called the `margin`. The margin width is determined by the points that are present on the limit of the margin, known as the `support vectors`. The dashed line in the middle of the margin width would be where the decision boundary would be.  

![alt tag](https://udacity-github-sync-content.s3.amazonaws.com/_imgs/372/1457702183/SVM_2.png)

- Not all data behaves linearly and therefore a linear separating hyperplane will not work. A special technique can be applied so that the non-linear datasets can be seperated by transforming the input variables so that the shape of the dataset becomes more linear. This is known as the `kernel trick`. A kernel is essentially a mapping function in which we take the input variables and transform into a much higher dimensional space. After separating the data using complex separating plane known as hyperplanes (shown below on the left), you take the solution and return back to the original space, and now you have a non-linear separation.   
 SVM's can use a complex separating plane known as hyperplanes (shown in the figure below).

![alt tag](https://udacity-github-sync-content.s3.amazonaws.com/_imgs/372/1457702176/data_2d_to_3d_hyperplane.png)

- Several parameters can be adjusted to the SVM learning algorithm. First, `C` controls the tradeoff between a smooth decision boundary and classifying training points correctly. The higher the `C` value, the more intricate the decision boundary becomes. Second, `Gamma` defines how far the influence of a single training example reaches; low values meaning ‘far’ & high values meaning ‘close’. If the value of gamma is too small then the model is too constrained and cannot capture the complexity or “shape” of the data. 

- As previously shown, the Support Vector Classifier's performance score (0.810) on the test dataset is marginally higher than Random Forest and Naive Bayes; but the tradeoff for having a better F1 score results in the computational time to train the data to take nearly 20% longer compared to Naive Bayes. Overall, the Support Vector Classifier would be a sufficient model of choice from the standpoint F1 score, prediction/training time, and the ability to capture non-linear trends. SVM's performs the better when applied to binary classification problems such as this scenario, however one of the drawbacks when using SVM's is interrupting the results and may not be as clear as other algorithms, such as Decision Trees. 

#### Question 10: Fine-tune the model. Use Gridsearch with at least one important parameter tuned and with at least 3 settings. 
 
 - The three parameters fine turned using [`GridSearch`](http://scikit-learn.org/stable/modules/generated/sklearn.grid_search.GridSearchCV.html#sklearn.grid_search.GridSearchCV) for the `Support Vector Classifier` model are `C`, `gamma`, and `kernel`. 
 
 - `C`: [1,10,50,100,200,300,400,500,1000]
 - `Gamma`: [1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1]
 - `Kernel`: rbf (useful when the data-points are not linearly separable; faster than the other kernels and can approximate almost any nonlinear function)
 
```python
def reformat(col_data):
    return col_data.replace(['yes', 'no'], [1, 0])

def iterate_fit_predict(number_runs):
        f1_scores = []
        gamma = []
        C = []

        # Get the features and labels from the Boston housing data
        y = reformat(student_data['passed'])
        
        for num in range(0,number_runs):
            X_train, X_test, y_train, y_test = Stratified_Shuffle_Split(X_all, y, 300)
            clf_SVC = SVC()
            parameters = [{'C':[1,10,50,100,200,300,400,500,1000,],
                         'gamma':[1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1],
                         'kernel': ['rbf']}]
    
            
            clf = GridSearchCV(clf_SVC, parameters, scoring = 'f1')
            
            # Fit the learner to the training data to obtain the best parameter set
            clf.fit(X_train, y_train)
            f1_scores.append(clf.score(X_test, y_test))
            gamma.append(clf.best_params_['gamma'])
            C.append(clf.best_params_['C'])
            clf = clf.best_estimator_
            #print clf
        
        df_f1 = pd.Series(f1_scores)
        df_gamma = pd.Series(gamma)
        df_C = pd.Series(C)
        
        print clf
        print "\nF1 Scores:"
        print df_f1
        print "\nAverage F1 Test Scores:"
        print df_f1.mean()
        print "\nC:"
        print df_C
        print "\nGamma:"
        print df_gamma
 ```
#### Question 12:  What is the model's final F1 score?

- Multiple iterations are desired to be performed in order to provide a conclusive F1 score (performance of model). After running `10` iterations, the average F1 score of the tuned SVC model is: 0.813.

``` python
SVC(C=200, cache_size=200, class_weight=None, coef0=0.0,
  decision_function_shape=None, degree=3, gamma=0.0001, kernel='rbf',
  max_iter=-1, probability=False, random_state=None, shrinking=True,
  tol=0.001, verbose=False)
```
| F1 Score       | 
| ---------------| 
| 0.866667       |  
| 0.781250       |    
| 0.793103       |  
| 0.812500       |  
| 0.812500       |    
| 0.813559       |  
| 0.774194       |  
| 0.754098       |    
| 0.825397       |  
| 0.805970       |

Average C:
100.2
Average Gamma:
0.03007

#### Appendix

| Feature                         | Description     | Info            |
| :-------------------------------| :-------------- |:----------------|
| school     | student's school   | binary: "GP" or "MS"              |
| sex        | student's sex | binary: "F" - female or "M" - male |
| age        | student's age | numeric: from 15 to 22 |
| address    | student's home address type | binary: "U" - urban or "R" - rural) |
| famsize    | family size | binary: "LE3" - less or equal to 3 or "GT3" - greater than 3 |
| Pstatus    | parent's cohabitation status | binary: "T" - living together or "A" - apart |
| Medu       | mother's education | numeric: 0 -none,1 - primary ed. (4th grade), 2 –5th to 9th grade, 3 – ec. ed. or 4–higher ed.) |
| Fedu       | father's education | numeric: 0 -none,1 - primary ed. (4th grade), 2 –5th to 9th grade, 3 – ec. ed. or 4–higher ed.) |
| Mjob       | mothers's job | nominal: "teacher", "health", civil "services" (e.g. admin,police), "at_home" or "other" |
| Fjob       | father's job | nominal: "teacher", "health", civil "services" (e.g. admin,police), "at_home" or "other" |
| reason     | reason to choose this school | nominal: close to "home", school "reputation", "course" pref. or "other" |
| guardian   | student's guardian | nominal: "mother", "father" or "other" |
| traveltime | home to school travel time | numeric: 1 -<15 min., 2 - 15-30 min., 3 - 30 min.-1 hour, or 4 - >1 hour |
| studytime  | weekly study time | numeric: 1 - <2 hours, 2 - 2 to 5 hours, 3 - 5 to 10 hours, or 4 - >10 hours |
| failures   | number of past class failures | numeric: n if 1<=n<3, else 4 |
| schoolsup  | extra educational support | binary: yes or no |
| famsup     | family educational support | binary: yes or no |
| paid       | extra paid classes within the course subject | Math or Portuguese, binary: yes or no |
| activities | extra-curricular activities | binary: yes or no |
| nursery    | attended nursery school | binary: yes or no |
| higher     | wants to take higher education | binary: yes or no |
| internet   | Internet access at home| binary: yes or no |
| romantic   | with a romantic relationship | binary: yes or no |
| famrel     | quality of family relationships | numeric: from 1 - very bad to 5 - excellent |
| freetime   | free time after school | numeric: from 1 - very low to 5 - very high |
| goout      | going out with friends | numeric: from 1 - very low to 5 - very high |
| Dalc       | workday alcohol consumption | numeric: from 1 - very low to 5 - very high |
| Walc       | weekend alcohol consumption | numeric: from 1 - very low to 5 - very high |
| health     | current health status | numeric: from 1 - very bad to 5 - very good |
| absences   | number of school absences | numeric: from 0 to 93 |
| passes     | did the student pass the final exam | binary: yes or no |

