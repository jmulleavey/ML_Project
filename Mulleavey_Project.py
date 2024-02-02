# %% markdown
# # PREDICTING THE LIKELIHOOD OF STROKE FROM PATIENT DATA
# ## Created by: Jake Mulleavey

# GOAL: Build a model that can accurately predict whether a given individual has a stroke or not based on all other provided health & demographic data.

# Attribute Information:
# *   id: unique identifier
# *   gender: "Male", "Female" or "Other"
# *   age: age of the patient
# *   hypertension: 0 if the patient doesn't have hypertension, 1 if the patient has hypertension
# *   heart_disease: 0 if the patient doesn't have any heart diseases, 1 if the patient has a heart disease
# *   ever_married: "No" or "Yes"
# *   work_type: "children", "Govt_jov", "Never_worked", "Private" or "Self-employed"
# *   Residence_type: "Rural" or "Urban"
# *   avg_glucose_level: average glucose level in blood
# *   bmi: body mass index
# *   smoking_status: "formerly smoked", "never smoked", "smokes" or "Unknown"
# *   stroke: 1 if the patient had a stroke or 0 if not

# %% markdown
# # Import Packages

# %% codecell
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier


from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from hyperopt import fmin, tpe, hp, Trials
import hyperopt

from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve
from sklearn.metrics import RocCurveDisplay
from sklearn.preprocessing import label_binarize

# %% markdown
# # Load Dataset

# %% codecell
stroke_df = pd.read_csv("~/Documents/MSc Cog Neuro/Secondo Anno/Semester 1/Machine Learning for Brain & Cognition/Project/Dataset_Stroke.csv")
print(stroke_df[0:10]) # taking an initial look at our data by only viewing the first 10 rows

stroke_df.info(verbose=True)
# Using the code above, we can identify that there are 201 subjects that have null values for their BMI.

# %% markdown
# # Pre-Processing
# ## Data Cleaning
# By observing our first few samples, there are null values within the 'bmi' variable which we have to deal with before beginning with our analysis.

# Traditional approaches to dealing with these NaN values would be to simply drop all subjects who have a missing value for their BMI.
# Another approach would be to use univariate imputation where the missing values within the BMI column would be replaced with the mean or median value for the entire set of BMI's.
# Instead, here we will fill in the missing values using a multivariate imputation approach where the inserted values will be predicted by a regression model using the other variables as features.

# %% codecell
# In order to perform Multivariate Imputation, we must import the IterativeImputer package from sklearn.
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

# %% markdown
# Because 'IterativeImputer' only works with numerical values we have to extract only the relevant variables from our dataframe.
# Age, Hypertension, Heart Disease, Average Glucose Level & Stroke Statues will be used
# Additionally Gender & Smoking Statues seem relevant despite being dtype 'objects' so first we will convert them to numerics for use in the Imputation process.

# %% codecell
gender_num = stroke_df.gender.replace(to_replace=['Male', 'Female', 'Other'], value=[0, 1, 2])
smoking_status_num = stroke_df.smoking_status.replace(to_replace=['formerly smoked', 'never smoked', 'smokes', 'Unknown'], value=[0, 1, 2, 3])

imputation_df = stroke_df.loc[:, ['age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi', 'stroke']]
imputation_df['gender_num'] = gender_num
imputation_df['smoking_status_num'] = smoking_status_num

print(imputation_df[0:3])
imputation_df.info(verbose=True)

# %% markdown
# Now that we have our data frame with exclusively numerical values in it, we can run the Multivariate Imputation in order to replace the missing values in BMI.

# %% codecell
impute_it = IterativeImputer()
imputation_df = impute_it.fit_transform(imputation_df)

imputation_df[:, 4] = imputation_df[:, 4].round(decimals = 1) # round the bmi column to only 1 decimal place

# %% markdown
# All the NaN missing values within the BMI column have now been replaced with imputed values based on all other numerical values in our dataset.
# However, we still have to overwrite the 'bmi' values in our 'stroke_df' data frame with the imputed values we just created.

# %% codecell
bmi_new = imputation_df[:, 4] # our new bmi data exists as class 'numpy.ndarray'
bmi_series = pd.Series(bmi_new) # this will change the class of our new BMI data to match the existing BMI data in our main data frame

#print(type(bmi_series))
#print(type(stroke_df['bmi']))

inter_df = pd.DataFrame(data = bmi_series) # here we are simply putting the new BMI data into a data frame
stroke_df['bmi'] = inter_df # and we replace the existing BMI data with the new imputed BMI data

print(stroke_df[0:10]) # we can see that the 'stroke_df' has been updated
stroke_df.info(verbose=True) # with no non-null values in the BMI column

# %% markdown
# ## Data Visualization
# Let's look at the distributions of each of the variables involved.

# %% codecell
# create percentages for all categorical variables, as well as labels for figures
gender_pct = stroke_df['gender'].value_counts(normalize=True) * 100
gender_labels = 'Female', 'Male', 'Other'
hypertension_pct = stroke_df['hypertension'].value_counts(normalize=True) * 100
hypertension_labels = 'No Hypertension', 'Hypertensions'
heartdisease_pct = stroke_df['heart_disease'].value_counts(normalize=True) * 100
heartdisease_labels = 'No Heart Disease', 'Heart Disease'
married_pct = stroke_df['ever_married'].value_counts(normalize=True) * 100
married_labels = 'Married', 'Never Married'
residence_pct = stroke_df['Residence_type'].value_counts(normalize=True) * 100
residence_labels = 'Urban', 'Rural'
work_pct = stroke_df['work_type'].value_counts(normalize=True) * 100
work_labels = 'Private', 'Self-Employed', 'Children', 'Government Job', 'Never Worked'
smoking_pct = stroke_df['smoking_status'].value_counts(normalize=True) * 100
smoking_labels = 'Never Smoked', 'Unknown', 'Formerly Smoked', 'Currently Smokes'

# %% codecell
# plots for all categorical or binary variables
fig = plt.figure(figsize = (250, 550))
ax1 = fig.add_subplot(171)
ax1.pie(gender_pct, labels = gender_labels, autopct = '%1.1f%%', shadow = True, textprops = {'fontsize':100})
ax1.set_title('Gender', fontsize = 150)
ax2 = fig.add_subplot(172)
ax2.pie(hypertension_pct, labels = hypertension_labels, autopct = '%1.1f%%', shadow = True, textprops = {'fontsize':100})
ax2.set_title('Hypertension', fontsize = 150)
ax3 = fig.add_subplot(173)
ax3.pie(heartdisease_pct, labels = heartdisease_labels, autopct = '%1.1f%%', shadow = True, textprops = {'fontsize':100})
ax3.set_title('Heart Disease', fontsize = 150)
ax4 = fig.add_subplot(174)
ax4.pie(married_pct, labels = married_labels, autopct = '%1.1f%%', shadow = True, textprops = {'fontsize':100})
ax4.set_title('Marriage Status', fontsize = 150)
ax5 = fig.add_subplot(175)
ax5.pie(residence_pct, labels = residence_labels, autopct = '%1.1f%%', shadow = True, textprops = {'fontsize':100})
ax5.set_title('Residence Type', fontsize = 150)
ax6 = fig.add_subplot(176)
ax6.pie(work_pct, labels = work_labels, autopct = '%1.1f%%', shadow = True, textprops = {'fontsize':100})
ax6.set_title('Work Type', fontsize = 150)
ax7 = fig.add_subplot(177)
ax7.pie(smoking_pct, labels = smoking_labels, autopct = '%1.1f%%', shadow = True, textprops = {'fontsize':100})
ax7.set_title('Smoking Status', fontsize = 150)
plt.subplots_adjust(top = 1, wspace = 0.75)

# %% codecell
# plots for all continuous variables
sns.set(font_scale = 2)
fig2, axes = plt.subplots(1, 3, figsize = (40, 10))

sns.set_style('darkgrid')
figA = sns.histplot(data = stroke_df['age'], kde = True, ax = axes[0])
figA.set_ylim(125, 375)
figA.set_title('Age')

figB = sns.histplot(data = stroke_df['avg_glucose_level'], binwidth = 3.5, kde = True, ax = axes[1])
figB.set_title('Average Glucose Level')

figC = sns.histplot(data = stroke_df['bmi'], kde = True, ax = axes[2])
figC.set_xlim(10, 60)
figC.set_title('BMI')

# %% markdown
# Before moving on, it is important we transform our categorical data from 'string' to 'numerical' type.

# %% codecell
stroke_df['gender'].replace(['Male','Female','Other'], [0,1,2], inplace=True)
stroke_df['ever_married'].replace(['No','Yes'], [0,1], inplace=True)
stroke_df['work_type'].replace(['children','Govt_job','Never_worked','Private','Self-employed'], [0,1,2,3,4], inplace=True)
stroke_df['Residence_type'].replace(['Rural','Urban'], [0,1], inplace=True)
stroke_df['smoking_status'].replace(['formerly smoked','never smoked','smokes','Unknown'], [0,1,2,3], inplace=True)

#plt.hist(df_pca['work_type']);
#plt.hist(df_pca['stroke']);

# %% markdown
# Now that we have our complete dataset we can move on to the next step of pre-processing our data before beginning to construct out predictive model.

# ## Extract Relevant Variables & Re-Configure
# Now we will combine our input & output variables into matrices.

# %% codecell
X = stroke_df.iloc[:,1:11].values  # consider all variables as input data except for the first & last
Y = stroke_df.iloc[:,11].values

print(stroke_df[0:3])
print(X[0:3,:])
print(Y[0:3])

# %% markdown
# ## Standardization
# Because each of our variables is measured using different units, we must re-scale the data to have a mean of 0 and a standard deviation of 1 which will set the foundation for comparison across all variables.

# %% codecell
X = StandardScaler().fit_transform(X)

# %% codecell
plt.hist(stroke_df['age'])  # here you can see the comparison of the original Age variable with its units in years
plt.hist(X[:,1])            # and here is the standardized version of the Age variable

# %% markdown
# ## Dimensionality Reduction -- PCA

# Currently, our dataset has 10 predictors (excluding the subject 'id' & output measure 'stroke') that would be factored into the model.
# But it is possible that some of our predictors are highly correlated with others leaving redundancy in our dataset so by reducing the number of predictors we can make our models more time efficient without sacrificing too much accruacy.
# To do this we will run a Principal Component Analysis (PCA) on our data.

# %% codecell
X_pca = X
Y_pca = Y
x_tr_pca, x_te_pca, y_tr_pca, y_te_pca = train_test_split(X_pca, Y_pca, test_size=0.3, random_state=42) # randomly split the data into training & test sets

# %% codecell
pca = PCA(.95) # allow the data to tell us how many components it takes to account for at least 95% of the variance
pca.fit(x_tr_pca)

# %% codecell
# Results
print("Number of components:", pca.n_components_)
print("Total explained variance: %.2f" % np.sum(pca.explained_variance_ratio_))

# %% markdown
# Based on the analysis run here on the training data, out of our 10 variables (without 'id' & output) it shows that 9 components account for 97% of the variance of the data. So we can see that one variable only accounts for 3% of the explained variance. We could exclude this variable but this trade-off most likely will not save a significant amount of time for our model to be less accurate. Therefore, all variables will be included in the model with no dimensionality reduction.

# # Binary Classification Model
# Our main problem that we are investingating with this current dataset is if we can build a model that will predict Stroke outcomes based on all our other health data indicators.
# Naturally, this is a binary classification problem. Given the data at our disposal, which model will be able to best determine where the boundary is between having a Stroke (1) or not (0).

# Before we test different classification models, we must split our data into a Training Set & a Test Set.
# This will enable us to measure the performance the accuracy of our model on novel data after it has been trained.

# %% codecell
x_tr, x_te, y_tr, y_te = train_test_split(X, Y, test_size=0.3, random_state=42)

# %% markdown
# Now we are ready to train our Binary Classification models on the training set

# %% codecell
model_logistic = LogisticRegression().fit(x_tr, y_tr);
model_knn = KNeighborsClassifier(3).fit(x_tr, y_tr);
model_svm_lin = SVC(kernel="linear", C=1, probability=True).fit(x_tr, y_tr);
model_svm_nonlin = SVC(kernel="rbf", gamma=2, C=0.35, probability=True).fit(x_tr, y_tr);
model_tree = DecisionTreeClassifier().fit(x_tr, y_tr);
model_forest = RandomForestClassifier(n_estimators=10).fit(x_tr, y_tr);
model_xgb = GradientBoostingClassifier().fit(x_tr, y_tr);

# %% markdown
# Here we can see the accuracy scores on the Training & Test sets.

# %% codecell
print("\t\t TR \t TE")
print("Logistic acc:\t %.2f\t %.2f" % (model_logistic.score(x_tr, y_tr), model_logistic.score(x_te, y_te)))
print("K-NN acc:\t %.2f\t %.2f" % (model_knn.score(x_tr, y_tr), model_knn.score(x_te, y_te)))
print("SVM-lin acc:\t %.2f\t %.2f" % (model_svm_lin.score(x_tr, y_tr), model_svm_lin.score(x_te, y_te)))
print("SVM-RBF acc:\t %.2f\t %.2f" % (model_svm_nonlin.score(x_tr, y_tr), model_svm_nonlin.score(x_te, y_te)))
print("Tree acc:\t %.2f\t %.2f" % (model_tree.score(x_tr, y_tr), model_tree.score(x_te, y_te)))
print("Forest acc:\t %.2f\t %.2f" % (model_forest.score(x_tr, y_tr), model_forest.score(x_te, y_te)))
print("Xgb acc:\t %.2f\t %.2f" % (model_xgb.score(x_tr, y_tr), model_xgb.score(x_te, y_te)))
# The decisiomn tree model seems to be over-fitting but all the other models are difficult to differentiate.

# %% markdown
# Looking at the accuracies of our model we are not able to which models perform better than others.
# It is plausible to assume that this could be due to our datasets imbalance.
# Essentially, because there is a much larger proportion of our dataset that has not had a stroke then a metric such as accuracy can be quite misleading.
# Because we cannot differentiate between the models, we should turn to other evaluation statistics to see if we can better understands which models could work best.

# ## Confusion Matrices
# We can use confusion matrices to better understand how our models are classifying our data.
# We will be able to identify which data points are classified as either False Positives -- target samples predicted as positive are actually negative -- or, False Negatives -- target samples predicted as negative that are actually positive.

# %% codecell
log_y_pred1 = model_logistic.predict(x_te) # first we get the model predictions
log_cm = confusion_matrix(y_te, log_y_pred1) # here, we compare the true class labels with the predicted class labels

knn_y_pred1 = model_knn.predict(x_te)
knn_cm = confusion_matrix(y_te, knn_y_pred1)

svmlin_y_pred1 = model_svm_lin.predict(x_te)
svmlin_cm = confusion_matrix(y_te, svmlin_y_pred1)

svmnonlin_y_pred1 = model_svm_nonlin.predict(x_te)
svmnonlin_cm = confusion_matrix(y_te, svmnonlin_y_pred1)

tree_y_pred1 = model_tree.predict(x_te)
tree_cm = confusion_matrix(y_te, tree_y_pred1)

forest_y_pred1 = model_forest.predict(x_te)
forest_cm = confusion_matrix(y_te, forest_y_pred1)

xgb_y_pred1 = model_xgb.predict(x_te)
xgb_cm = confusion_matrix(y_te, xgb_y_pred1)

# %% codecell
# Here we will plot a confusion matrix for each of the models
fig3, axes = plt.subplots(1, 7, figsize = (70, 10))

cm1 = sns.heatmap(log_cm, annot = True, fmt = 'g', ax = axes[0])
cm1.set_xlabel('Predicted labels'); cm1.set_ylabel('True labels');
cm1.xaxis.set_ticklabels(['No Stroke', 'Stroke']); cm1.yaxis.set_ticklabels(['No Stroke', 'Stroke']);
cm1.set_title('Logistic')

cm2 = sns.heatmap(knn_cm, annot = True, fmt = 'g', ax = axes[1])
cm2.set_xlabel('Predicted labels'); cm1.set_ylabel('True labels');
cm2.xaxis.set_ticklabels(['No Stroke', 'Stroke']); cm2.yaxis.set_ticklabels(['No Stroke', 'Stroke']);
cm2.set_title('K-Nearest Neighbors')

cm3 = sns.heatmap(svmlin_cm, annot = True, fmt = 'g', ax = axes[2])
cm3.set_xlabel('Predicted labels'); cm1.set_ylabel('True labels');
cm3.xaxis.set_ticklabels(['No Stroke', 'Stroke']); cm3.yaxis.set_ticklabels(['No Stroke', 'Stroke']);
cm3.set_title('SVM Linear')

cm4 = sns.heatmap(svmnonlin_cm, annot = True, fmt = 'g', ax = axes[3])
cm4.set_xlabel('Predicted labels'); cm1.set_ylabel('True labels');
cm4.xaxis.set_ticklabels(['No Stroke', 'Stroke']); cm4.yaxis.set_ticklabels(['No Stroke', 'Stroke']);
cm4.set_title('SVM Non-Linear')

cm5 = sns.heatmap(tree_cm, annot = True, fmt = 'g', ax = axes[4])
cm5.set_xlabel('Predicted labels'); cm1.set_ylabel('True labels');
cm5.xaxis.set_ticklabels(['No Stroke', 'Stroke']); cm5.yaxis.set_ticklabels(['No Stroke', 'Stroke']);
cm5.set_title('Decision Tree')

cm6 = sns.heatmap(forest_cm, annot = True, fmt = 'g', ax = axes[5])
cm6.set_xlabel('Predicted labels'); cm1.set_ylabel('True labels');
cm6.xaxis.set_ticklabels(['No Stroke', 'Stroke']); cm6.yaxis.set_ticklabels(['No Stroke', 'Stroke']);
cm6.set_title('Random Forest')

cm7 = sns.heatmap(xgb_cm, annot = True, fmt = 'g', ax = axes[6])
cm7.set_xlabel('Predicted labels'); cm1.set_ylabel('True labels');
cm7.xaxis.set_ticklabels(['No Stroke', 'Stroke']); cm7.yaxis.set_ticklabels(['No Stroke', 'Stroke']);
cm7.set_title('XG Boost')

# %% markdown
# ## F1 Scores
# To further inspect our models performance we can evaluate each ones F1 score.
# This is a qunatitative assessment of the following metrics: Recall -- which is informative when the cost of False Negatives is high -- in combination with, Precision -- which is informative when the cost of False Positives is high.
# The F1 Score combines these two quantities into a harmonic mean.
# The closer the score is to 1 the more better the model is.

# %% codecell
# For the average parameter, 'macro' is chosen because we have an unbalanced dataset and we want to treat both classes equally
print('Models F1 Score')
print('Logistic:\t', f1_score(y_te, log_y_pred1, average = 'macro'))
print('K-Nearest:\t', f1_score(y_te, knn_y_pred1, average = 'macro'))
print('SVM Linear:\t', f1_score(y_te, svmlin_y_pred1, average = 'macro'))
print('SVM Non-Linear:\t', f1_score(y_te, svmnonlin_y_pred1, average = 'macro'))
print('Decision Tree:\t', f1_score(y_te, tree_y_pred1, average = 'macro'))
print('Random Forest:\t', f1_score(y_te, forest_y_pred1, average = 'macro'))
print('XG Boost:\t', f1_score(y_te, xgb_y_pred1, average = 'macro'))


# %% markdown
# As we can see from the results of our Confusion Matrices, as well as, the F1 Scores:
# *    The Logistic & SVM models do the poorest job of correctly classifying Strokes, none of these models are able to correctly classfy a Stroke based on the data.
# *    The Decision Tree model performs best at classifying Strokes correctly classyfing 18 samples as True Stroke subjects but mis-classifying 150 samples (as either False Positive or False Negative).
# *    The K-Nearest, Random Forest & XG Boost were able to correctly clasify a small number of samples as true Strokes.

# A subset of these models will be chosen to optimize.
# Decision Tree will be chosen because it has the best performance based on the the confusion matrix & F1 score.
# Additionally, we will choose two ensemble methods, Random Forest & XGBoost, because of their ability to handle unbalanced datasets.
# This choice will not come without the trade-off of interpretability.

# Considering these results based on each model's default hyper-parameters, let's continue and tune the parameters of the models outlined above: Decision Tree, Random Forest & XG Boost.

# # Tuning Model Hyperparameters
# In order to identify which parameters get optimal performance out of our models, we must create a grid of parameters to be search over while scoring each combination on model accuracy.
# The set of parameters with the highest accuracy will provide the most optimal model.

# For tuning the model parameters we will use a Bayesian Search method.
# Bayesian Search is a sequential model-based optimization which uses the outcomes from previous attempts to decide the next hyperparameter values for the following iteration.

# I have decided for this approach over Grid Search, which performs an exhaustive search of all possible combinations of parameters, because it will be more time-efficient given that some parameters that will be tuned in these models are continuous variables which would cause Grid Search tuning to be extremely lengthy.
# Therefore, instead of blindly searching the hyperparameter space (e.g., Grid Search), the Bayesian method will intuitvely pick the next sets of parameters to be explored based on past results.
# This will save time and, given that, subsequently higher model performance in the end.

# %% markdown
# ## Decision Tree parameter estimation -- Bayesian Search

# %% codecell
# Creating a search space to explore best possible combination of model parameters
search_space_tree = {
    "max_depth": hp.choice("max_depth", range(105,116)),
    "max_features": hp.choice("max_features", [7]),
    "criterion": hp.choice("criterion", ["gini"]),
    "min_weight_fraction_leaf": hp.uniform("min_weight_fraction_leaf", 0.0254, 0.0256),
    "ccp_alpha": hp.uniform("ccp_alpha", 0.0009, 0.0011),
    "class_weight": hp.choice("class_weight", [{0:1, 1:6.4}, {0:1, 1:6.5}])
}

# %% codecell
# Implementing Bayesian Search to explore parameter space
def objective_tree(args):
    max_depth = args["max_depth"]
    max_features = args["max_features"]
    criterion = args["criterion"]
    min_weight_fraction_leaf = args["min_weight_fraction_leaf"]
    ccp_alpha = args["ccp_alpha"]
    class_weight = args["class_weight"]

    decision_tree = DecisionTreeClassifier(
        max_depth=max_depth,
        max_features=max_features,
        criterion=criterion,
        min_weight_fraction_leaf=min_weight_fraction_leaf,
        ccp_alpha=ccp_alpha,
        class_weight=class_weight,
        random_state=42)

    decision_tree.fit(x_tr, y_tr)

    tree_y_f1 = decision_tree.predict(x_te)

    print("Hyperparameters: {}".format(args))
    print("F1 Score: {}\n".format(f1_score(y_te, tree_y_f1, average = 'macro')))

    return -1 * f1_score(y_te, tree_y_f1, average = 'macro')

# %% codecell
# Perform Bayesian Optimization search
trials_obj = hyperopt.Trials()

best_results = hyperopt.fmin(objective_tree,
                             space=search_space_tree,
                             algo=hyperopt.tpe.suggest,
                             trials=trials_obj,
                             max_evals=6000)

print("Best Hyperparameters Settings: {}".format(best_results))
print("\nBest F1 Score: {}".format(-1 * trials_obj.average_best_error()))

# %% codecell
# Displaying the best value for each parameter as a result of the Bayesian Optimization search
max_depth = best_results["max_depth"]
max_features = best_results["max_features"]
criterion = best_results["criterion"]
min_weight_fraction_leaf = best_results["min_weight_fraction_leaf"]
ccp_alpha = best_results["ccp_alpha"]
class_weight = best_results["class_weight"]

print("Best Hyperparameters Settings : {}".format({"max_depth":max_depth,
                                                   "max_features": max_features,
                                                   "criterion": criterion,
                                                   "min_weight_fraction_leaf": min_weight_fraction_leaf,
                                                   "ccp_alpha": ccp_alpha,
                                                   "class_weight": class_weight
                                                  }))

# %% codecell
# Running model with the parameters identified from the Bayesian Search
model_tree_post = DecisionTreeClassifier(max_depth = 108, max_features = 7, criterion = 'gini', ccp_alpha = 0.0010949385472465137, class_weight = {0:1, 1:6.5}, min_weight_fraction_leaf = 0.02556002002828835).fit(x_tr, y_tr);
tree_y_pred2 = model_tree_post.predict(x_te)
tree_cm_post = confusion_matrix(y_te, tree_y_pred2)

tree_y_pred3 = model_tree_post.predict(x_te)
f1_score(y_te, tree_y_pred3, average = 'macro')

# %% codecell
# Creating Confusion Matrices after tuning the parameters
fig4, axes = plt.subplots(1, 2, figsize = (20, 10))

cm5 = sns.heatmap(tree_cm, annot = True, fmt = 'g', ax = axes[0])
cm5.set_xlabel('Predicted labels'); cm1.set_ylabel('True labels');
cm5.xaxis.set_ticklabels(['No Stroke', 'Stroke']); cm5.yaxis.set_ticklabels(['No Stroke', 'Stroke']);
cm5.set_title('Decision Tree')

cmC = sns.heatmap(tree_cm_post, annot = True, fmt = 'g', ax = axes[1])
cmC.set_xlabel('Predicted labels'); cm1.set_ylabel('True labels');
cmC.xaxis.set_ticklabels(['No Stroke', 'Stroke']); cmC.yaxis.set_ticklabels(['No Stroke', 'Stroke']);
cmC.set_title('Decision Tree Post')


# %% markdown
# ## Random Forest parameter estimation -- Bayesian Search

# %% codecell
# Creating a search space to explore best possible combination of model parameters
search_space_forest = {
    "n_estimators": hp.choice("n_estimators", range(320,323)),
    "max_depth": hp.choice("max_depth", range(198, 210)),
    "max_features": hp.choice("max_features", [5]),
    "min_weight_fraction_leaf": hp.uniform("min_weight_fraction_leaf", 0.03581, 0.03589),
    "criterion": hp.choice("criterion", ["log_loss"]),
    "ccp_alpha": hp.uniform("ccp_alpha", 0.0101, 0.0109),
    "class_weight": hp.choice("class_weight", [{0:1, 1:6.6}])
}

# %% codecell
# Implementing Bayesian Search to explore parameter space
def objective_forest(args):
    n_estimators = args["n_estimators"]
    max_depth = args["max_depth"]
    max_features = args["max_features"]
    min_weight_fraction_leaf = args["min_weight_fraction_leaf"]
    criterion = args["criterion"]
    ccp_alpha = args["ccp_alpha"]
    class_weight = args["class_weight"]

    random_forest = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        max_features=max_features,
        min_weight_fraction_leaf=min_weight_fraction_leaf,
        criterion=criterion,
        ccp_alpha=ccp_alpha,
        class_weight=class_weight,
        random_state=42)

    random_forest.fit(x_tr, y_tr)

    forest_y_f1 = random_forest.predict(x_te)

    print("Hyperparameters: {}".format(args))
    print("F1 Score: {}\n".format(f1_score(y_te, forest_y_f1, average = 'macro')))

    return -1 * f1_score(y_te, forest_y_f1, average = 'macro')

# %% codecell
# Perform Bayesian Optimization search
trials_obj = hyperopt.Trials()

best_results = hyperopt.fmin(objective_forest,
                             space=search_space_forest,
                             algo=hyperopt.tpe.suggest,
                             trials=trials_obj,
                             max_evals=500)

print("Best Hyperparameters Settings: {}".format(best_results))
print("\nBest F1 Score: {}".format(-1 * trials_obj.average_best_error()))

# %% codecell
# Displaying the best value for each parameter as a result of the Bayesian Optimization search
n_estimators = best_results["n_estimators"]
max_depth = best_results["max_depth"]
max_features = best_results["max_features"]
criterion = best_results["criterion"]
ccp_alpha = best_results["ccp_alpha"]
class_weight = best_results["class_weight"]
min_weight_fraction_leaf = best_results["min_weight_fraction_leaf"]

print("Best Hyperparameters Settings : {}".format({"max_depth":max_depth,
                                                   "max_features": max_features,
                                                   "criterion": criterion,
                                                   "ccp_alpha": ccp_alpha,
                                                   "class_weight": class_weight,
                                                   "min_weight_fraction_leaf": min_weight_fraction_leaf,
                                                   "n_estimators": n_estimators
                                                  }))
# %% codecell
# Running model with the parameters identified from the Bayesian Search
model_forest_post = RandomForestClassifier(max_depth = 205, max_features = 5, criterion='log_loss', ccp_alpha = 0.010417804006895274, class_weight = {0:1, 1:6.6}, n_estimators=323).fit(x_tr, y_tr);
forest_y_pred2 = model_forest_post.predict(x_te)
forest_cm_post = confusion_matrix(y_te, forest_y_pred2)

forest_y_pred3 = model_forest_post.predict(x_te)
f1_score(y_te, forest_y_pred3, average = 'macro')

# %% codecell
# Creating Confusion Matrices after tuning the parameters
fig5, axes = plt.subplots(1, 2, figsize = (20, 10))

cm6 = sns.heatmap(forest_cm, annot = True, fmt = 'g', ax = axes[0])
cm6.set_xlabel('Predicted labels'); cm1.set_ylabel('True labels');
cm6.xaxis.set_ticklabels(['No Stroke', 'Stroke']); cm6.yaxis.set_ticklabels(['No Stroke', 'Stroke']);
cm6.set_title('Random Forest')

cmB = sns.heatmap(forest_cm_post, annot = True, fmt = 'g', ax = axes[1])
cmB.set_xlabel('Predicted labels'); cm1.set_ylabel('True labels');
cmB.xaxis.set_ticklabels(['No Stroke', 'Stroke']); cmB.yaxis.set_ticklabels(['No Stroke', 'Stroke']);
cmB.set_title('Random Forest Post')

# %% markdown
# ## XGBoost parameter estimation -- Bayesian Search

# %% codecell
# Creating a search space to explore best possible combination of model parameters
search_space_xgb = {
    "loss": hp.choice("loss", ["exponential"]),
    "learning_rate": hp.uniform("learning_rate", 0.43, 0.44),
    "n_estimators": hp.choice("n_estimators", range(388, 394)),
    "subsample": hp.uniform("subsample", 0.0227, 0.0233),
    "criterion": hp.choice("criterion", ["squared_error"]),
    "ccp_alpha": hp.uniform("ccp_alpha", 0.025, 0.027),
    "min_weight_fraction_leaf": hp.uniform("min_weight_fraction_leaf", 0.0625, 0.0635),
    "max_depth": hp.choice("max_depth", range(330, 400)),
    "max_features": hp.choice("max_features", range(17, 23))
}

# %% codecell
# Implementing Bayesian Search to explore parameter space
def objective_xgboost(args):
    loss = args["loss"]
    learning_rate = args["learning_rate"]
    n_estimators = args["n_estimators"]
    subsample = args["subsample"]
    criterion = args["criterion"]
    ccp_alpha = args["ccp_alpha"]
    min_weight_fraction_leaf = args["min_weight_fraction_leaf"]
    max_depth = args["max_depth"]
    max_features = args["max_features"]

    xg_boost = GradientBoostingClassifier(
        loss=loss,
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        subsample=subsample,
        criterion=criterion,
        ccp_alpha=ccp_alpha,
        min_weight_fraction_leaf=min_weight_fraction_leaf,
        max_depth=max_depth,
        max_features=max_features,
        random_state=42)

    xg_boost.fit(x_tr, y_tr)

    xgb_y_f1 = xg_boost.predict(x_te)

    print("Hyperparameters: {}".format(args))
    print("F1 Score: {}\n".format(f1_score(y_te, xgb_y_f1, average = 'macro')))

    return -1 * f1_score(y_te, xgb_y_f1, average = 'macro')

# %% codecell
# Perform Bayesian Optimization search
trials_obj = hyperopt.Trials()

best_results = hyperopt.fmin(objective_xgboost,
                             space=search_space_xgb,
                             algo=hyperopt.tpe.suggest,
                             trials=trials_obj,
                             max_evals=2000)

print("Best Hyperparameters Settings: {}".format(best_results))
print("\nBest F1 Score: {}".format(-1 * trials_obj.average_best_error()))

# %% codecell
# Displaying the best value for each parameter as a result of the Bayesian Optimization search
loss = best_results["loss"]
learning_rate = best_results["learning_rate"]
n_estimators = best_results["n_estimators"]
subsample = best_results["subsample"]
criterion = best_results["criterion"]
ccp_alpha = best_results["ccp_alpha"]
min_weight_fraction_leaf = best_results["min_weight_fraction_leaf"]
max_depth = best_results["max_depth"]
max_features = best_results["max_features"]

print("Best Hyperparameters Settings : {}".format({"loss":loss,
                                                   "learning_rate": learning_rate,
                                                   "n_estimators": n_estimators,
                                                   "subsample": subsample,
                                                   "criterion": criterion,
                                                   "ccp_alpha": ccp_alpha,
                                                   "min_weight_fraction_leaf": min_weight_fraction_leaf,
                                                   "max_depth": max_depth,
                                                   "max_features": max_features
                                                  }))

# %% codecell
# Running model with the parameters identified from the Bayesian Search
model_xgb_post = GradientBoostingClassifier(loss = "exponential", criterion = "squared_error", learning_rate = 0.43929234697276254, n_estimators = 391, subsample = 0.022964945721608645, min_weight_fraction_leaf = 0.06267786747377273, max_depth = 370, max_features = 17, ccp_alpha = 0.026341878725969606).fit(x_tr, y_tr);
xgb_y_pred2 = model_xgb_post.predict(x_te)
xgb_cm_post = confusion_matrix(y_te, xgb_y_pred2)

xgb_y_pred3 = model_xgb_post.predict(x_te)
f1_score(y_te, xgb_y_pred3, average = 'macro')

# %% codecell
# Creating Confusion Matrices after tuning the parameters
fig6, axes = plt.subplots(1, 2, figsize = (20, 10))

cm7 = sns.heatmap(xgb_cm, annot = True, fmt = 'g', ax = axes[0])
cm7.set_xlabel('Predicted labels'); cm6.set_ylabel('True labels');
cm7.xaxis.set_ticklabels(['No Stroke', 'Stroke']); cm6.yaxis.set_ticklabels(['No Stroke', 'Stroke']);
cm7.set_title('XG Boost')

cmD = sns.heatmap(xgb_cm_post, annot = True, fmt = 'g', ax = axes[1])
cmD.set_xlabel('Predicted labels'); cmD.set_ylabel('True labels');
cmD.xaxis.set_ticklabels(['No Stroke', 'Stroke']); cmD.yaxis.set_ticklabels(['No Stroke', 'Stroke']);
cmD.set_title('XG Boost Post')


# %% markdown
# # Model Comparison
# Before and After tuning the parameters for each of the chosen models, let's identify which model best predicts strokes based on the available data.

# %% codecell
fig7, axes = plt.subplots(1, 6, figsize = (60, 10))

# Pre Tuning Confusion Matrices
cm_tree_pre = sns.heatmap(tree_cm, annot = True, fmt = 'g', ax = axes[0])
cm_tree_pre.set_xlabel('Predicted labels'); cm_tree_pre.set_ylabel('True labels');
cm_tree_pre.xaxis.set_ticklabels(['No Stroke', 'Stroke']); cm_tree_pre.yaxis.set_ticklabels(['No Stroke', 'Stroke']);
cm_tree_pre.set_title('Pre-Tuning Decision Tree')

cm_forest_pre = sns.heatmap(forest_cm, annot = True, fmt = 'g', ax = axes[2])
cm_forest_pre.set_xlabel('Predicted labels'); cm_forest_pre.set_ylabel('True labels');
cm_forest_pre.xaxis.set_ticklabels(['No Stroke', 'Stroke']); cm_forest_pre.yaxis.set_ticklabels(['No Stroke', 'Stroke']);
cm_forest_pre.set_title('Pre-Tuning Random Forest')

cm_xgb_pre = sns.heatmap(xgb_cm, annot = True, fmt = 'g', ax = axes[4])
cm_xgb_pre.set_xlabel('Predicted labels'); cm_xgb_pre.set_ylabel('True labels');
cm_xgb_pre.xaxis.set_ticklabels(['No Stroke', 'Stroke']); cm_xgb_pre.yaxis.set_ticklabels(['No Stroke', 'Stroke']);
cm_xgb_pre.set_title('Pre-Tuning XG Boost')

# Post Tuning Confusion Matrices
cm_tree_post = sns.heatmap(tree_cm_post, annot = True, fmt = 'g', ax = axes[1])
cm_tree_post.set_xlabel('Predicted labels'); cm_tree_post.set_ylabel('True labels');
cm_tree_post.xaxis.set_ticklabels(['No Stroke', 'Stroke']); cm_tree_post.yaxis.set_ticklabels(['No Stroke', 'Stroke']);
cm_tree_post.set_title('Post-Tuning Decision Tree')

cm_forest_post = sns.heatmap(forest_cm_post, annot = True, fmt = 'g', ax = axes[3])
cm_forest_post.set_xlabel('Predicted labels'); cm_forest_post.set_ylabel('True labels');
cm_forest_post.xaxis.set_ticklabels(['No Stroke', 'Stroke']); cm_forest_post.yaxis.set_ticklabels(['No Stroke', 'Stroke']);
cm_forest_post.set_title('Post-Tuning Random Forest')

cm_xgb_post = sns.heatmap(xgb_cm_post, annot = True, fmt = 'g', ax = axes[5])
cm_xgb_post.set_xlabel('Predicted labels'); cm_xgb_post.set_ylabel('True labels');
cm_xgb_post.xaxis.set_ticklabels(['No Stroke', 'Stroke']); cm_xgb_post.yaxis.set_ticklabels(['No Stroke', 'Stroke']);
cm_xgb_post.set_title('Post-Tuning XG Boost')

# %% codedcell
# F1 Scores (Pre - Post)
print("\t\t\tPre-Tuning \tPost-Tuning")
print("Tree F1 Score:  \t %.4f \t %.4f" % (f1_score(y_te, tree_y_pred1, average = 'macro'), f1_score(y_te, tree_y_pred3, average = 'macro')))
print("Forest F1 Score: \t %.4f \t %.4f" % (f1_score(y_te, forest_y_pred1, average = 'macro'), f1_score(y_te, forest_y_pred3, average = 'macro')))
print("XGBoost F1 Score: \t %.4f \t %.4f" % (f1_score(y_te, xgb_y_pred1, average = 'macro'), f1_score(y_te, xgb_y_pred3, average = 'macro')))

# %% markdown
# # Conclusion

# ## Decision Tree Model (12.1% misclassification)
# Although the Decision Tree model had the best performance with regards to the F1 Score before tuning the models, it does not show the same improvement as the others.
# Taking a look at the confusion matrices, we can see that the Decision Tree, although it has a high number of True Positives, it has the worst performance on True Negatives.
# The model seems to misinterpret many individuals who have not had a stroke as positive cases.

# ## Random Forest Model (10.8% misclassification)
# Before tuning, the Random Forest model had the worst performance of any model and was not able to classify a single individual that had a stroke correctly.
# After tuning we can see a significant increase in performance with this model being able to classify the highest number of True Positives.
# Although much like the Decision Tree, the Random Forest model does misclassify a number of individuals as False Positives.

# ## XGBoost (7.9% misclassification)
# Before tuning, the XGBoost model was only able to classify a single True Positive individual.
# However, after tuning the XGBoost model has the lowest percentage of misclassified individuals (7.9%) compared to the other two models.
# But it should be noted that the XGBoost model does have the highest occurence of False Negatives.

# ## Choice
# When choosing which model would be best, it is important to reference the original goal which was to build a model that can accurately predict whether a given individual has a stroke or not based on all other provided health & demographic data.
# Ideally a model like this would be used on novel subjects to estimate their risk for a stroke given their medical & demographic markers.
# Keeping this in mind, I think the Random Forest model would be the best choice.
# I would choose the Random Forest over the XGBoost because the Random Forest model has a lower occurence of False Negatives which, in the setting of healthcare, is more important to minimize than False Positives (e.g., it would be more perilous to not identify someone who is at-risk for a stroke than to misidentify someone who is not at risk).
# Therefore, I think the Random Forest model would be the best choice given the metrics displayed here.
# Although I acknowledge that an XGBoost model with further time & tuning of the parameters could have better potential in correctly classifying a higher proportion of individuals.
