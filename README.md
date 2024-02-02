# Machine Learning Project
This GitHub repository stores a .pdf document of the project description from a course from the Universita degli Studi di Padova on Machine Learning, as well as, the dataset from which the model was trained & tested and, lastly, the python code with comments outlining the model that was built & interpretation of it's results.

In this README file I will leave short descriptions of each of these elements. This repository will be open access so please consult the materials for further detail. 

**Purpose**: This repository is intended to be a demonstrattion of the skills I acquired while completing this project, as well as, an educational guide for anyone looking to acquire those same skills.

### Dataset
The dataset used for this project is a .csv file containing the following attributes regarding health & demographic information:
*   id: unique identifier
*   gender: "Male", "Female" or "Other"
*   age: age of the patient
*   hypertension: 0 if the patient doesn't have hypertension, 1 if the patient has hypertension
*   heart_disease: 0 if the patient doesn't have any heart diseases, 1 if the patient has a heart disease
*   ever_married: "No" or "Yes"
*   work_type: "children", "Govt_jov", "Never_worked", "Private" or "Self-employed"
*   Residence_type: "Rural" or "Urban"
*   avg_glucose_level: average glucose level in blood
*   bmi: body mass index
*   smoking_status: "formerly smoked", "never smoked", "smokes" or "Unknown"
*   stroke: 1 if the patient had a stroke or 0 if not

**GOAL**: This information was used to build a model that could accurately predict whether a given individual had a stroke or not based on all other datapoints.

### Code
This project was my first foray into both machine learning and practically implementing python as a programming language. Through this work, I was able to better understand the concepts studied during the _Machine Leanring for Brain & Cognition_ course and how to use python to tackle problems concerning large amounts of data. 

The entirety of the code is made up of the following sub-sections. Additionally I have provided a short description on the function of each of these sections to better understand why they were included.

1. **Pre-processing & Data Cleaning**: Identifying data points with missing values within the dataset. This issue was addressed using Multivariate Imputation.
2. **Data Visualization**: Observing the distribution of data for each of the variables involved.
3. **Data Standardization**: Re-scaling the data for comparison across variables with different units of measurement.
4. **PCA Dimensionality Reduction**: Checking for the redundancy in the number of predictor variables to aid in model efficiency without trading off too much accuracy in prediction. (All variables ended up being included without any dimensional reduction)
5. **Binary Classification**: Creating a model that establishes a boundary between having had a Stroke or not based on the predictor variables.
6. **Confusion Matrices**: Visualizing how different models properly or improperly classify subjects to better understand model performance.
7. **Model Evaluation & Hyperparameter Tuning using Bayesian Search**: After evaluating the different models & selecting a subset of the best performing ones, Bayesian Search - a sequential model-based optimization technique which uses the outcomes from previous attempts to decide the next hyperparameter values for the following iteration - to find the optimal combination of hyperparameters for the selected models.
8. **Model Comparison**: Comparing the 3 best models on how they perform on predicting stroke outcomes.

### Conclusion
Here are the 3 highest performing models that I found with their misclassification percentage:

**Decision Tree Model** _12.1% misclassification_
Although the Decision Tree model had the best performance with regards to the F1 Score before tuning the models, it does not show the same improvement as the others. Taking a look at the confusion matrices, we can see that the Decision Tree, although it has a high number of True Positives, it has the worst performance on True Negatives. The model seems to misinterpret many individuals who have not had a stroke as positive cases.

**Random Forest Model** _10.8% misclassification_
Before tuning, the Random Forest model had the worst performance of any model and was not able to classify a single individual that had a stroke correctly. After tuning we can see a significant increase in performance with this model being able to classify the highest number of True Positives. Although much like the Decision Tree, the Random Forest model does misclassify a number of individuals as False Positives.

**XGBoost Model** _7.9& misclassification_
Before tuning, the XGBoost model was only able to classify a single True Positive individual. However, after tuning the XGBoost model has the lowest percentage of misclassified individuals (7.9%) compared to the other two models. But it should be noted that the XGBoost model does have the highest occurence of False Negatives.

#### Choice
When choosing which model would be best, it is important to reference the original goal which was to build a model that can accurately predict whether a given individual has a stroke or not based on all other provided health & demographic data. Ideally a model like this would be used on novel subjects to estimate their risk for a stroke given their medical & demographic markers. Keeping this in mind, I think the Random Forest model would be the best choice. I would choose the Random Forest over the XGBoost because the Random Forest model has a lower occurence of False Negatives which, in the setting of healthcare, is more important to minimize than False Positives (e.g., it would be more perilous to not identify someone who is at-risk for a stroke than to misidentify someone who is not at risk). Therefore, I think the Random Forest model would be the best choice given the metrics displayed here. Although I acknowledge that an XGBoost model with further time & tuning of the parameters could have better potential in correctly classifying a higher proportion of individuals.
