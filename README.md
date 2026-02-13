# Heart Disease Classification – Machine Learning Assignment 2

## Problem Statement
The objective of this project is to build and compare multiple machine learning classification models to predict the presence of heart disease in patients based on clinical attributes. The goal is to evaluate how different algorithms perform on the same dataset using standard evaluation metrics.

---

## Dataset Description
The dataset used in this project is the Heart Disease dataset obtained from Kaggle.  
It contains more than 1000 patient records and 13 input features such as age, sex, chest pain type, cholesterol level, maximum heart rate, etc.

Target Variable:
- 0 → No heart disease  
- 1 → Presence of heart disease

The dataset satisfies the assignment requirement of having more than 500 instances and at least 12 features.

---

## Models Used
The following six classification algorithms were implemented:

1. Logistic Regression  
2. Decision Tree  
3. K-Nearest Neighbors (kNN)  
4. Naive Bayes  
5. Random Forest (Ensemble)  
6. XGBoost (Ensemble)

---

## Model Comparison Table

| ML Model | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|----------|----------|-----|-----------|--------|----------|-----|
| Logistic Regression | 0.795 | 0.877 | 0.756 | 0.873 | 0.810 | 0.597 |
| Decision Tree | 0.985 | 0.985 | 1.000 | 0.970 | 0.985 | 0.971 |
| kNN | 0.731 | 0.860 | 0.730 | 0.737 | 0.734 | 0.463 |
| Naive Bayes | 0.800 | 0.870 | 0.754 | 0.893 | 0.817 | 0.610 |
| Random Forest | 0.985 | 1.000 | 1.000 | 0.970 | 0.985 | 0.971 |
| XGBoost | 0.985 | 0.989 | 1.000 | 0.970 | 0.985 | 0.971 |


---

## Observations

### Logistic Regression
Logistic Regression provided a good baseline performance with a reasonable balance between precision and recall. However, it may not capture complex non-linear patterns as effectively as ensemble methods.

### Decision Tree
Decision Tree achieved very high accuracy and MCC. It is strong in learning non-linear decision boundaries but may sometimes overfit.

### kNN
kNN showed comparatively lower performance. The algorithm depends heavily on distance measures and may not capture deeper feature relationships.

### Naive Bayes
Naive Bayes performed fairly well despite assuming feature independence. It is efficient but slightly behind tree-based ensemble models.

### Random Forest
Random Forest delivered one of the best performances across all metrics. Combining multiple trees helped improve robustness and reduce overfitting.

### XGBoost
XGBoost also achieved excellent results with high accuracy and AUC. Its boosting mechanism improves predictions iteratively, making it highly effective.
