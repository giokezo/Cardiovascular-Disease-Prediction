# Cardiovascular-Disease-Prediction
A comprehensive machine learning project for predicting cardiovascular disease using the Kaggle CVD dataset (70,000+ patient records). This project demonstrates a complete ML workflow from exploratory data analysis to model deployment, comparing multiple algorithms to identify the most effective approach for early disease detection.


## Data description
There are 3 types of input features:

Objective: factual information;
Examination: results of medical examination;
Subjective: information given by the patient.
Features:

* Age | Objective Feature | age | int (days)
* Height | Objective Feature | height | int (cm) |
* Weight | Objective Feature | weight | float (kg) |
* Gender | Objective Feature | gender | categorical code |
* Systolic blood pressure | Examination Feature | ap_hi | int |
* Diastolic blood pressure | Examination Feature | ap_lo | int |
* Cholesterol | Examination Feature | cholesterol | 1: normal, 2: above normal, 3: well above normal |
* Glucose | Examination Feature | gluc | 1: normal, 2: above normal, 3: well above normal |
* Smoking | Subjective Feature | smoke | binary |
* Alcohol intake | Subjective Feature | alco | binary |
* Physical activity | Subjective Feature | active | binary |
* Presence or absence of cardiovascular disease | Target Variable | cardio | binary |
* All of the dataset values were collected at the moment of medical examination.


## Model Evaluation Results

### Performance Metrics Summary

| Model | Accuracy | Precision | Recall | F1 Score | AUC-ROC |
|-------|----------|-----------|--------|----------|---------|
| Logistic Regression | 70.10% | 70.74% | 68.24% | 69.47% | 0.75 |
| Decision Tree | 72.83% | 72.58% | **73.11%** | 72.84% | 0.78 |
| XGBoost | 73.59% | 75.52% | 69.56% | 72.42% | **0.80** |
| Naive Bayes | 72.19% | 77.45% | 62.35% | 69.08% | 0.78 |

## Key Findings

### Best Performing Model: **Decision Tree** 
- **Recall:** 73.11% (highest - most important for medical diagnosis)
- **Accuracy:** 72.83%
- **AUC-ROC:** 0.78
- **F1 Score:** 72.84%

### Why Recall Matters Most

In cardiovascular disease prediction, **minimizing false negatives is critical**. Missing a patient with CVD (false negative) can have life-threatening consequences, while a false positive only leads to further testing.

**Decision Tree** catches **73.11% of all CVD cases**, making it the safest choice for clinical screening.

### Model Comparison

**Decision Tree** 
- **Highest Recall (73.11%)** - catches the most CVD cases
- Minimizes dangerous false negatives
- Good balance with 72.58% precision
- Most interpretable for clinicians

**XGBoost**
- Best accuracy (73.59%) and AUC (0.80)
- Lower recall (69.56%) - misses more CVD cases
- Higher precision but less suitable for screening

**Logistic Regression**
- Moderate recall (68.24%)
- Simple and interpretable
- Baseline performance

**Naive Bayes**
- **Lowest Recall (62.35%)** - misses too many CVD cases
- Highest precision but unsafe for screening
- Not recommended for medical diagnosis

## Recommendation

**Deploy Decision Tree** as the primary screening model due to:
1. **Highest Recall (73.11%)** - catches the most patients with CVD
2. Minimizes life-threatening false negatives
3. Reasonable precision (72.58%) keeps false positives manageable
4. Interpretable decision rules for clinicians

## Clinical Impact

With the Decision Tree model:
- **73% of CVD patients are correctly identified** (highest among all models)
- Only ~27% of CVD cases are missed (lowest false negative rate)
- When predicting CVD, ~73% are true positives
- Best balance between catching disease and minimizing unnecessary testing

## Conclusion

For cardiovascular disease screening, **Decision Tree is the optimal model** with 73.11% recall, ensuring the fewest missed diagnoses. While XGBoost has higher overall accuracy, Decision Tree's superior recall makes it the safest and most appropriate choice for medical screening where false negatives can be fatal.