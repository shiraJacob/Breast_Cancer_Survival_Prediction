# Breast Cancer Survival Prediction using Machine Learning
Application of machine learning classification algorithms (XGBoost, RF, SVM, MLP) to predict breast cancer patient survival, enhancing predictive accuracy through data preprocessing, feature engineering, and RFECV-based feature selection, effectively addressing challenges of imbalanced medical datasets.

🔧 **Installation & Setup**
Install the required packages with:
pip install numpy pandas seaborn matplotlib scikit-learn xgboost imbalanced-learn statsmodels tabulate
Make sure Breast_Cancer.csv is in the same directory as main.py, then run: main.py.

## Dataset:
- 4,024 records from the Kaggle dataset based on the November 2017 update of the National Cancer Institute (NCI) SEER program.
- 16 clinical and demographic features.
- Highly imbalanced: ~15% did not survive, ~85% survived.

Full statistical analysis was performed for each feature and the interactions between them.
<div align="center"> <img src="https://github.com/user-attachments/assets/d1859c02-d889-4b40-b6ff-ecef736475ea" width="400"/> </div>
<sub>Left: Interaction between Tumor Size and N Stage | Right: Tumor Size and T Stage</sub>

## Data Preparation
- **Outlier Handling:** Using a threshold of 3*IQR, suitable for medical data.
- **Balancing:** Downsampling the majority class to a 2:1 ratio.
- **Discretization:** Converted continuous age into meaningful categorical bins.
- **Normalization:** Min-Max scaling of quantitative variables.
- **Categorical Encoding:** One-hot encoding for non-ordinal variables; ordinal mapping for ordered variables.
- **Feature Engineering:** Created a new feature – ratio of positive to examined lymph nodes.
<div align="center"> <img src="https://github.com/user-attachments/assets/3482cb41-c5c4-4ca7-84a2-8195167e944a" width="400"/> </div>
<sub>Before data preparation</sub>

<div align="center"> <img src="https://github.com/user-attachments/assets/26339a7c-3962-4405-923d-33c8cf65aadf" width="400"/> </div>
<sub>After data preparation</sub>

## Feature Selection
- Applied Recursive Feature Elimination with Cross-Validation (RFECV) using a Random Forest classifier and AUC-ROC as the evaluation metric.
- Iteratively removed the least important features using 5-fold stratified cross-validation.
- Resulted in a focused, high-performing feature set used for model training.

## Modeling and Evaluation
**Modeling:**
- Trained and evaluated: Random Forest (RF), XGBoost, Support Vector Machine (SVM), and Multi-Layer Perceptron (MLP).
- Used RandomizedSearchCV for hyperparameter tuning across all models.
**Evaluation:**
- AUC-ROC: The primary metric, chosen for its robustness with imbalanced datasets.
- Recall: Prioritized to minimize false negatives.
- F2 Score: Weighted metric emphasizing Recall over Precision, critical in medical contexts.

## Results & Discussion
<div align="center"> <img src="https://github.com/user-attachments/assets/65314a16-6fdc-417a-bfb0-de6ad50afb3c" width="400"/> </div>
- MLP Achieved the highest AUC-ROC (0.867) but showed poor Recall (0.661) and F2 Score (0.6842).
- XGBoost the best balanced performance with high AUC-ROC (0.8662), Recall (0.8813) and F2 Score (0.7715).
- Feature importance: SurvivalMonthsNormalized and RegionalNodeRatio emerged as key predictors.

## What I learned:
- Applied classification algorithms for medical survival prediction.
- Performed data preprocessing, feature engineering, and RFECV-based feature selection.
- Tuned models with cross-validation and evaluated them on imbalanced data.
- Gained experience with model comparison and metric-based selection (AUC-ROC, Recall, F2).

👥 This project was completed in collaboration with: Shira Aronovich, Orin Cohen, Shir Greif  













