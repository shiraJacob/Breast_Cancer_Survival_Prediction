# Libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from statsmodels.stats.outliers_influence import variance_inflation_factor
from tabulate import tabulate
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold, train_test_split, RandomizedSearchCV
from sklearn.metrics import make_scorer,fbeta_score, roc_auc_score,recall_score
from imblearn.under_sampling import RandomUnderSampler
from itertools import product
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier


# Import data
data = pd.read_csv("Breast_Cancer.csv")
# Check of missing values
print("Number of missing values:")
print(data.isnull().sum().reset_index(name='Missing Values').to_string(index=False))

############################################ Data understanding ############################################

quantitative_data = data[['Age', 'Tumor Size', 'Regional Node Examined', 'Regional Node Positive', 'Survival Months']]

# Descriptive statistics
summary = quantitative_data.describe().round(2)
print(tabulate(summary, headers='keys', tablefmt='pretty'))

# Distributions
fig, axes = plt.subplots(3, 2, figsize=(14, 18), gridspec_kw={'hspace': 0.6, 'wspace': 0.3})
sns.set(color_codes=True)
sns.histplot(data.Age, kde=True, ax=axes[0, 0])
sns.histplot(data['Regional Node Examined'], kde=True, ax=axes[0, 1])
sns.histplot(data['Regional Node Positive'], kde=True, ax=axes[1, 0])
sns.histplot(data['Tumor Size'], kde=True, ax=axes[1, 1])
sns.histplot(data['Survival Months'], kde=True, ax=axes[2, 0])
axes[2, 1].axis('off')
plt.show()

# Quantitative data box-plots with limits of 3IQR
for feature in quantitative_data:
    plt.figure(figsize=(6, 5))
    sns.set_theme(style="white")
    sns.boxplot(data=data, y=feature, color='lightcoral')

    q1 = data[feature].quantile(0.25)
    q3 = data[feature].quantile(0.75)
    iqr = q3 - q1
    upper_bound = q3 + 3 * iqr
    lower_bound = q1 - 3 * iqr

    # Boundaries lines
    plt.axhline(upper_bound, color='lightskyblue', linestyle='--', label='Upper Bound (3*IQR)')
    plt.axhline(lower_bound, color='red', linestyle='--', label='Lower Bound (3*IQR)')

    plt.title(f'Boxplot of {feature}', fontsize=14)
    plt.ylabel(feature, fontsize=12)
    plt.show()


# Boxplots by Status
for feature in quantitative_data:
    plt.figure(figsize=(5, 5))
    sns.boxplot(
        x='Status',
        y=feature,
        hue='Status',
        data=data,
        palette={'Alive': 'lightblue', 'Dead': 'lightcoral'}
    )
    plt.title(f'Boxplot of {feature} by Status')
    plt.legend([], [], frameon=False)
    plt.show()

# Correlation matrix - between continuous variables
corr_matrix = quantitative_data.corr()
formatted_labels = [label.replace(" ", "\n") for label in corr_matrix.columns]
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr_matrix, cmap='coolwarm', annot=True, fmt=".2f", linewidths=.5, ax=ax)
ax.set_xticklabels(formatted_labels, rotation=0, ha='center')
ax.set_yticklabels(formatted_labels, rotation=0, ha='right')
plt.title('Correlation Between Continuous Variables')
plt.show()

# Function that makes bar plots of 2 categorical variables
def barPlots(x_attribute, y_attribute):
    bar_plot_df = data[[x_attribute, y_attribute]]
    cross_tab_prop = pd.crosstab(index=bar_plot_df[x_attribute],
                                 columns=bar_plot_df[y_attribute],
                                 normalize="index")
    num_categories = len(bar_plot_df[y_attribute].unique())
    cmap = plt.cm.get_cmap("Pastel1", num_categories)
    ax = cross_tab_prop.plot(kind='bar', stacked=True, colormap=cmap, figsize=(10, 6))
    plt.legend(loc="upper left", bbox_to_anchor=(1.05, 1), ncol=1, title=y_attribute)
    plt.xticks(rotation=0)
    plt.xlabel(x_attribute)
    plt.ylabel("Proportion")
    plt.title(f"Stacked Bar Plot: {x_attribute} vs {y_attribute}")
    for p in ax.patches:
        width, height = p.get_width(), p.get_height()
        x, y = p.get_xy()
        if height > 0:  # Avoid annotating zero heights
            ax.annotate(f'{height:.2%}', (x + width / 2, y + height / 2), ha='center', va='center')

    plt.tight_layout()
    plt.show()

# Bar plots
barPlots('T Stage ', '6th Stage')
barPlots('N Stage', '6th Stage')
barPlots('A Stage', '6th Stage')
barPlots('A Stage', 'T Stage ')
barPlots('A Stage', 'N Stage')
barPlots('Progesterone Status', 'Estrogen Status')

# Function that counts outliers and representing it in a table (3IQR)
def calculate_outliers_with_status(data, status_column):
    outlier_summary = []

    for column in data.select_dtypes(include=['int64', 'float64']).columns:
        q1 = data[column].quantile(0.25)
        q3 = data[column].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 3 * iqr
        upper_bound = q3 + 3 * iqr

        outliers_lower = data[data[column] < lower_bound]
        outliers_upper = data[data[column] > upper_bound]

        total_outliers = len(outliers_lower) + len(outliers_upper)
        total_count = len(data[column])
        outliers_percentage = (total_outliers / total_count) * 100

        # Calculate outliers by status
        status_dead_lower = outliers_lower[outliers_lower[status_column] == 'Dead'].shape[0]
        status_dead_upper = outliers_upper[outliers_upper[status_column] == 'Dead'].shape[0]
        status_alive_lower = outliers_lower[outliers_lower[status_column] == 'Alive'].shape[0]
        status_alive_upper = outliers_upper[outliers_upper[status_column] == 'Alive'].shape[0]

        dead_count = status_dead_lower + status_dead_upper
        alive_count = status_alive_lower + status_alive_upper

        outlier_summary.append({
            'Feature': column,
            'Dead_Outliers': dead_count,
            'Alive_Outliers': alive_count,
            'Total_Outliers': total_outliers,
            'Outliers_Percentage': f"{round(outliers_percentage, 2)}%",
        })

    return pd.DataFrame(outlier_summary)

# Printing table of outliers
outliers_df = calculate_outliers_with_status(data, status_column='Status')
pd.set_option('display.max_rows', None)  # Show all rows
pd.set_option('display.max_columns', None)  # Show all columns
print(tabulate(outliers_df, headers='keys', tablefmt='pretty'))

# Function that creates Barplots between categorical variables
def barPlots(x_attribute, y_attribute):
    bar_plot_df = data[[x_attribute, y_attribute]]
    cross_tab_prop = pd.crosstab(index=bar_plot_df[x_attribute],
                                 columns=bar_plot_df[y_attribute],
                                 normalize="index")
    num_categories = len(bar_plot_df[y_attribute].unique())
    cmap = plt.cm.get_cmap("Pastel1", num_categories)
    ax = cross_tab_prop.plot(kind='bar', stacked=True, colormap=cmap, figsize=(10, 6))
    plt.legend(loc="upper left", bbox_to_anchor=(1.05, 1), ncol=1, title=y_attribute)
    plt.xticks(rotation=0)
    plt.xlabel(x_attribute)
    plt.ylabel("Proportion")
    plt.title(f"Stacked Bar Plot: {x_attribute} vs {y_attribute}")
    for p in ax.patches:
        width, height = p.get_width(), p.get_height()
        x, y = p.get_xy()
        if height > 0:
            ax.annotate(f'{height:.2%}', (x + width / 2, y + height / 2), ha='center', va='center')

    plt.show()

# VIF Score- Multicollinearity check for continuous variables
X = quantitative_data
X_with_constant = pd.concat([pd.Series(1, index=X.index, name='const'), X], axis=1)
vif_data = pd.DataFrame()
vif_data["Variable"] = X_with_constant.columns
vif_data["VIF"] = [variance_inflation_factor(X_with_constant.values, i) for i in range(X_with_constant.shape[1])]
vif_data = vif_data[vif_data['Variable'] != 'const']
print("Variance Inflation Factor (VIF):")
print(vif_data)


############################################ Data preparation ############################################

## Outliers handling + DownSampling on the data

#Outliers handling- Winsorization - 3*IQR
# Function to calculate boundaries for all columns at once
def calculate_boundaries(data, columns, multiplier=3):
    boundaries = {}
    for column in columns:
        q1 = data[column].quantile(0.25)
        q3 = data[column].quantile(0.75)
        iqr = q3 - q1
        boundaries[column] = {
            'lower_bound': q1 - multiplier * iqr,
            'upper_bound': q3 + multiplier * iqr
        }
    return boundaries

# Function to calculate outliers
def calculate_outliers(data, boundaries):
    outlier_summary = []
    for column, bounds in boundaries.items():
        outliers_lower = (data[column] < bounds['lower_bound']).sum()
        outliers_upper = (data[column] > bounds['upper_bound']).sum()
        total_outliers = outliers_lower + outliers_upper
        total_count = len(data[column])
        outlier_percentage = (total_outliers / total_count) * 100 if total_count > 0 else 0

        outlier_summary.append({
            'Feature': column,
            'Outliers_Lower': outliers_lower,
            'Outliers_Upper': outliers_upper,
            'Total_Outliers': total_outliers,
            'Outliers_Percentage': f"{round(outlier_percentage, 2)}%"
        })
    return pd.DataFrame(outlier_summary)

# Function to adjust outliers - 3IQR
def adjust_outliers(data, boundaries):
    for column, bounds in boundaries.items():
        data.loc[data[column] < bounds['lower_bound'], column] = bounds['lower_bound']
        data.loc[data[column] > bounds['upper_bound'], column] = bounds['upper_bound']

columns_to_adjust = ['Regional Node Examined', 'Regional Node Positive', 'Tumor Size']
boundaries = calculate_boundaries(data, columns_to_adjust, multiplier=3)
adjust_outliers(data, boundaries)


# Balancing - DownSampling
X = data.drop(columns=['Status'])  # Features
y = data['Status']  # Target variable
RS = RandomUnderSampler(sampling_strategy=0.5, random_state=42)
x_rs, y_rs = RS.fit_resample(X, y)
data = pd.concat([x_rs, y_rs], axis=1)
print("Class distribution after downSampling:")
print(pd.Series(y_rs).value_counts())

# Converting categorical variables - oneHot encoding & binary variables
categorical_data_D = data[['Race', 'Marital Status']]
categorical_data = data[['T Stage ', 'N Stage', '6th Stage', 'differentiate', 'Grade']]
encoder = OneHotEncoder(drop=None, sparse_output=False)
encoded_array = encoder.fit_transform(categorical_data_D)
encoded_columns = encoder.get_feature_names_out(categorical_data_D.columns)
encoded_df = pd.DataFrame(encoded_array, columns=encoded_columns, index=data.index).astype(int)
data = pd.concat([data, encoded_df], axis=1)

# Custom mappings for specific columns
custom_mappings = {
    '6th Stage': {'IIA': 0, 'IIB': 1, 'IIIA': 2, 'IIIB': 3, 'IIIC': 4},
    'differentiate': {'Undifferentiated': 0, 'Poorly differentiated': 1,
                      'Moderately differentiated': 2, 'Well differentiated': 3},
    'Grade': {'1': 0, '2': 1, '3': 2, ' anaplastic; Grade IV': 3}
}
for col, mapping in custom_mappings.items():
    data[col + '_Mapped'] = data[col].map(mapping)
for col in categorical_data:
    if col not in custom_mappings:
        mapping = {category: idx for idx, category in enumerate(data[col].unique())}
        print(f"Mapping for {col}: {mapping}")

data['Estrogen Status'] = data['Estrogen Status'].map({'Negative': 0, 'Positive': 1})
data['Progesterone Status'] = data['Progesterone Status'].map({'Negative': 0, 'Positive': 1})
data['A Stage'] = data['A Stage'].map({'Regional': 0, 'Distant': 1})
data['Status'] = data['Status'].map({'Alive': 0, 'Dead': 1})


# Normalization - quantitative data
data['Age_Normalized'] = (data['Age'] - data['Age'].min()) / (data['Age'].max() - data['Age'].min())
data['Tumor_Size_Normalized'] = (data['Tumor Size'] - data['Tumor Size'].min()) / (data['Tumor Size'].max() - data['Tumor Size'].min())
data['Regional_Node_Examined_Normalized'] = (data['Regional Node Examined'] - data['Regional Node Examined'].min()) / (data['Regional Node Examined'].max() - data['Regional Node Examined'].min())
data['Regional_Node_Positive_Normalized'] = (data['Regional Node Positive'] - data['Regional Node Positive'].min()) / (data['Regional Node Positive'].max() - data['Regional Node Positive'].min())
data['Survival_Months_Normalized'] = (data['Survival Months'] - data['Survival Months'].min()) / (data['Survival Months'].max() - data['Survival Months'].min())


# Discretization - Age
bins_original = [30, 46, 60, data['Age'].max() + 1]
labels = [0, 1, 2]  # 0 = Young (30-45), 1 = Middle-aged (46-59), 2 = Elderly (60+)
data['Age_Binned'] = pd.cut(
    data['Age'],
    bins=bins_original,
    labels=labels,
    include_lowest=True
)
data['Age_Binned'] = data['Age_Binned'].astype(int)
age_binned_distribution = data['Age_Binned'].value_counts()
print(age_binned_distribution)

# New feature-Create Regional Node ratio variable
data['Regional_Node_Ratio'] = data['Regional Node Positive'] / data['Regional Node Examined']
data['Regional_Node_Ratio'] = data['Regional_Node_Ratio'].replace([float('inf'), -float('inf')], 0).fillna(0)

# Move the 'Status' to the end of the data Frame
column_to_move = data.pop('Status')
data.insert(len(data.columns), 'Status', column_to_move)

# Drop columns that we've already extracted features out of them to get DF of features only
data.drop('Age', axis=1, inplace=True)
data.drop('Race', axis=1, inplace=True)
data.drop('Marital Status', axis=1, inplace=True)
data.drop('T Stage ', axis=1, inplace=True)
data.drop('N Stage', axis=1, inplace=True)
data.drop('6th Stage', axis=1, inplace=True)
data.drop('differentiate', axis=1, inplace=True)
data.drop('Grade', axis=1, inplace=True)
data.drop('Tumor Size', axis=1, inplace=True)
data.drop('Regional Node Examined', axis=1, inplace=True)
data.drop('Regional Node Positive', axis=1, inplace=True)
data.drop('Survival Months', axis=1, inplace=True)
data.drop('Regional_Node_Examined_Normalized', axis=1, inplace=True)
data.drop('Regional_Node_Positive_Normalized', axis=1, inplace=True)
data.drop('Age_Normalized', axis=1, inplace=True)

############################################ Feature Extraction ############################################

# Feature and target split after encoding
X = data.drop(columns=['Status'])
y = data['Status']

# Perform Recursive Feature Elimination with Cross-Validation
RFECV = RFECV(estimator=RandomForestClassifier(random_state=42),
              step=1,
              cv=StratifiedKFold(5),
              scoring=make_scorer(roc_auc_score))
RFECV.fit(X, y)

selected_features = X.columns[RFECV.support_]
data_selected = data[selected_features]

print(f"Optimal number of features: {RFECV.n_features_}")
print(f"Selected features: {selected_features}")


#Splitting the data 20% test and 80% train on the training data
x_data = data_selected
X_train,X_test,y_train, y_test =train_test_split(x_data, data['Status'],test_size=0.2, random_state=42)


# Function to print scores before continuing to the models' stage
def evaluate_model(model, X_train, y_train, X_test, y_test, model_name):
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    y_train_proba = model.predict_proba(X_train)[:, 1]
    y_test_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "Metric": ["AUC-ROC", "Recall", "F2 Score"],
        "Train Set": [
            roc_auc_score(y_train, y_train_proba),
            recall_score(y_train, y_train_pred),
            fbeta_score(y_train, y_train_pred, beta=2),
        ],
        "Test Set": [
            roc_auc_score(y_test, y_test_proba),
            recall_score(y_test, y_test_pred),
            fbeta_score(y_test, y_test_pred, beta=2),
        ]
    }

    metrics_df = pd.DataFrame(metrics)
    metrics_df.insert(0, "Model Name", model_name)
    return metrics_df

############################################ Random Forest ############################################

# RandomizedSearchCV parameters for Random Forest
param_grid = {
    'n_estimators': [100, 300, 500],
    'max_depth': np.arange(8, 15, 1),
    'criterion': ['gini', 'entropy'],
    'max_features': [3,5,7],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

comb = 1
for list_ in param_grid.values():
    comb *= len(list_)
print("number of Combinations Random Forest:"+str(comb))
param_grid.values()

# RandomizedSearchCV
random_search = RandomizedSearchCV(
    estimator=RandomForestClassifier(random_state=42,class_weight='balanced'),
    param_distributions=param_grid,
    n_iter=60,  # Number of combinations
    scoring='f1',
    cv=10,
    random_state=42,
    n_jobs=-1
)

# Fit the model
random_search.fit(X_train, y_train)
best_modelRF = random_search.best_estimator_
print("After RandomizedSearch, the best Random Forest model is:", best_modelRF)

results_rf = evaluate_model(best_modelRF, X_train, y_train, X_test, y_test, model_name="Random Forest")
print(results_rf)

print("Best parameters:", random_search.best_params_)

# Feature importance
importance = best_modelRF.feature_importances_
feature_importance_df = pd.DataFrame({'Feature': X_train.columns, 'Importance': importance})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
print(feature_importance_df)

plt.figure(figsize=(10, 6))
plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'])
plt.xlabel("Feature Importance")
plt.ylabel("Features")
plt.title("Feature Importance of Random Forest")
plt.gca().invert_yaxis()  # Ensure the most important features are at the top
plt.show()


############################################ XGBoost ############################################

# RandomizedSearchCV parameters for XGBoost
param_grid = {
    'learning_rate': [0.001, 0.002, 0.01, 0.1],
    'n_estimators': [50, 100],
    'max_depth': [1, 2, 3, 4, 5, 6],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'gamma': [0, 1, 5],
    'scale_pos_weight': [1, 2, 3, 5, 10]
}
comb = 1
for list_ in param_grid.values():
    comb *= len(list_)
print("number of Combinations XGBoost:"+str(comb))
param_grid.values()

# RandomizedSearchCV
random_search = RandomizedSearchCV(
    estimator=XGBClassifier(eval_metric='aucpr', random_state=42),
    param_distributions=param_grid,
    n_iter=30,
    scoring='roc_auc',
    cv=10,
    random_state=42,
    n_jobs=-1
)

# Fit the model
random_search.fit(X_train, y_train)
best_modelXGBoost = random_search.best_estimator_
print("After RandomizedSearch, the best XGBoost model is:", best_modelXGBoost)
# Results
results_XGB = evaluate_model(best_modelXGBoost, X_train, y_train, X_test, y_test, model_name="XGBoost")
print(results_XGB)

print("Best parameters:", random_search.best_params_)

# Feature importance
importance_xgb = best_modelXGBoost.feature_importances_
feature_importance_xgb_df = pd.DataFrame({'Feature': X_train.columns, 'Importance': importance_xgb})
feature_importance_xgb_df = feature_importance_xgb_df.sort_values(by='Importance', ascending=False)
print(feature_importance_xgb_df)

plt.figure(figsize=(10, 6))
plt.barh(feature_importance_xgb_df['Feature'], feature_importance_xgb_df['Importance'])
plt.xlabel("Feature Importance")
plt.ylabel("Features")
plt.title("Feature Importance of XGBoost")
plt.gca().invert_yaxis()
plt.show()

############################################ SVM- SVC ############################################

# Apply standardization
columns_to_standardize = ['Age_Binned', '6th Stage_Mapped', 'differentiate_Mapped', 'Grade_Mapped']
# Initialize scaler
scaler = StandardScaler()
X_train[columns_to_standardize] = scaler.fit_transform(X_train[columns_to_standardize])
X_test[columns_to_standardize] = scaler.transform(X_test[columns_to_standardize])

# RandomizedSearchCV parameters for SVM
param_grid = {
    'C': [0.01, 0.1, 1, 10, 100, 1000],
    'kernel': ['linear', 'rbf', 'poly'],
    'gamma': [0.0001,0.001, 0.01, 0.1],
    'tol': [1e-4, 1e-3]
}

combinations = list(product(*param_grid.values()))
num_combinations = len(combinations)
print("Number of Combinations SVC:", num_combinations)

# RandomizedSearchCV
svc_model = svm.SVC(probability=True, random_state=42, class_weight='balanced')
random_search = RandomizedSearchCV(
    estimator=svc_model,
    param_distributions=param_grid,
    n_iter=50,
    scoring='roc_auc',
    cv=10,
    random_state=42,
    n_jobs=-1
)

# Fit the model
random_search.fit(X_train, y_train)
best_model_SVC= random_search.best_estimator_
print("After RandomSearch, the best SVC model is:", best_model_SVC)
# Results
results_SVC = evaluate_model(best_model_SVC, X_train, y_train, X_test, y_test, model_name="SVC")
print(results_SVC)

print("Best parameters:", random_search.best_params_)

############################################ MLP ############################################

# Initialize scaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(x_data)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# RandomizedSearchCV parameters for XGBoost
param_grid = {
    'hidden_layer_sizes': [(100, 50), (150, 100, 50)],
    'activation': ['relu', 'tanh','logistic'],
    'alpha': [0.001, 0.01, 0.1],
    'learning_rate': ['constant', 'adaptive', 'invscaling'],
    'max_iter': [2000, 4000, 8000]
}

combinations = list(product(*param_grid.values()))
num_combinations = len(combinations)
print("Number of Combinations MLP:", num_combinations)

# RandomizedSearchCV
random_search = RandomizedSearchCV(
    estimator=MLPClassifier(random_state=42,early_stopping=True, validation_fraction=0.1,n_iter_no_change=20),
    param_distributions=param_grid,
    n_iter=50,
    scoring='roc_auc',
    cv=10,
    random_state=42,
    n_jobs=-1
)

# Fit the model
random_search.fit(X_train, y_train)
best_modelMLP = random_search.best_estimator_
print("After RandomizedSearch, the best MLP model is:", best_modelMLP)
# Results
results_MLP = evaluate_model(best_modelMLP, X_train, y_train, X_test, y_test, model_name="MLP")
print(results_MLP)

print("Best parameters:", random_search.best_params_)