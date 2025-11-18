# LoanGuard — Bank Loan Default Predictor

A reproducible machine learning pipeline for predicting loan defaults using tabular credit data. The project includes data cleaning, feature engineering, model training, evaluation, and example code to export predictions. Built as a Jupyter notebook and easily extensible to scripts or a REST API.

##  Features
- Data preprocessing and missing value handling
- Categorical encoding and feature engineering
- Model training with scikit-learn (e.g., Logistic Regression, Random Forest, XGBoost)
- Model evaluation: accuracy, precision, recall, F1, ROC-AUC, confusion matrix
- Simple inference/export of predicted probabilities and risk labels
- Notebook walkthrough with visualizations (feature importance, ROC curve)


##  Dataset (expected format)
The notebook expects a CSV with one row per applicant. Typical columns:
- `loan_id` (optional) — unique id  
- `loan_amount`  
- `term`  
- `interest_rate`  
- `installment`  
- `grade` / `sub_grade` (categorical)  
- `employment_length`  
- `home_ownership`  
- `annual_income`  
- `purpose`  
- `dti` (debt-to-income)  
- `delinquent_2yrs`  
- `open_accounts`  
- `revol_util`  
- `total_acc`  
- `target` (0 = paid / 1 = default) — required for supervised training

