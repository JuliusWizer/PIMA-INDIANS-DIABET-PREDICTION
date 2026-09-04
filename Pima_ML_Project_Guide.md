# Pima Indians Diabetes — Complete ML Project

## Files
- `Pima_Indians_Diabetes_Complete_ML_Solution.ipynb` — full notebook with explanations, EDA, preprocessing, five models, CV, tuning, evaluation, error analysis, interpretation, saving and deployment.
- `pima_diabetes_ml_solution.py` — standalone Python version.
- `Pima_ML_Project_Guide.md` — project workflow and answers to the 23 investigation questions.

## How to run
1. Download the assigned Kaggle file `pima_Missing_values.csv`.
2. Keep the original file unchanged.
3. Put it in the same folder as the notebook/script.
4. Open the notebook in Jupyter or VS Code.
5. Run the cells from top to bottom.
6. Install packages if needed:
   `pip install pandas numpy matplotlib seaborn scikit-learn joblib`

## Important methodological decisions
- Outcome is the binary target.
- Zero is treated as missing only for Glucose, BloodPressure, SkinThickness, Insulin and BMI.
- Pregnancies=0 and Outcome=0 are valid values.
- Median and KNN imputation are compared.
- Preprocessing is inside pipelines to prevent data leakage.
- Five models are evaluated.
- The test set is used only for final evaluation.
- Recall is emphasized because false negatives are important in screening.

## Academic/clinical limitation
This is an academic predictive-modeling exercise. It is not a medical diagnostic device and should not be used to make clinical decisions without appropriate external validation and professional oversight.
