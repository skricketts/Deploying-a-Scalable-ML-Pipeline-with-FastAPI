# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

- Model used to predict whether a person earns more than $50k per year, as based on U.S. Census data. 
- Logistic Regression model using scikit-learn.
- Data preprocessing: OneHotEncoder for categorical features and LabelBinarizer for target. 
- Deployed via FastAPI for inference.

## Intended Use

- The intended use of this model is strictly for educational-use only. 
- It demonstrates an end-to-end ML pipeline, including training, CI/CD, deployment, and API inference.
- It is not intended for real-world decision making.

## Training Data

- The model uses a U.S. Census Bureau Adult Income dataset (data/census.gov).
- ~32k rows, including demographic and employement features (age, educations, occupation, etc.).
- Label: salary >$50k (positive) or less than or equal to $50k (negative). 
- 80/20 stratified train/test split.

## Evaluation Data

- Model contains 20% test split from the same dataset.
- This is used for computing overall metrics and slice-based evaluations by categorical feature.

## Metrics
- Precision: 0.7088
- Recall: 0.2717 
- F1-Score: 0.3928
- Evaluated on categorical slices
- Consistent with moderate precision and low recall.

## Ethical Considerations

- This model consists of several sensitive attributes (sex, race, marital status, and native country).
- There is a risk of bias or discrimination if used in real decision systems.

## Caveats and Recommendations

- This is a simple baseline model that has not been optimized for accuracy or fairness. 
- It could be improved with feature scaling, more advanced models, or bias mitigation.
- The small data slices yield unstable metrics and should be interpreted cautiously.