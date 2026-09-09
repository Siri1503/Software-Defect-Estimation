# Original Project Output Reference

This page records the functionality and results shown in the original project documentation used to reconstruct this repository.

## Output screens represented in the documentation

1. Upload Dataset
2. View Dataset
3. Preprocessing
4. Model trained with CatBoost
5. Model trained with XGBoost
6. Model trained with SVM
7. Model trained with LightGBM
8. Model trained with Random Forest
9. Model trained with Gradient Boosting
10. Model trained with Hybrid Model
11. Model trained with Extra Trees Classifier
12. Predicting Defected Software

The original interface used a dark security-themed layout with navigation controls, dataset upload, preprocessing and model-selection pages, and a manual input screen for software metrics.

## Historical model results

| Model | Accuracy reported in the original output document |
|---|---:|
| CatBoost | ~85% |
| XGBoost | ~81% |
| SVM | ~51% |
| LightGBM | ~85% |
| Random Forest | ~86% |
| Gradient Boosting | ~81% |
| Hybrid Model | ~87.66% |
| Extra Trees Classifier | ~87.12% |

These values are documentation references only. The rebuilt Flask application trains models on the active dataset and calculates fresh evaluation metrics.

## Prediction inputs

The documented prediction screen contains these software metrics:

- loc
- v(g)
- ev(g)
- iv(g)
- l
- d
- i
- e
- IOCode
- IOComment
- IOBlank
- Unique OP
- Unique OPND
- Total OP
- Total OPND
- Branch Count

The rebuilt application uses the same input names so that its workflow stays consistent with the documented project.
