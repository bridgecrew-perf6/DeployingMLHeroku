# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

A Random Forest Classifier to predict a person's income from demographic data. 

## Model Details
A Random Forest Classifier imported from the scikit-learn ensembles model library. The model parameter "n_estimators" is set to 100, all other parameters are held at the default values. 

## Intended Use
This model is used to predict an income range for a person based on their demographics. This information can be used along with other information to infer if the person is suitable for lending or likely to default on payments.

## Training Data
The training data is the UCI census data. This data is split and 80% is used for training.

## Evaluation Data
The evaluation data for the model is the remaining 20% of the UCI census data. 

## Metrics
The metrics for model performance and their values are:
- Precision - 0.7352721849366145
- Recall - 0.6248415716096325
- F-beta score - 0.6755738266529633

## Ethical Considerations
The UCI must be acknowledged for acquiring the census data used to build this model, and all subsequent use of the model should reference and acknowledge the UCI. 

## Caveats and Recommendations
The data should be cleaned appropriately before use, the data for this model was pre-treated by removing whitespace from the raw dataset. 