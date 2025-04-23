# Titanic-survival-prediction
This project is a machine learning analysis and prediction model to determine which passengers survived the Titanic disaster using a Random Forest Classifier. The dataset is sourced from the popular Titanic dataset on [Kaggle](https://www.kaggle.com/c/titanic).

## Overview
The goal is to predict whether a passenger survived the Titanic disaster using features such as:
- Age
- Gender
- Passenger Class (Pclass)
- Fare
- Cabin
- Embarked
- SibSp, Parch, etc.

## Dataset
The dataset contains the following columns:
- *PassengerId*: Unique identifier
- *Survived*: Target variable (0 = No, 1 = Yes)
- *Pclass*: Ticket class (1st, 2nd, 3rd)
- *Name, Sex, Age*
- *SibSp, Parch*: Family relations aboard
- *Ticket, Fare, Cabin, Embarked*

## Preprocessing
The preprocessing pipeline includes:
- Handling missing values:
  - Imputed Age and Embarked with median/mode
  - Dropped Cabin due to excessive missing data
- Encoding categorical features:
  - Used Label Encoding and One-Hot Encoding for Sex, Embarked, etc.
- Feature scaling:
  - Normalized numerical columns using StandardScaler

## Modeling
- *Algorithm used*: Random Forest Classifier
- Hyperparameter tuning (GridSearchCV or RandomizedSearchCV)
- Trained on 80% of the data and tested on 20%

```python
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
model.fit(X_train, y_train)
