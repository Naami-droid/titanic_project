# Titanic Survival Prediction Project

This project aims to predict the survival of passengers on the Titanic using machine learning.

## Project Structure

```
titanic_project/
├── data/               # Contains the Titanic-Dataset.csv (not included in repo)
├── notebooks/          # Original exploratory data analysis notebook
│   └── titanic_notebook.ipynb
├── src/                # Refactored Python source code
│   └── train.py        # Main script for preprocessing, training, and evaluation
├── README.md           # Project documentation
└── .git/               # Git repository configuration
```

## Features and Preprocessing
- **Title Extraction:** Extracted titles (Mr, Mrs, Miss, etc.) from names to better impute ages and understand social status.
- **Age Imputation:** Missing ages filled using the mean age of the corresponding title group.
- **Cabin Mapping:** Missing cabins filled based on shared ticket numbers and Pclass/Fare quantiles.
- **Feature Engineering:** 
  - `CabinCat`: Extracting the deck level from cabin numbers.
  - `Individual_Fare`: Calculating fare per person for group tickets.
  - `familySize`: Combining SibSp and Parch.
- **Scaling:** Normalization of `Age` and `Individual_Fare` using MinMaxScaler.
- **Encoding:** One-hot encoding for categorical variables.

## Evaluation Metrics
The project now includes comprehensive evaluation metrics:
1. **Accuracy:** The overall percentage of correct predictions.
2. **Confusion Matrix:** Shows True Positives, True Negatives, False Positives, and False Negatives.
3. **Precision:** Ratio of correct positive observations to the total predicted positives.
4. **Recall (Sensitivity):** Ratio of correct positive observations to all observations in actual class.
5. **F1-Score:** The weighted average of Precision and Recall.

## How to Run
1. Ensure you have the required libraries installed:
   ```bash
   pip install pandas numpy scikit-learn
   ```
2. Place `Titanic-Dataset.csv` in the `data/` folder.
3. Run the training script:
   ```bash
   cd src
   python train.py
   ```
