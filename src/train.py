import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score, recall_score, f1_score
import os

def load_data(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found at {file_path}. Please place Titanic-Dataset.csv in the data/ folder.")
    return pd.read_csv(file_path)

def preprocess_data(df):
    # Extract titles
    df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\\.', expand=False)
    
    # Impute Age based on Title
    df['Age'] = df['Age'].fillna(df.groupby('Title')['Age'].transform('mean'))
    
    # Impute Cabin based on Ticket and Pclass
    ticket_cabin_dict = df.dropna(subset=['Cabin']).set_index('Ticket')['Cabin'].to_dict()
    df['Cabin'] = df['Cabin'].fillna(df['Ticket'].map(ticket_cabin_dict))
    
    # Pclass 2 and 3: Fill missing cabins with 'U'
    df.loc[(df['Pclass'].isin([2, 3])) & (df['Cabin'].isnull()), 'Cabin'] = 'U'
    
    # Pclass 1: Assign cabin letters based on Fare intervals
    pclass1_mask = df['Pclass'] == 1
    if pclass1_mask.any():
        df.loc[pclass1_mask, 'Fare_Bin'] = pd.qcut(df.loc[pclass1_mask, 'Fare'], q=3, labels=['Lower', 'Mid', 'Expensive'])
        bin_to_cabin_map = {'Expensive': 'B', 'Mid': 'C', 'Lower': 'E'}
        pclass1_missing_mask = (df['Pclass'] == 1) & (df['Cabin'].isnull())
        df.loc[pclass1_missing_mask, 'Cabin'] = df.loc[pclass1_missing_mask, 'Fare_Bin'].map(bin_to_cabin_map)
        df = df.drop(columns=['Fare_Bin'])
    
    # Drop instances with no Embarked data
    df = df.dropna(subset=['Embarked'])
    
    # Feature Engineering
    df["CabinCat"] = df['Cabin'].str[0]
    
    # Individual Fare
    ticket_counts = df['Ticket'].value_counts()
    df['Group_Size'] = df['Ticket'].map(ticket_counts)
    df['Individual_Fare'] = df['Fare'] / df['Group_Size']
    
    # Family Size
    df['familySize'] = df['SibSp'] + df['Parch'] + 1
    
    # Encoding
    columns_to_encode = ['CabinCat', 'Embarked', 'Title', 'Group_Size', 'Pclass', 'familySize', 'Gender']
    df = pd.get_dummies(df, columns=columns_to_encode, drop_first=True, dtype=int)
    
    # Scale continuous columns
    scaler = MinMaxScaler()
    columns_to_scale = ['Age', 'Individual_Fare']
    df[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])
    
    # Drop unnecessary columns
    cols_to_drop = ['PassengerId', 'Name', 'Ticket', 'Cabin']
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
    
    return df

def train_and_evaluate(df):
    X = df.drop('Survived', axis=1)
    y = df['Survived'].astype(int)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    
    predictions = model.predict(X_test)
    
    # Metrics
    acc = accuracy_score(y_test, predictions)
    conf_matrix = confusion_matrix(y_test, predictions)
    class_report = classification_report(y_test, predictions)
    
    print(f"Accuracy: {acc:.4f}")
    print("\nConfusion Matrix:")
    print(conf_matrix)
    print("\nClassification Report:")
    print(class_report)
    
    return model

if __name__ == "__main__":
    data_path = os.path.join("..", "data", "Titanic-Dataset.csv")
    try:
        df = load_data(data_path)
        processed_df = preprocess_data(df)
        train_and_evaluate(processed_df)
    except FileNotFoundError as e:
        print(e)
