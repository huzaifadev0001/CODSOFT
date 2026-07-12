import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

def main():
    try:
        file_name = "Churn_Modelling.csv" 
        
        if not os.path.exists(file_name):
            print(f"[ERROR] Source file '{file_name}' not found in the working directory.")
            return
        
        print(f"Loading dataset: {file_name}")
        df = pd.read_csv(file_name)
        print(f"Dataset loaded successfully. Total records: {df.shape[0]}")
        
        y = df.iloc[:, -1]
        X = df.iloc[:, :-1]
        X = X.select_dtypes(include=['number'])
        
        if 'RowNumber' in X.columns: X = X.drop(columns=['RowNumber'])
        if 'CustomerId' in X.columns: X = X.drop(columns=['CustomerId'])
        
        print(f"Selected training features: {list(X.columns)}")
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        print("Training Random Forest Classifier...")
        model = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        print("\n================ MODEL PERFORMANCE ================")
        print(f"Accuracy Score: {accuracy_score(y_test, y_pred) * 100:.2f}%")
        print("\nClassification Report:\n", classification_report(y_test, y_pred))
        print("====================================================")
        
    except Exception as e:
        print(f"\nExecution failed: {e}")

if __name__ == "__main__":
    main()