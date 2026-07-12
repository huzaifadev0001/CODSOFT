import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

def main():
    try:
        file_name = "fraudTest.csv" 
        
        if not os.path.exists(file_name):
            print(f"\n[STATUS] ready!")
            return
        
        print(f"--- Loading dataset: {file_name} ---")
        df = pd.read_csv(file_name)
        print(f"Dataset loaded! Total rows: {df.shape[0]}")
        
        y = df.iloc[:, -1]
        X = df.iloc[:, :-1]
        X = X.select_dtypes(include=['number'])
        
        print(f"Training features selected: {list(X.columns)}")
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        print("--- Training Model (It will take around 30-40 seconds)... ---")
        model = RandomForestClassifier(n_estimators=20, max_depth=10, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        print("\n================ MODEL PERFORMANCE ================")
        print(f"Accuracy Score: {accuracy_score(y_test, y_pred) * 100:.2f}%")
        print("\nClassification Report:\n", classification_report(y_test, y_pred))
        print("====================================================")
        
    except Exception as e:
        print(f"\nError occurred: {e}")

if __name__ == "__main__":
    main()