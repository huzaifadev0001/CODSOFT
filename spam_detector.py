import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

def load_dataset():
    """
    Attempts to look for common SMS spam dataset files in the current folder.
    Modify 'file_name' if your downloaded file has a different name.
    """
    
    file_name = "spam.csv" 
    
    if not os.path.exists(file_name):
        raise FileNotFoundError(f"Could not find '{file_name}' in the current folder. Please verify the filename.")

    print(f"--- Loading dataset: {file_name} ---")
    
    
    for encoding in ['utf-8', 'latin-1', 'cp1252']:
        try:
          
            df = pd.read_csv(file_name, encoding=encoding)
            break
        except UnicodeDecodeError:
            continue
            
    # Clean up columns if the dataset has extra unnamed ones (common in SMS spam csv files)
    df = df.dropna(axis=1, how='all')
    
    
    df.columns = ['label', 'text'] + list(df.columns[2:])
    return df[['label', 'text']]

def main():
    try:
       
        df = load_dataset()
        print(f"Dataset loaded successfully! Total records: {len(df)}")
        print("\nSample Data:")
        print(df.head())
        
       
        X = df['text']
        y = df['label']
        
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
       
        print("\n--- Vectorizing Text Data ---")
        vectorizer = TfidfVectorizer(stop_words='english', lowercase=True)
        X_train_tfidf = vectorizer.fit_transform(X_train)
        X_test_tfidf = vectorizer.transform(X_test)
        
        
        print("--- Training Logistic Regression Model ---")
        model = LogisticRegression(solver='liblinear')
        model.fit(X_train_tfidf, y_train)
        
        
        y_pred = model.predict(X_test_tfidf)
        
        accuracy = accuracy_score(y_test, y_pred)
        print("\n================ MODEL PERFORMANCE ================")
        print(f"Accuracy Score: {accuracy * 100:.2f}%")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        print("====================================================")
        
        
        print("\n--- Testing Custom Inputs ---")
        custom_messages = [
            "Hey, are we still meeting up for dinner tonight?",
            "CONGRATULATIONS! You have won a guaranteed £1000 cash prize. Call 09011110000 NOW to claim!"
        ]
        
        custom_tfidf = vectorizer.transform(custom_messages)
        predictions = model.predict(custom_tfidf)
        
        for msg, pred in zip(custom_messages, predictions):
            print(f"Message: '{msg}' --> Predicted Category: **{pred.upper()}**")
            
    except Exception as e:
        print(f"\n[ERROR] An error occurred: {e}")
        print("Tip: Make sure your downloaded dataset is in the same folder as this script and named correctly.")

if __name__ == "__main__":
    main()