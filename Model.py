import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

try:
    # 1. Dataset load karein
    path = r"C:\Users\dell\OneDrive\Desktop\DATA ANALYST\HEART\heart.csv"
    data = pd.read_csv(path)
    
    print("--- Dataset Loaded Successfully ---")
    print(f"Total Rows: {len(data)}")
    print(data.head())

    # 2. Features aur Target split
    # Make sure your CSV has a 'target' column
    X = data.drop('target', axis=1)
    y = data['target']

    # 3. Training
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("\nTraining Model... Please wait.")
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)

    # 4. Save model
    joblib.dump(model, "rf_model.pkl")
    print("\nSUCCESS: Model trained and saved as 'rf_model.pkl'")

except FileNotFoundError:
    print("ERROR: CSV file nahi mili. Path check karein.")
except KeyError:
    print("ERROR: 'target' naam ka column nahi mila. CSV check karein.")
except Exception as e:
    print(f"An error occurred: {e}")