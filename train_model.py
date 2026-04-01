import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle

try:
    # 1. Data load karein
    df = pd.read_csv('heart.csv', encoding='latin1')

    # 2. Column names ko saaf karein (Extra spaces hatane ke liye)
    df.columns = df.columns.str.strip().str.lower()

    print("Columns in your file:", list(df.columns))

    # 3. Check karein ki 'target' column hai ya nahi
    if 'target' in df.columns:
        X = df.drop('target', axis=1)
        Y = df['target']
    else:
        # Agar 'target' nahi mila, toh last column ko target maan lo
        print("⚠️ 'target' column nahi mila, last column ko output maan rahe hain.")
        X = df.iloc[:, :-1]
        Y = df.iloc[:, -1]

    # 4. Model Training (PPT: Machine Learning Prediction Model)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, Y)

    # Feature names save karna UI ke liye zaroori hai
    model.feature_names_in_ = list(X.columns)

    # 5. Save Model
    with open('heart_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    print("🚀 Success! 'heart_model.pkl' ban chuki hai.")

except Exception as e:
    print(f"❌ Abhi bhi error hai: {e}")