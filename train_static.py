import argparse
import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from config import MODEL_DIR, STATIC_MODEL_PATH
from data_loader import load_static_dataset

def main():
    parser = argparse.ArgumentParser(description="Train Tier 1 Static Sign Classifier")
    parser.add_argument('--dataset', type=str, required=True, help="Path to static dataset directory or file")
    args = parser.parse_args()

    print(f"[INFO] Loading static dataset from: {args.dataset}")
    X, y = load_static_dataset(args.dataset)
    
    if len(X) == 0:
        print("\n[ERROR] No valid hand landmarks were extracted from the dataset!")
        print("Please check image visibility and hand landmark detection.")
        return

    print(f"\n[INFO] Dataset prepared successfully!")
    print(f" Total Samples Extracted: {len(X)}")
    print(f" Feature Shape per Sample: {X.shape[1] if len(X)>0 else 0}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print("\n[INFO] Training Random Forest Classifier...")
    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)

    print("\n--- Tier 1 Static Evaluation Results ---")
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred))

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(clf, STATIC_MODEL_PATH)
    print(f"\n[SUCCESS] Static model saved successfully to: {STATIC_MODEL_PATH}")

if __name__ == "__main__":
    main()