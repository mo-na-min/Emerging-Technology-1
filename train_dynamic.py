import argparse
import os
import json
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from config import MODEL_DIR, DYNAMIC_MODEL_PATH, DYNAMIC_VOCAB_PATH, SEQUENCE_LENGTH
from data_loader import load_dynamic_dataset
from models import DynamicGRUClassifier

def main():
    parser = argparse.ArgumentParser(description="Train Tier 2 Dynamic Word Classifier")
    parser.add_argument('--dataset', type=str, required=True, help="Path to dynamic dataset directory")
    parser.add_argument('--epochs', type=int, default=40, help="Number of training epochs")
    args = parser.parse_args()

    print(f"[INFO] Loading dynamic dataset from: {args.dataset}")
    X, y_labels = load_dynamic_dataset(args.dataset, target_seq_len=SEQUENCE_LENGTH)
    
    if len(X) == 0:
        print("\n[ERROR] No valid landmark sequences were loaded! Check folder paths and files.")
        return

    unique_labels = sorted(list(set(y_labels)))
    label_to_idx = {label: i for i, label in enumerate(unique_labels)}
    y_idx = np.array([label_to_idx[l] for l in y_labels])

    print(f"\n[DATA AUDIT] Total Sequences={len(X)} | Target Matrix Shape={X.shape} | Classes={len(unique_labels)}")
    for l in unique_labels:
        print(f" - Class '{l}': {np.sum(y_labels == l)} samples")

    X_train, X_test, y_train, y_test = train_test_split(X, y_idx, test_size=0.2, random_state=42, stratify=y_idx)

    train_ds = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    test_ds = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))
    
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False)

    model = DynamicGRUClassifier(input_size=63, hidden_size=64, num_layers=2, num_classes=len(unique_labels))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    print("\n[INFO] Training GRU Network...")
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for bx, by in train_loader:
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{args.epochs} | Loss: {total_loss/len(train_loader):.4f}")

    # Evaluation
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for bx, by in test_loader:
            out = model(bx)
            p = torch.argmax(out, dim=1)
            preds.extend(p.cpu().numpy())
            targets.extend(by.cpu().numpy())

    print("\n--- Tier 2 Dynamic Evaluation Results ---")
    print(classification_report(targets, preds, target_names=unique_labels))

    os.makedirs(MODEL_DIR, exist_ok=True)
    torch.save(model.state_dict(), DYNAMIC_MODEL_PATH)
    with open(DYNAMIC_VOCAB_PATH, 'w') as f:
        json.dump(unique_labels, f)
        
    print(f"\n[SUCCESS] Dynamic model saved to: {DYNAMIC_MODEL_PATH}")
    print(f"[SUCCESS] Dynamic vocab saved to: {DYNAMIC_VOCAB_PATH}")

if __name__ == "__main__":
    main()