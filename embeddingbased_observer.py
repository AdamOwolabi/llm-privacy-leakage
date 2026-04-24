import importlib
import json
import numpy as np
import sklearn
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import LabelEncoder
from collections import Counter

# Detect best available device: CUDA -> DirectML -> CPU
if torch.cuda.is_available():
    device = "cuda"
else:
    try:
        td = importlib.import_module("torch_directml")
        device = td.device()          # torch_directml.device() works with torch_directml
    except Exception:
        device = "cpu"

print("Using device:", device)


# Class for multiclass classifier observer, uses precomputed sentence embeddings as tensors
class EmbeddingbasedObserver(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_size=128, dropout=0.0):
        super(EmbeddingbasedObserver, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout) if dropout and dropout > 0.0 else None
        self.fc2 = nn.Linear(hidden_size, output_dim)
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        if self.dropout is not None:
            x = self.dropout(x)
        return self.fc2(x)
    

# Training loop for observer embedding model, takes dataloader and optional run name for saving results to a directory with that name
def training_loop(dataloader, run_name='run', num_epochs=50, lr=1e-3, weight_decay=0.0, test_dataloader=None, early_stopping_patience=5, reduce_lr_on_plateau=True, clip_grad_norm=1.0, min_lr=1e-6, class_weights=None, hidden_size=128, dropout=0.0):

    input_dim = dataloader.dataset[0][0].shape[0] # Get input dimension from first sample
    output_dim = len(set([label.item() for _, label in dataloader.dataset])) # Get number of unique classes from dataset labels

    model = EmbeddingbasedObserver(input_dim, output_dim, hidden_size=hidden_size, dropout=dropout).to(device)
    # If class_weights provided (torch tensor on device), use them in CrossEntropyLoss
    if class_weights is not None:
        loss_fcn = nn.CrossEntropyLoss(weight=class_weights)
    else:
        loss_fcn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    scheduler = None
    if reduce_lr_on_plateau and test_dataloader is not None:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, min_lr=min_lr, verbose=True)
    
    losses = [] # For plotting loss curve
    accuracies = [] # For plotting accuracy curve
    val_losses = []
    val_accuracies = []

    best_val_loss = float('inf')
    epochs_since_improve = 0

    for epoch in range(num_epochs) :
        
        model.train()
        
        running_loss = 0.0
        running_accuracy = 0.0

        for data, labels in dataloader :
            
            data = data.to(device)
            labels = labels.to(device)
            
            # Forward pass
            output = model(data)
            loss = loss_fcn(output, labels)
            
            # Zero gradients, backward pass, and update
            optimizer.zero_grad()
            loss.backward()
            if clip_grad_norm and clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
            optimizer.step()

            running_loss += loss.item() * data.size(0) # Multiply by batch size to get total loss for the batch
            running_accuracy += (output.argmax(dim=1) == labels).sum().item() # Count correct predictions in the batch

        # Divide by dataset size to get avg for epoch
        epoch_loss = running_loss / len(dataloader.dataset)
        epoch_accuracy = running_accuracy / len(dataloader.dataset)

        losses.append(epoch_loss)
        accuracies.append(epoch_accuracy)

        # Optionally evaluate on test set
        if test_dataloader is not None:
            model.eval()
            val_running_loss = 0.0
            val_running_correct = 0
            with torch.no_grad():
                for x_val, y_val in test_dataloader:
                    x_val = x_val.to(device)
                    y_val = y_val.to(device)
                    out = model(x_val)
                    vloss = loss_fcn(out, y_val)
                    val_running_loss += vloss.item() * x_val.size(0)
                    val_running_correct += (out.argmax(dim=1) == y_val).sum().item()

            val_loss = val_running_loss / len(test_dataloader.dataset)
            val_acc = val_running_correct / len(test_dataloader.dataset)
            val_losses.append(val_loss)
            val_accuracies.append(val_acc)
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Acc: {epoch_accuracy:.4f}, ValLoss: {val_loss:.4f}, ValAcc: {val_acc:.4f}")
            # scheduler step
            if scheduler is not None:
                scheduler.step(val_loss)
            # early stopping
            if val_loss < best_val_loss - 1e-6:
                best_val_loss = val_loss
                epochs_since_improve = 0
            else:
                epochs_since_improve += 1
                if epochs_since_improve >= early_stopping_patience:
                    print(f"Early stopping triggered (no val improvement for {early_stopping_patience} epochs)")
                    break
        else:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")

    # Make dir to save results to if it doesn't already exist
    out_dir = os.path.join('runs', run_name)
    os.makedirs(out_dir, exist_ok=True)

    plt.figure(figsize=(10, 5))
    # plot using 1-based epoch indices so x-axis shows Epoch 1..N
    plt.plot(range(1, len(losses) + 1), losses, label='Training Loss')
    plt.title('Training Loss Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    loss_path = os.path.join(out_dir, 'loss_curve.png')
    plt.savefig(loss_path)
    plt.close()
    print(f"Saved loss curve: {loss_path}")

    # Plot accuracy curve
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(accuracies) + 1), accuracies, label='Training Accuracy')
    plt.title('Training Accuracy Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    acc_path = os.path.join(out_dir, 'accuracy_curve.png')
    plt.savefig(acc_path)
    plt.close()
    print(f"Saved accuracy curve: {acc_path}")

    train_history = {"train_losses": losses, "train_accuracies": accuracies, "val_losses": val_losses, "val_accuracies": val_accuracies}
    return model, train_history


# Load and split dataset, create dataloader, and call training loop
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
def load_and_train(embeddings_path, labels_path, metadata_path, run_name='run', num_epochs=50, train_batch_size=16, test_batch_size=32, lr=1e-3, weight_decay=0.0, use_class_weights=False, use_weighted_sampler=False, hidden_size=128, dropout=0.0):
    
    embeddings = torch.tensor(np.load(embeddings_path), dtype=torch.float32)
    # Normalize embeddings
    embeddings = torch.nn.functional.normalize(embeddings, dim=1)
    labels = torch.tensor(np.load(labels_path), dtype=torch.long)
    metadata = json.load(open(metadata_path))
    
    # Split dataset into train and test sets (80/20 split)
    # Might have a problem... need to be very careful about how we split the dataset since we have 4 classes and want to make sure all classes are represented in both train and test sets (stratified split)
    # Logic in embed_dataset produces metadata.json with trait and num_turns stratified keys
    
    # Make list of dicts, convert to TensorDataset after stratified split
    dataset = []
    for i in range(len(embeddings)):
        dataset.append((embeddings[i], labels[i], metadata[i]))

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    
    # Combine trait, num_turns into a single key per row
    stratify_keys = [f"{m['trait']}_{m['num_turns']}" for m in metadata]
    # Print counts per stratify key to help diagnose class imbalance
    strat_counts = Counter(stratify_keys)
    print('Stratify key counts (trait_numTurns):')
    for k, v in sorted(strat_counts.items()):
        print(f'  {k}: {v}')
    
    # Encode to integer labels for stratification
    le = LabelEncoder()
    stratify_labels = le.fit_transform(stratify_keys)
    # Print distribution of encoded stratify labels
    enc_counts = Counter(stratify_labels)
    print('Encoded stratify label distribution:')
    for lbl, cnt in sorted(enc_counts.items()):
        print(f'  label {lbl}: {cnt} rows (strat key={le.inverse_transform([lbl])[0]})')

    train_dataset, test_dataset = sklearn.model_selection.train_test_split(dataset, test_size=test_size, train_size=train_size, random_state=42, shuffle=True, stratify=stratify_labels)
    
    # Convert to TensorDatasets
    train_embeddings = torch.stack([x[0] for x in train_dataset])
    train_labels = torch.stack([x[1] for x in train_dataset])
    train_dataset = TensorDataset(train_embeddings, train_labels)
    
    test_embeddings = torch.stack([x[0] for x in test_dataset])
    test_labels = torch.stack([x[1] for x in test_dataset])
    test_dataset = TensorDataset(test_embeddings, test_labels)
    
    # Initialize dataloaders - optionally use a WeightedRandomSampler for oversampling
    if use_weighted_sampler:
        try:
            # class counts -> weight per class -> sample weight per example
            output_dim = len(torch.unique(train_labels).numpy())
            class_counts = np.bincount(train_labels.numpy(), minlength=output_dim)
            sample_weights = 1.0 / class_counts[train_labels.numpy()]
            sample_weights = torch.tensor(sample_weights, dtype=torch.double)
            sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
            train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, sampler=sampler)
            print('Using WeightedRandomSampler for training dataloader')
        except Exception as e:
            print('Could not create WeightedRandomSampler, falling back to shuffle=True:', e)
            train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
    else:
        train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)

    test_dataloader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)
    
    # Compute class weights if requested (balanced by inverse frequency)
    cw_tensor = None
    if use_class_weights:
        try:
            from sklearn.utils.class_weight import compute_class_weight
            # classes are 0..(output_dim-1)
            output_dim = len(torch.unique(train_labels).numpy())
            classes = np.arange(output_dim)
            cw = compute_class_weight(class_weight='balanced', classes=classes, y=train_labels.numpy())
            cw_tensor = torch.tensor(cw, dtype=torch.float32).to(device)
            print(f"Using class weights: {cw}")
        except Exception as e:
            print("Could not compute class weights, continuing without them:", e)

    model, train_history = training_loop(train_dataloader, run_name=run_name, num_epochs=num_epochs, lr=lr, weight_decay=weight_decay, test_dataloader=test_dataloader, early_stopping_patience=5, reduce_lr_on_plateau=True, clip_grad_norm=1.0, class_weights=cw_tensor, hidden_size=hidden_size, dropout=dropout)

    # Minimal test evaluation: print accuracy, classification report, confusion matrix
    def evaluate_model(model, loader, name='Test'):
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for xb, yb in loader:
                xb = xb.to(device)
                yb = yb.to(device)
                out = model(xb)
                preds = out.argmax(dim=1)
                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(yb.cpu().tolist())

        acc = (np.array(all_preds) == np.array(all_labels)).mean()
        print(f"{name} accuracy: {acc:.4f}")
        try:
            from sklearn.metrics import classification_report, confusion_matrix
            print(f"\n{name} Classification report:")
            print(classification_report(all_labels, all_preds))
            print(f"{name} Confusion matrix:")
            print(confusion_matrix(all_labels, all_preds))
        except Exception as e:
            print("Could not compute detailed metrics:", e)
        return {"accuracy": float(acc)}

    # Evaluate on both train and test sets so we can detect overfitting/leakage
    train_metrics = evaluate_model(model, train_dataloader, name='Train')
    test_metrics = evaluate_model(model, test_dataloader, name='Test')
    metrics = {"train_accuracy": train_metrics.get('accuracy'), "accuracy": test_metrics.get('accuracy')}
    # attach final training values
    if train_history and 'train_losses' in train_history and len(train_history['train_losses'])>0:
        metrics['final_train_loss'] = float(train_history['train_losses'][-1])
        metrics['final_train_accuracy'] = float(train_history['train_accuracies'][-1])
    else:
        metrics['final_train_loss'] = None
        metrics['final_train_accuracy'] = None
    return model, metrics


if __name__ == "__main__":

    # Ran embed_dataset separately, but can also call it here to not have to run two separate scripts
    # Commented out for testing
    # import embed_dataset    
    # embed_dataset.load_and_embed('synthetic_conversations_full.csv', out_dir='embeddings', batch_size=32)
    
    # Train model and save it to a file
    
    # Try a bunch of different hyperparameter combinations, printing results
    # Selected experiments (focused): prefer larger hidden size, lr=1e-4, batch=64, and sampler on
    selected_experiments = [
        {"hidden": 256, "bs": 64, "lr": 1e-4, "sampler": True},
        {"hidden": 256, "bs": 64, "lr": 1e-4, "sampler": False},
        {"hidden": 256, "bs": 16, "lr": 1e-4, "sampler": True},
    ]

    # Ensure runs dir exists and results file
    os.makedirs('runs', exist_ok=True)
    results_path = os.path.join('runs', 'experiment_results.txt')

    import io
    import contextlib

    for exp in selected_experiments:
        h = exp['hidden']
        bs = exp['bs']
        lr = exp['lr']
        sampler_flag = exp['sampler']
        lr_str = str(lr).replace('.', 'p').replace('-', 'm')
        run_name = f"observer_h{h}_bs{bs}_lr{lr_str}{'_sampler' if sampler_flag else ''}"
        header = f"=== RUN: {run_name} | hidden={h} bs={bs} lr={lr} sampler={sampler_flag} ===\n"
        print(header)
        # Capture all stdout produced by load_and_train (so classification reports, confusion matrices, prints) into the results file
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            model, metrics = load_and_train(embeddings_path='embeddings/embeddings_lowleakage.npy', labels_path='embeddings/labels_lowleakage.npy', metadata_path='embeddings/metadata_lowleakage.json', run_name=run_name, num_epochs=30, train_batch_size=bs, test_batch_size=32, lr=lr, weight_decay=0.0, use_class_weights=False, use_weighted_sampler=sampler_flag, hidden_size=h, dropout=0.2)
        run_output = buf.getvalue()
        # Append to results file
        with open(results_path, 'a', encoding='utf-8') as rf:
            rf.write(header)
            rf.write(run_output)
            rf.write(f"Metrics: {metrics}\n")
            rf.write("\n---\n\n")
        # Also print a short summary to console
        print(f"Finished {run_name}: accuracy={metrics.get('accuracy')}, final_train_loss={metrics.get('final_train_loss')}\n")

    # model_path = 'observer_embedding_model.pth'
    # torch.save(model.state_dict(), model_path)
    # print(f"Saved trained model: {model_path}")