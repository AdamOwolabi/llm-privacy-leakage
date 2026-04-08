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
    def __init__(self, input_dim, output_dim):
        super(EmbeddingbasedObserver, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, output_dim)
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.fc2(x)
    

# Training loop for observer embedding model, takes dataloader and optional run name for saving results to a directory with that name
def training_loop(dataloader, run_name='run') :

    num_epochs = 50
    input_dim = dataloader.dataset[0][0].shape[0] # Get input dimension from first sample
    output_dim = len(set([label.item() for _, label in dataloader.dataset])) # Get number of unique classes from dataset labels

    model = EmbeddingbasedObserver(input_dim, output_dim).to(device)
    loss_fcn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    losses = [] # For plotting loss curve
    accuracies = [] # For plotting accuracy curve

    for epoch in range(num_epochs) :
        
        model.train()
        
        running_loss = 0.0
        epoch_loss = 0.0
        
        running_accuracy = 0.0
        epoch_accuracy = 0.0
        
        for data, labels in dataloader :
            
            data = data.to(device)
            labels = labels.to(device)
            
            # Forward pass
            output = model(data)
            loss = loss_fcn(output, labels)
            
            # Zero gradients, backward pass, and update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * data.size(0) # Multiply by batch size to get total loss for the batch
            running_accuracy += (output.argmax(dim=1) == labels).sum().item() # Count correct predictions in the batch
            
        # Divide by batch size to get avg for epoch
        epoch_loss = running_loss / len(dataloader.dataset)
        epoch_accuracy = running_accuracy / len(dataloader.dataset)
        
        losses.append(epoch_loss)
        accuracies.append(epoch_accuracy)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")
                
    # Plot loss curve
    
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
    print(f"Saved accuracy curve: {acc_path}")
    
    # return the trained model so caller can evaluate it
    return model


# Load and split dataset, create dataloader, and call training loop
from torch.utils.data import DataLoader, TensorDataset
def load_and_train(embeddings_path, labels_path, metadata_path, run_name='run') :
    
    embeddings = torch.tensor(np.load(embeddings_path), dtype=torch.float32)
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
    
    # Encode to integer labels for stratification
    le = LabelEncoder()
    
    # TEMP: ensure at least 2 items per combined-stratum for small-test runs
    _force_min_per_stratum = True  # set False / remove this block for real runs
    _min_required = 2
    if _force_min_per_stratum:
        counts = Counter(stratify_keys)
        added = 0
        for key, cnt in list(counts.items()):
            while counts[key] < _min_required:
                # find an index with this key and duplicate that entry
                idx = next(i for i, k in enumerate(stratify_keys) if k == key)
                dataset.append(dataset[idx])
                stratify_keys.append(key)
                counts[key] += 1
                added += 1
        if added:
            print(f"TEMP DUPLICATION: added {added} samples to satisfy min {_min_required} per stratum.")
    # Recompute stratify labels after any temporary duplication so lengths match
    stratify_labels = le.fit_transform(stratify_keys)

    train_dataset, test_dataset = sklearn.model_selection.train_test_split(dataset, test_size=test_size, train_size=train_size, random_state=42, shuffle=True, stratify=stratify_labels)
    
    # Convert to TensorDatasets
    train_embeddings = torch.stack([x[0] for x in train_dataset])
    train_labels = torch.stack([x[1] for x in train_dataset])
    train_dataset = TensorDataset(train_embeddings, train_labels)
    
    test_embeddings = torch.stack([x[0] for x in test_dataset])
    test_labels = torch.stack([x[1] for x in test_dataset])
    test_dataset = TensorDataset(test_embeddings, test_labels)
    
    # Initialize dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    model = training_loop(train_dataloader, run_name)
    
    return model

# Can add fcns for evaluation here later


if __name__ == "__main__":

    # Ran embed_dataset separately, but can also call it here to not have to run two separate scripts
    # Commented out for testing
    # import embed_dataset
    # embed_dataset.load_and_embed('synthetic_conversations.csv', out_dir='embeddings', batch_size=32)
    
    # Train model and save it to a file
    model = load_and_train('embeddings/embeddings.npy', 'embeddings/labels.npy', 'embeddings/metadata.json', run_name='observer_embedding_model')
    model_path = 'observer_embedding_model.pth'
    torch.save(model.state_dict(), model_path)
    print(f"Saved trained model: {model_path}")