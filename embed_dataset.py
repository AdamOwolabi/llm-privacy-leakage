# Iterate through dataset (csv) -- embed responses in batches using sentence transformers (chunk and aggregate for long responses) -- add embeddings and labels to .npy files for faster loading when training observer embedding model
# Label = hidden trait indices (0 = control, 1 = low sensitivity, 2 = med sensitivity, 3 = high sensitivity)

from sentence_transformers import SentenceTransformer
import torch
import importlib

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

model = SentenceTransformer('all-MiniLM-L6-v2')


def chunk_and_aggregate(text) :
    # Use HuggingFace tokenizer to split by tokens -- chunks should overlap a little bit, but not too much (maybe 30 token overlap?)
    tokens = model.tokenize(text) # returns dict, NOT list of tokens
    token_list = tokens['input_ids'][0].tolist() # get list of token ids from batch of 1
    
    # Max length of tokens for this model
    max_length = model.get_max_seq_length()
    
    chunks = [token_list[i:i+max_length] for i in range(0, len(token_list), max_length - 30)]  # 30 token overlap

    # Get embeddings for each chunk and average them
    chunk_embeddings = model.encode(chunks, device=device)
    aggregated_embedding = chunk_embeddings.mean(axis=0)
    
    return aggregated_embedding


def embed_batch(batch):
    embeddings = []
    labels = []
    metadata = [] # trait and num_turns -- for doing stratified split later in the observer logic

    # DataLoader collates batch into a tuple (texts_list, labels_list, ks_list) instead of list of tuples
    texts, labs, ks = batch
    for response, label, k in zip(texts, labs, ks):
        label_val = int(label.item()) if isinstance(label, torch.Tensor) else int(label)
        embedding = chunk_and_aggregate(response)
        embeddings.append(embedding)
        labels.append(label_val)
        metadata.append({'trait': label_val, 'num_turns': int(k)})
    
    # Stack into single ndarray first before converting to tensor, since list of np arrays --> tensor is super slow apparantly
    batch_embeddings = np.vstack(embeddings)
    batch_labels = np.array(labels, dtype=int)
    return torch.from_numpy(batch_embeddings), torch.from_numpy(batch_labels), metadata


# Embed all responses in the dataset and save to .npy files
import numpy as np
import os
def embed_dataset(dataloader, out_dir='embeddings') :
    os.makedirs(out_dir, exist_ok=True)
    
    all_embeddings = []
    all_labels = []
    metadata = []
    
    for batch in dataloader:
        batch_embeddings, batch_labels, batch_metadata = embed_batch(batch)
        all_embeddings.append(batch_embeddings)
        all_labels.append(batch_labels)
        metadata.append(batch_metadata)
        
    all_embeddings = torch.cat(all_embeddings).numpy()
    all_labels = torch.cat(all_labels).numpy()
    metadata = [item for sublist in metadata for item in sublist]
    
    np.save(os.path.join(out_dir, 'embeddings.npy'), all_embeddings)
    np.save(os.path.join(out_dir, 'labels.npy'), all_labels)
    
    # Save metadata as JSON
    import json
    
    # Metadata is a list of tensors, convert to list of dicts first
    metadata = [{'trait': int(m['trait']), 'num_turns': int(m['num_turns'])} for m in metadata]
    
    with open(os.path.join(out_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f)
    
    
# Load responses from csv, concatenate responses for each convo length, and embed
import pandas as pd
def load_and_embed(csv_path, out_dir='embeddings', batch_size=32):
   
    df = pd.read_csv(csv_path, dtype=str).fillna("")

    # Get llm output columns (e.g. llm_output_0 .. llm_output_4)
    output_cols = sorted([c for c in df.columns if c.startswith('llm_output_')],
                         key=lambda s: int(s.split('_')[-1]) if s.split('_')[-1].isdigit() else s)

    items = []  # list of (concat_text, label)

    for _, row in df.iterrows():
        # Try trait_selected, fallback to trait or '0'
        trait = row.get('trait_selected', row.get('trait', '0'))
        try:
            label = int(trait) if trait != '' else 0
        except Exception:
            label = 0

        outputs = []
        for col in output_cols:
            txt = row.get(col, "")
            if txt is None:
                txt = ""
            txt = str(txt).strip()
            if txt != "":
                outputs.append(txt)

        if len(outputs) == 0:
            continue

        # create concatenated texts for each convo length, include num_turns in each item for stratified splitting later
        for k in range(1, len(outputs) + 1):
            concat = "\n\n".join(outputs[:k])
            items.append((concat, label, k))  # (text, label, num_turns)

    # Initialize dataloader and embed dataset
    dataloader = torch.utils.data.DataLoader(items, batch_size=batch_size, shuffle=False)
    embed_dataset(dataloader, out_dir)


# Test data run:
load_and_embed('synthetic_conversations.csv', out_dir='embeddings', batch_size=32)