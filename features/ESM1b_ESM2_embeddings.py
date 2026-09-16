import argparse
import pathlib
import pickle
import torch
from esm import FastaBatchedDataset, pretrained

parser = argparse.ArgumentParser(description="Generate ESM1b/ESM2 protein embeddings")
parser.add_argument("-i", "--input", required=True, type=pathlib.Path, help="Input protein FASTA file")
parser.add_argument("-o", "--output", required=True, type=pathlib.Path, help="Output pickle file")
parser.add_argument("-d", "--device", default=0, type=int, help="CUDA device id (default: 0)")
parser.add_argument("-l", "--truncation_seq_length", type=int, default=1022, help="truncate sequences longer than the given value")
args = parser.parse_args()

# Load model
model_name = "esm1b_t33_650M_UR50S"   #ESM1b
# model_name = "esm2_t33_650M_UR50D"  #ESM2
model, alphabet = pretrained.load_model_and_alphabet(model_name)
model.eval()
print("Model:", model_name)

# Select GPU / CPU
if torch.cuda.is_available():
    device = torch.device(f"cuda:{args.device}")
    model = model.to(device)
    print("Using GPU:", device)
else:
    device = torch.device("cpu")
    print("Using CPU")

# Load FASTA
dataset = FastaBatchedDataset.from_file(args.input)
batches = dataset.get_batch_indices(4096, extra_toks_per_seq=1)

data_loader = torch.utils.data.DataLoader(dataset, collate_fn=alphabet.get_batch_converter(args.truncation_seq_length),batch_sampler=batches)
print(f"Read {args.input} with {len(dataset)} sequences")

# Extract mean representation
esm_mean_features = {}

with torch.no_grad():

    for batch_idx, (labels, strs, toks) in enumerate(data_loader):
        
        print(f"Processing {batch_idx + 1} of {len(batches)} batches ({toks.size(0)} sequences)")
        toks = toks.to(device, non_blocking=True)

        # Extract the final layer
        out = model(toks,repr_layers=[model.num_layers])
        representations = out["representations"]

        for i, label in enumerate(labels):
            truncate_len = min(args.truncation_seq_length, len(strs[i]))
            # Remove BOS token and calculate mean
            mean_representation = (representations[model.num_layers][i, 1:truncate_len + 1].mean(0).cpu().numpy())
            esm_mean_features[label] = mean_representation

# Save embeddings
with open(args.output, "wb") as f:
    pickle.dump(esm_mean_features, f)

print("Finished.")
print("Number of embeddings:", len(esm_mean_features))

if len(esm_mean_features) > 0:
    print("Embedding dimension:", next(iter(esm_mean_features.values())).shape[0])

print("Saved to:", args.output)


