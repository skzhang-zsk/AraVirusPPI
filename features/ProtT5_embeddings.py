from transformers import T5Tokenizer, T5EncoderModel
import torch
import re
import pickle
import argparse

# Parameters
parser = argparse.ArgumentParser(description="Generate ProtT5 protein embeddings")
parser.add_argument("-i", "--input", required=True, type=str, help="Input protein FASTA file")
parser.add_argument("-o", "--output", required=True, type=str, help="Output pickle file")
parser.add_argument("-d", "--device", default=0, type=int, help="CUDA device id (default: 0)")
args = parser.parse_args()

# Select GPU / CPU
if torch.cuda.is_available():
    device = torch.device(f"cuda:{args.device}")
    print("Using GPU:", device)
else:
    device = torch.device("cpu")
    print("Using CPU")

# Load ProtT5 model
model_name = "../ProtT5/prot_t5_xl_half_uniref50-enc"
print("Loading:", model_name)
tokenizer = T5Tokenizer.from_pretrained(model_name, do_lower_case=False)
model = T5EncoderModel.from_pretrained(model_name).to(device)

# only GPUs support half-precision currently; if you want to run on CPU use full-precision (not recommended, much slower)
if device.type == "cpu":
    model = model.to(torch.float32)
model = model.eval()

# Read FASTA
print("Loading the sequence fasta:", args.input)
protein_ids = []
protein_seqs = []

with open(args.input, "r") as f:
    current_id = None
    current_seq = []
    for line in f:
        line = line.strip()
        if line.startswith(">"):
            if current_id is not None:
                protein_ids.append(current_id)
                protein_seqs.append("".join(current_seq))
            current_id = line[1:]
            current_seq = []
        else:
            current_seq.append(line)

    # Save the last protein
    if current_id is not None:
        protein_ids.append(current_id)
        protein_seqs.append("".join(current_seq))

# replace all rare/ambiguous amino acids by X and introduce white-space between all amino acids
processed_sequences = [" ".join(list(re.sub(r"[UZOB]", "X", sequence))) for sequence in protein_seqs]

# Extract embeddings
protein_embeddings = {}
for idx, protein_sequence in enumerate(processed_sequences):
    # tokenize sequences and pad up to the longest sequence in the batch
    ids = tokenizer(protein_sequence, add_special_tokens=True, padding="longest", return_tensors="pt")
    input_ids = ids["input_ids"].to(device)
    attention_mask = ids["attention_mask"].to(device)
    # generate embeddings
    with torch.no_grad():
        embedding_repr = model(input_ids, attention_mask)

    # Residue representations
    emb_0 = embedding_repr.last_hidden_state[0, :len(protein_seqs[idx])]
    # Mean representation
    emb_0_per_protein = emb_0.mean(dim=0)
    
    # Save the protein's embedding in the dictionary
    protein_embeddings[protein_ids[idx]] = (emb_0_per_protein.cpu().numpy())
    print(f"Processed {idx + 1}/{len(processed_sequences)}: Protein ID = {protein_ids[idx]}, Seq Len = {len(protein_seqs[idx])}, Embedding Shape = {emb_0_per_protein.shape}")

# Save embeddings
with open(args.output, "wb") as f:
    pickle.dump(protein_embeddings, f)

print("Finished.")
print("Number of embeddings:", len(protein_embeddings))
print("Saved to:", args.output)


