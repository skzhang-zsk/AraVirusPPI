import argparse
import os
import pickle
import torch
from Bio import SeqIO
from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig

parser = argparse.ArgumentParser(description="Generate ESMC protein embeddings")
parser.add_argument("-i", "--input", required=True, help="Input protein FASTA file")
parser.add_argument("-o", "--output", required=True, help="Output pickle file")
parser.add_argument( "-d", "--device", default=0, type=int, help="cuda device id (default: 0)")
args = parser.parse_args()

# Model setup
os.environ["INFRA_PROVIDER"] = "True"  # Use the pre-downloaded parameters
device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
client = ESMC.from_pretrained("esmc_600m").to(device)  # Load the pre-trained model to the selected device
print(f"Using device: {device}")

# Load protein sequences from a FASTA file
print("Loading the sequence fasta")
protein_id = []
sequences = []
for record in SeqIO.parse(args.input, "fasta"):
    protein_id.append(str(record.id))
    sequences.append(str(record.seq))

# Dictionary to store protein embeddings
protein_embs = {}
for i, protein_sequence in enumerate(sequences):
    print(f"Processing sequence {i+1}/{len(sequences)}")
    
    # Create an ESMProtein object from the sequence
    protein = ESMProtein(sequence=protein_sequence)
    
    # Encode the protein sequence
    protein_tensor = client.encode(protein)
    
    # Obtain logits (embeddings) for the protein
    logits_output = client.logits(
        protein_tensor, LogitsConfig(sequence=True, return_embeddings=True)
    )  # Output shape: torch.Size([1, sequence+2, 1152])
    
    # Calculate the mean of the embeddings
    embedding_mean = torch.mean(logits_output.embeddings[0], dim=0)
    
    # Store the mean embedding in the dictionary
    protein_embs[protein_id[i]] = embedding_mean.cpu().numpy()

# Save the protein embeddings to a pickle file
with open(args.output, "wb") as f:
    pickle.dump(protein_embs, f)

print("Saved:", args.output)
print("Dictionary length:", len(protein_embs))

