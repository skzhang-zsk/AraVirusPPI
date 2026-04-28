from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig
import torch
from Bio import SeqIO
import pickle, os

# Model setup
os.environ["INFRA_PROVIDER"] = "True"  # Use the pre-downloaded parameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Automatically choose GPU if available, otherwise use CPU
client = ESMC.from_pretrained("esmc_300m").to(device)  # Load the pre-trained model to the selected device
print(f"Using device: {device}")

# Load protein sequences from a FASTA file
print("Loading the sequence fasta")
protein_id = []
sequences = []
for record in SeqIO.parse("../data/Ara-virus.fasta", "fasta"):
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

    # Optionally, print sequence length and embedding shape for debugging
    #print(f"{i+1}/{len(sequences)} length:", len(sequences[i]), f"Shape : {embedding_mean.shape}")

# Save the protein embeddings to a pickle file
with open("Ara-virus_ESMC_1152.pkl", "wb") as f:
    pickle.dump(protein_embs, f)

