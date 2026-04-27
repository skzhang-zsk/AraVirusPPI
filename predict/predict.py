
import os
import pickle
import numpy as np
import pandas as pd

# 1. Set Paths
os.chdir("predict")
model_path = f"../output/5fold_models"
Ara_virus_embeddings_file = f"../features/Ath-virus_ESMC_1152.pkl"  # Replace with your own data embeddings
pair_file = "./Ara-virus_pair.txt"

# 2. Load Protein Embedding Data
with open(Ara_virus_embeddings_file, "rb") as f1:
    Ara_virus_embeddings = pickle.load(f1)

# 3. Read Protein Pairs 
protein_pairs = []
with open(pair_file, "r") as f:
    for line in f:
        protein_pairs.append(line.strip().split("\t"))

# 4. Construct Predict Data 
def transform_features(protein_pairs, Ara_virus_embeddings):
    """
    Convert protein pairs into feature vectors for the model input.
    """
    X_test = []
    for a, b in protein_pairs:
        feature_vector = np.hstack([Ara_virus_embeddings[a], Ara_virus_embeddings[b]]) # a is Arabidopsis(host), b is virus
        X_test.append(feature_vector)
    return np.array(X_test)
X_test = transform_features(protein_pairs, Ara_virus_embeddings)

# 5. Load 5 Models
models = []
for fold in range(5):
    model_filename = f"{model_path}/XGboost+ESMC_model_fold{fold}.pkl"
    with open(model_filename, "rb") as model_file:
        model = pickle.load(model_file)
        models.append(model)

# 6. Predict
y_scores = []
for model in models:
    y_scores.append(model.predict_proba(X_test)[:, 1])  
y_scores = np.array(y_scores)
y_score_mean = np.mean(y_scores, axis=0)  # Take the mean

# 8. Save Prediction Results
df_result = pd.DataFrame({
    'Protein1': [pair[0] for pair in protein_pairs],
    'Protein2': [pair[1] for pair in protein_pairs],
    'Mean_Score': y_score_mean
})

output_file = f"./output/XGBoost+ESMC_Predictions.txt"
df_result.to_csv(output_file, sep="\t", index=False)

# 9. Additional Output for Mean_Score > 0.5 
filtered_output_file = f"./output/XGBoost+ESMC_Predictions_filtered.txt"
df_result[df_result['Mean_Score'] > 0.5].to_csv(filtered_output_file, index=False, sep='\t')

