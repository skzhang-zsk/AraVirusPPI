import os
import pickle
import argparse
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser(description="Predict protein-protein interaction using AraVirusPPI models.")
parser.add_argument("-m", "--model_path", type=str, required=True, help="Path to AraVirusPPI models")
parser.add_argument("-e", "--embedding", type=str, required=True, help="Protein embedding pickle file")
parser.add_argument("-i", "--input", type=str, required=True, help="Protein pair file for prediction")
parser.add_argument("-o", "--output", type=str, required=True, help="Output prediction file")
parser.add_argument("-t", "--threshold", type=float, default=0.5, help="Threshold for filtered prediction output")
args = parser.parse_args()

# 1. Set Paths
model_path = args.model_path
embeddings_file = args.embedding
pair_file = args.input

# 2. Load Protein Embedding Data
with open(embeddings_file, "rb") as f1:
    embeddings = pickle.load(f1)

# 3. Read Protein Pairs
protein_pairs = []
with open(pair_file, "r") as f:
    for line in f:
        protein_pairs.append(line.strip().split("\t"))

# 4. Construct Predict Data
def transform_features(protein_pairs, embeddings):
    """
    Convert protein pairs into feature vectors for the model input.
    """
    X_test = []

    for a, b in protein_pairs:
        feature_vector = np.hstack(
            [embeddings[a], embeddings[b]]
        )
        X_test.append(feature_vector)

    return np.array(X_test)

X_test = transform_features(protein_pairs, embeddings)


# 5. Load 5 Models
models = []
for fold in range(5):
    model_filename = f"{model_path}/ESMC_XGBoost_model_fold{fold}.pkl"
    with open(model_filename, "rb") as model_file:
        model = pickle.load(model_file)
        models.append(model)

# 6. Predict
y_scores = []
for model in models:
    y_scores.append(model.predict_proba(X_test)[:, 1])

y_scores = np.array(y_scores)
y_score_mean = np.mean(y_scores, axis=0)

# 7. Save Prediction Results
df_result = pd.DataFrame({
    'Protein1': [pair[0] for pair in protein_pairs],
    'Protein2': [pair[1] for pair in protein_pairs],
    'Score1': y_scores[0],
    'Score2': y_scores[1],
    'Score3': y_scores[2],
    'Score4': y_scores[3],
    'Score5': y_scores[4],
    'Mean_Score': y_score_mean
})

df_result.to_csv(args.output, sep="\t", index=False)

# 8. Additional Output for Mean_Score > threshold
filtered_output_file = args.output.replace(".txt", "_filtered.txt")
df_result[df_result['Mean_Score'] > args.threshold].to_csv(filtered_output_file, sep="\t", index=False)

print("Prediction finished.")
print("Output:", args.output)
print("Filtered output:", filtered_output_file)
print("Total pairs:", len(df_result))
print("Filtered pairs:", len(df_result[df_result['Mean_Score'] > args.threshold]))

