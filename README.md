# AraVirusPPI
Plant-virus protein-protein interactions (PPIs) play crucial roles in viral infection and host immune responses, yet their systematic identification remains limited by experimental constraints. We present **AraVirusPPI**, the first machine learning-based *A. thaliana*-virus PPI predictor. AraVirusPPI employs the protein language model ESM Cambrian (ESMC) to encode sequence features and combines these representations with Extreme Gradient Boosting (XGBoost) to build the prediction model.

# Requirements
`AraVirusPPI.yml`: Environment configuration file for the AraVirusPPI model.

# Dataset
We provided the following data:
1. **Positive-Negative Samples**: `Ara-virus_positive-negative.txt` — Contains both positive and negative samples, which are used for training and testing the model.
2. **Training Set**: `Ara-virus_train.txt` — Contains the training data for model training.
3. **Test Set**: `Ara-virus_test.txt` — Contains the test data for model evaluation.
4. **Protein Sequences**: `Ara-virus.fasta` — Contains the corresponding protein sequences in FASTA format.
5. **Additional Independent Test Set**: `Additional_independent_test.txt` — Contains an additional independent dataset of unseen virus species used for model evaluation.
6. **Additional Independent Test Protein Sequences**: `Additional_independent_test.fasta` — Contains the corresponding protein sequences of the additional independent test set in FASTA format.

# Features
The **ESMC_embeddings.py** script extracts embeddings from the **ESMC** (esmc-600m-2024-12) model using protein sequences in FASTA format. Running the script will generate the embeddings, and here we generate the embeddings for Ara-virus, which are saved in the feature file `Ara-virus_ESMC_1152.pkl`. Similarly, the ESM-1b, ESM-2, and ProtT5 models can be used to generate protein embeddings with the corresponding scripts.

To run the script, execute the following command:
```bash
python ESMC_embeddings.py -i ../data/Ara-virus.fasta -o Ara-virus_ESMC_1152.pkl
python ESM1b_ESM2_embeddings.py -i ../data/Ara-virus.fasta -o Ara-virus_ESM1b_1280.pkl
python ESM1b_ESM2_embeddings.py -i ../data/Ara-virus.fasta -o Ara-virus_ESM2_1280.pkl
python ProtT5_embeddings.py -i ../data/Ara-virus.fasta -o Ara-virus_ProtT5_1024.pkl
```
- `-h, --help`: Display help information.
- `-i, --input`: The input protein FASTA file.
- `-o, --output`: The output file containing ESMC embeddings.
- `-d, --device`: The device used for embedding generation (default: GPU 0).
- `-l, --truncation_seq_length`: The maximum sequence length used for embedding generation (ESM-1b and ESM-2 only; default: 1022).

You can access and download the models from Hugging Face at the following link: https://huggingface.co/EvolutionaryScale/esmc-600m-2024-12, https://huggingface.co/facebook/esm1b_t33_650M_UR50S, https://huggingface.co/facebook/esm2_t33_650M_UR50D,  https://huggingface.co/Rostlab/prot_t5_xl_half_uniref50-enc

# Scripts
The **XGBoost.py** script trains the AraVirusPPI model using XGBoost with ESMC protein embeddings.

### Training Process
To train the model, run:

```bash
python XGBoost.py ESMC
```

# Output
The output includes the five models trained using 5-fold cross-validation, along with the cross-validation performance and prediction results on the independent test set. The results are stored in the following files:
1. `models`: Contains the five XGBoost models trained using different folds of the 5-fold cross-validation.
2. `ESMC_XGBoost_parameter.txt`: Contains the best cross-validation AUC score and the corresponding optimal model parameters identified during hyperparameter optimization.
3. `ESMC_XGBoost_test.txt`: Contains the prediction scores from the five models on the independent test set, together with the averaged prediction score.

# Usage
To make predictions using the pre-trained models, follow this step:
### Run the Prediction Script
To use the `predict.py` script, execute the following command:
```bash
python predict.py -m ../output/models -e ../features/Ara-virus_ESMC_1152.pkl -i Ara-virus_toydata.txt -o Ara-virus_toydata_score.txt
```
- `-h, --help`: Display help information.
- `-m, --model_path`: The path to the AraVirusPPI models.
- `-e, --embedding`: The protein embeddings file.
- `-i, --input`: The input protein pair file for prediction.
- `-o, --output`: The output prediction result file.
- `-t, --threshold`: Prediction threshold for filtered predictions (default: 0.5).

The provided `Ara-virus_toydata.txt` file contains example protein pairs and can be used to generate prediction results.

