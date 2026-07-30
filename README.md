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

# Features
The **ESMC_embeddings.py** script extracts embeddings from the **ESMC** (esmc-600m-2024-12) model using protein sequences in FASTA format. Running the script will generate the embeddings, and here we generate the embeddings for Ara-virus, which are saved in the feature file `Ara-virus_ESMC_1152.pkl`.

To run the script, execute the following command:
```bash
python ESMC_embeddings.py -i ../data/Ara-virus.fasta -o Ara-virus_ESMC_1152.pkl
```
where:
- `-h, --help`: Display help information.
- `-i, --input`: The input protein FASTA file.
- `-o, --output`: The output file containing ESMC embeddings.
- `-d", "--device`: The device used for embedding generation (default: GPU 0).

You can access and download the ESMC model from Hugging Face at the following link: https://huggingface.co/EvolutionaryScale/esmc-600m-2024-12

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
where:
- `-h, --help`: Display help information.
- `-m, --model_path`: The path to the AraVirusPPI models.
- `-e, --embedding`: The protein embeddings file.
- `-i, --input`: The input protein pair file for prediction.
- `-o, --output`: The output prediction result file.
- `-t, --threshold`: Prediction threshold for filtered predictions (default: 0.5).

The provided `Ara-virus_toydata.txt` file contains example protein pairs and can be used to generate prediction results.

