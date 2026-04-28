# AraVirusPPI
Plant-virus protein-protein interactions (PPIs) play crucial roles in viral infection and host immune responses, yet their systematic identification remains limited by experimental constraints. We present **AraVirusPPI**, the first machine learning-based *A. thaliana*-virus PPI predictor. AraVirusPPI employs the protein language model ESM Cambrian (ESMC) to encode sequence features and combines these representations with Extreme Gradient Boosting (XGBoost) to build the prediction model.

# Data
We provided the following data:
1. **Positive-Negative Samples**: `Ara-virus_positive-negative.txt` — Contains both positive and negative samples, which are used for training and testing the model.
2. **Training Set**: `Ara-virus_train.txt` — Contains the training data for model training.
3. **Test Set**: `Ara-virus_test.txt` — Contains the test data for model evaluation.
4. **Protein Sequences**: `Ara-virus.fasta` — Contains the corresponding protein sequences in FASTA format.

# Features
The **ESMC_embeddings.py** script extracts embeddings from the **ESMC** (esmc-600m-2024-12) model using protein sequences in FASTA format. Running the script will generate the embeddings, and here we generate the embeddings for Ara-virus, which are saved in the feature file `Ara-virus_ESMC_1152.pkl`.

To run the script, execute the following command:
```bash
python ESMC_embeddings.py
```
You can access and download the ESMC model from Hugging Face at the following link: https://huggingface.co/EvolutionaryScale/esmc-600m-2024-12

# Scripts
This section includes the training process for the XGBoost model. The script outlines the steps for training the model using the provided datasets, including:

1. Data preprocessing
2. Feature extraction (using the **esmc-600m-2024-12** model)
3. Training the XGBoost model with 5-fold cross-validation
4. Evaluating model performance using cross-validation scores
5. Predicting on the independent test set

### Training Process
To train the XGBoost model, run the following command:

```bash
python XGBoost.py ESMC
```

# Output
The output includes the five models trained using 5-fold cross-validation with AraVirusPPI, along with the cross-validation scores and the scores obtained by using these models to predict the independent test set. The results are stored in the following files: 
1. ESMC_XGBoost_5fold.txt: Contains the scores from the 5-fold cross-validation. 
2. ESMC_XGBoost_test.txt: Contains the prediction scores on the independent test set using the 5 models, along with the averaged results.

# Usage

To make predictions using the pre-trained models, follow this step:

### Run the Prediction Script

To use the `predict.py` script, execute the following command:

```bash
python predict.py
```
You can use the provided `Ara-virus_toydata.txt` file for prediction, which contains sample data to test the model.


