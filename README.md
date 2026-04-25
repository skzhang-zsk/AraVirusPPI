# AraVirusPPI
Plant-virus protein-protein interactions (PPIs) play crucial roles in viral infection and host immune responses, yet their systematic identification remains limited by experimental constraints. In this manuscript, we report AraVirusPPI, the first machine learning-based A. thaliana-virus PPI predictor. AraVirusPPI employs the protein language model ESM Cambrian (ESMC) to encode sequence features and combines these representations with Extreme Gradient Boosting (XGBoost) to build the prediction model. 

# Data
We provided the following data: the training set (Ara-virus_train.txt), the test set (Ara-virus_test.txt), and the corresponding protein sequences (Ara-virus.fasta).

# Features
The feature file Ara-virus_ESMC_1152.pkl contains the features extracted using the ESMC (esmc-600m-2024-12) model, which was used for model training and test. This model can be accessed at Hugging Face (https://huggingface.co/EvolutionaryScale/esmc-600m-2024-12).

# Scripts
This section includes the training process for the XGBoost model. The script outlines the steps for training the model using the provided datasets, including:

(1) Data preprocessing

(2) Feature extraction (using the esmc-600m-2024-12 model)

(3) Training the XGBoost model with 5-fold cross-validation

(4) Evaluating model performance using cross-validation scores

(5) Predicting on the independent test set


# Output
The output includes the five models trained using 5-fold cross-validation with AraVirusPPI, along with the cross-validation scores and the scores obtained by using these models to predict the independent test set. The results are stored in the following files: 

(1) ESMC_XGBoost_5fold.txt: Contains the scores from the 5-fold cross-validation. 

(2) ESMC_XGBoost_test.txt: Contains the prediction scores on the independent test set using the 5 models, along with the averaged results.

