#Model training process
import os
import numpy as np
import pickle
import argparse
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold

parser = argparse.ArgumentParser(description="Load specified protein encoding file based on user input and build training and testing datasets.")
parser.add_argument("protein_type", type=str, choices=["ESM1b", "ESM2", "ESMC", "ProtT5"], help="Choose protein encoding type, e.g., ESM1b, ESM2, ESMC, ProtT5.")
args = parser.parse_args()

protein_files = {
    #"ESM1b": "../features/Ara-virus_ESM1b_1280.pkl",
    #"ESM2": "../features/Ara-virus_ESM2_1280.pkl",
    "ESMC": "../features/Ara-virus_ESMC_1152.pkl",
    #"ProtT5": "../features/Ara-virus_ProtT5_1024.pkl" 
}
protein_type = args.protein_type

protein_file = protein_files[protein_type]
with open(protein_file, 'rb') as f:
    protein_data = pickle.load(f)

train = np.genfromtxt('../data/Ara-virus_train.txt', str)
test = np.genfromtxt('../data/Ara-virus_test.txt', str)
X_train, y_train = train[:, :2], train[:, 2].astype(np.float32)
X_test, y_test = test[:, :2], test[:, 2].astype(np.float32)

x_train = np.array([np.hstack([protein_data[a], protein_data[b]]) for a, b in X_train])
x_test = np.array([np.hstack([protein_data[a], protein_data[b]]) for a, b in X_test])
print(len(x_train[0]))

model = xgb.XGBClassifier()
params = {
            'booster': ['gbtree'], 
            "n_estimators":range(100,501,100), "learning_rate" : [0.01,0.05],
            "max_depth" : range(5,16,5), "gamma": [0.0,0.2,0.5],
            "colsample_bytree" : [0.5,0.8,1], "n_jobs":[2] 
        }
GS_model=GridSearchCV(model, param_grid=params, scoring='roc_auc',n_jobs=20,cv=5,verbose=3)
GS_model.fit(x_train, y_train)

output_path=f'../output'
os.makedirs(output_path, exist_ok=True)

# Save the internal cross-validation performance from GridSearchCV
with open(f'{output_path}/{protein_type}_XGBoost_parameter.txt', 'w') as model_parameter:
    model_parameter.write(f'Best_AUC\t{GS_model.best_score_}\n')
    model_parameter.write(f'Best_Params\t{GS_model.best_params_}\n')

# Train five models using the optimal hyperparameters for ensemble prediction on the independent test set
y_score_test_list = [[] for _ in range(5)]

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for fold, (train_index, val_index) in enumerate(kf.split(X_train, y_train)):

    x_train_kf = x_train[train_index]
    y_train_kf = y_train[train_index]

    model = xgb.XGBClassifier(**GS_model.best_params_, random_state=42)
    model.fit(x_train_kf, y_train_kf)

    model_path = f'{output_path}/models'
    os.makedirs(model_path, exist_ok=True)
    with open(f'{model_path}/{protein_type}_XGBoost_model_fold{fold}.pkl', 'wb') as model_file:
        pickle.dump(model, model_file)

    y_score_test = model.predict_proba(x_test)
    y_score_test_list[fold].extend(y_score_test[:,1])

mean_score = np.mean(y_score_test_list, axis=0)
with open(f'{output_path}/{protein_type}_XGBoost_test.txt', 'w') as f:
    f.write('protein1\tprotein2\tlabel\t' + '\t'.join([f'predict_probability{i+1}' for i in range(5)]) + '\tmean_probability\n')
    for j in range(len(y_score_test_list[0])):
        f.write(f"{X_test[j][0]}\t{X_test[j][1]}\t{y_test[j]}\t" + '\t'.join([str(y_score_test_list[i][j]) for i in range(5)]) + f"\t{mean_score[j]}\n")

