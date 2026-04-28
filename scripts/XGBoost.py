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
GS_model=GridSearchCV(model, param_grid=params, scoring='average_precision',n_jobs=20,cv=5,verbose=3)
GS_model.fit(x_train, y_train)

output_path=f'../output'
os.makedirs(output_path, exist_ok=True)

XGBoost_best_model=GS_model.best_estimator_
protein1_2_list, y_label_list, y_pred_k, y_score_list=[], [], [], []
y_score_test_list = [[] for _ in range(5)] 

kf = StratifiedKFold(n_splits=5,shuffle=True,random_state=42)

for fold, (train_index,val_index) in enumerate(kf.split(X_train,y_train)):

    x_train_kf,x_val_kf = x_train[train_index],x_train[val_index]
    y_train_kf,y_val_kf = y_train[train_index],y_train[val_index]
    XGBoost_best_model.fit(x_train_kf,y_train_kf)

    model_path = f'{output_path}/5fold_models'
    os.makedirs(model_path, exist_ok=True)
    with open(f'{model_path}/XGBoost+{protein_type}_model_fold{fold}.pkl', 'wb') as model_file:
        pickle.dump(XGBoost_best_model, model_file)
    
    protein1_2_list.extend(X_train[val_index])  #protein1+'\t'+protein2
    y_label_list.extend(y_val_kf)  #label
    y_pred = XGBoost_best_model.predict(x_val_kf)
    y_pred_k.extend(y_pred)  #predict label
    y_score = XGBoost_best_model.predict_proba(x_val_kf)
    y_score_list.extend(y_score[:,1])  #predict 1 score

    y_score_test = XGBoost_best_model.predict_proba(x_test)  #predict score
    y_score_test_list[fold].extend(y_score_test[:,1])  #predict 1 score list


with open(f'{output_path}/XGBoost+{protein_type}_5fold.txt', 'w') as f:
    f.write('protein1\tprotein2\tlabel\tpredict\tpredict_probability\n')
    for i in range(len(y_label_list)):
        f.write(f"{protein1_2_list[i][0]}\t{protein1_2_list[i][1]}\t{y_label_list[i]}\t{y_pred_k[i]}\t{y_score_list[i]}\n")

mean_score = np.mean(y_score_test_list, axis=0)
with open(f'{output_path}/XGBoost+{protein_type}_test.txt', 'w') as f:
    f.write('protein1\tprotein2\tlabel\t' + '\t'.join([f'predict_probability{i+1}' for i in range(5)]) + '\tmean_probability\n')
    for j in range(len(y_score_test_list[0])):
        f.write(f"{X_test[j][0]}\t{X_test[j][1]}\t{y_test[j]}\t" + '\t'.join([str(y_score_test_list[i][j]) for i in range(5)]) + f"\t{mean_score[j]}\n")

