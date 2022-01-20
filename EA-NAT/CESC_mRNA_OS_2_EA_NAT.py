import pandas as pd
import random as rn
import os

from data.preProcess import get_one_race, get_n_years, standarize_dataset, get_dataset
from baseline.classify import run_mixture_cv, run_one_race_cv, run_supervised_transfer_cv

import tensorflow
tensorflow.random.set_seed(11111)

os.environ['PYTHONHASHSEED'] = '0'
os.environ["KERAS_BACKEND"] = "tensorflow"
rn.seed(11111)

def run_cv(cancer_type, feature_type, target, years=3):

    print (cancer_type, feature_type, target, years)
    dataset = get_dataset(cancer_type=cancer_type, feature_type=feature_type, target=target, groups=("WHITE", "NAT_A"))
    
    dataset = standarize_dataset(dataset)
    dataset_white = get_one_race(dataset, 'WHITE')
    dataset_white = get_n_years(dataset_white, years)
    dataset_native = get_one_race(dataset, 'NAT_A')
    dataset_native = get_n_years(dataset_native, years)
    dataset = get_n_years(dataset, years)

    k = 200 if 'mRNA' in feature_type else -1

    X, Y, R, y_sub, y_strat = dataset
    df = pd.DataFrame(y_strat, columns=['RY'])
    df['R'] = R
    df['Y'] = Y
    print(X.shape)
    print(df['RY'].value_counts())
    print(df['R'].value_counts())
    print(df['Y'].value_counts())


    # There are 5 multiethnic experiments schemes based on the source and target domains,
    # as shown in Table 1. For all these 5 experiments, there are 3 classification methods
    ## 1. mixture learning
    ## 2. independent (one_race) learning
    ## 3. transfer learning
    
    # we mainly work on the fine-tuning of transfer learning with different model settings and 
    # hyperparameter values with 4 and 5 stratified cross-validation settings. Furthermore, we change the
    # datasets in main method (run_cv()) and in get_one_race() methods to change the source and target
    # domains when working with different multiethnic data distributions.
    
    parametrs_mixture_cv = {'fold': 4, 'k': k, 'val_size': 0.0, 'batch_size': 20,
                     'learning_rate': 0.01, 'lr_decay': 0.0, 'dropout': 0.5,
                     'L1_reg': 0.001, 'L2_reg': 0.001, 'hiddenLayers': [128, 64]}
    parametrs_one_race_white = {'fold': 4, 'k': k, 'val_size': 0.0, 'batch_size': 20,
                   'learning_rate': 0.01, 'lr_decay': 0.0, 'dropout': 0.5,
                   'L1_reg': 0.001, 'L2_reg': 0.001, 'hiddenLayers': [128, 64]}
    parametrs_one_race_native = {'fold': 4, 'k': k, 'val_size': 0.0, 'batch_size': 4,
                   'learning_rate': 0.01, 'lr_decay': 0.0, 'dropout': 0.5,
                   'L1_reg': 0.001, 'L2_reg': 0.001, 'hiddenLayers': [128, 64]}
    parametrs_transfer_learning = {'fold': 4, 'k': k, 'val_size': 0.0, 'batch_size': 16, 'tune_epoch': 20, 'train_epoch': 100,
                    'learning_rate': 0.01, 'lr_decay': 0.0, 'dropout': 0.5, 'tune_lr': 0.0001,
                    'L1_reg': 0.001, 'L2_reg': 0.001, 'hiddenLayers': [128, 64], 'tune_batch': 32}
    

    res = pd.DataFrame()
    
    # each single experiments was run for 20 runs
    for i in range(20):
        seed = i
        # the following is to compute the first three multiethnic schemes
        df_mixture = run_mixture_cv(seed, dataset, **parametrs_mixture_cv)
        
        # the following is to compute the first independent learning scheme
        df_white = run_one_race_cv(seed, dataset_white, **parametrs_one_race_white)
        df_white = df_white.rename(columns={"Auc": "White_ind"})
        
        # the following is to compute the second independent learning scheme
        df_native = run_one_race_cv(seed, dataset_native, **parametrs_one_race_native)
        df_native = df_native.rename(columns={"Auc": "Native_ind"})
        
        # the following is to compute the transfer learning scheme
        df_tl = run_supervised_transfer_cv(seed, dataset, **parametrs_transfer_learning)
        
        # the AUROC scores for all 5 schemes  
        final_scores = pd.concat([df_mixture, df_white['White_ind'], df_native['Native_ind'], df_tl['TL_Auc']],
                        sort=False, axis=1)

        print(final_scores)
        final_results = res.append(final_scores)
        
    # change the path to working directory
    f_name = '/path_to_your_dir/Results/' + cancer_type + \
             '-AA-NAT_A-' + feature_type[0] + '-' + target + '-' + str(years) + 'YR-our_fine_tuning_model.xlsx'
    final_results.to_excel(f_name)


def main():
    # experiment main structure:
        # (1, 2, 3, 4)
        ## 1. cancer type
        ## 2. feature experssion type
        ## 3. clinical endpoint outcome type
        ## 4. target year
    run_cv('CESC', 'mRNA', 'OS', years=2)

if __name__ == '__main__':
    main()