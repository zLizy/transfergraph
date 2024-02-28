import os

import pandas as pd
from scipy import stats

path_1 = '../rank_final/cifar100/lr_node2vec/results_1.0_128_0.csv'
path_2 = '../rank_final/cifar100/lr_node2vec/results_1.0.csv'

# ft records
df_finetune = pd.read_csv(os.path.join('../..', 'doc', 'records.csv'), index_col=0)

for path in [path_1, path_2]:
    # ,model,score
    df_corr = pd.read_csv(path)

    df_score = df_corr.merge(df_finetune, on='model', how='inner')
    # print(df_score.columns)

    corr = stats.pearsonr(df_score['score'], df_score['test_accuracy'])[0]
    print(f'corr: {corr}')
