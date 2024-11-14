import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler

def scale_dataset(train,vali,test):
    fe_train = np.array([fe for fe in train['feature'].values])
    fe_vali = np.array([fe for fe in vali['feature'].values])
    fe_test = np.array([fe for fe in test['feature'].values])
    # zero_col_idx=[]
    # for i in tqdm(range(fe_matrix.shape[1])):
    #     if len(np.unique(fe_matrix[:,i]))==1:
    #         zero_col_idx.append(i)
    #         #print(i)
    # fe_matrix = np.delete(fe_matrix, zero_col_idx, axis=1)
    scale_tool = MinMaxScaler().fit(fe_train)

    fe_scale_train = scale_tool.transform(fe_train)
    fe_scale_vali = scale_tool.transform(fe_vali)
    fe_scale_test = scale_tool.transform(fe_test)
    
    train['shrink_fe'] = list(fe_scale_train)
    vali['shrink_fe'] = list(fe_scale_vali)
    test['shrink_fe'] = list(fe_scale_test)

    train.drop('feature', axis=1, inplace=True)  
    vali.drop('feature', axis=1, inplace=True)  
    test.drop('feature', axis=1, inplace=True)  

    train.rename(columns={'shrink_fe':'feature'}, inplace=True)
    vali.rename(columns={'shrink_fe':'feature'}, inplace=True)
    test.rename(columns={'shrink_fe':'feature'}, inplace=True)

    return train,vali,test
