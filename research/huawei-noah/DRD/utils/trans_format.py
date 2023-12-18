import pandas as pd

def format_trans(df, mode='train'):
    df.rename(columns={'queryID':'qid', 'docID':'did'}, inplace = True)
    if mode=='train' or mode=='vali':
        df = df[['qid','did','label','feature','rankPosition']]
    else:
        df = df[['qid','did','label','feature']]
    return df


if __name__=="__main__":
    pass