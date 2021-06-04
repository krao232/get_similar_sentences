import pandas as pd

def drop_level(df):
    '''helper function to handle errors'''
    to_drop = []
    if 'level_0' in df.columns:
        to_drop.append('level_0')
    if 'level_1' in df.columns:
        to_drop.append('level_1')
    return df.drop(to_drop, axis = 1)
def sample_new_sentences_per_original(df, N):
    variables = ['old_sentence']
    counts = df[variables+["sentence"]].groupby(by = variables).count().reset_index()
    counts['sentence_count'] = counts.sentence
    counts = counts.drop('sentence', axis = 1)

    df1 = df.merge(counts, on= variables)
    df2 = df1[df1.sentence_count<N]
    df3 = df1[df1.sentence_count>=N]
    if len(df3) !=0:
        df4 = df3.groupby(by = variables, as_index = False).apply(pd.DataFrame.sample, n=N).reset_index()
        df5 = pd.concat([df2, df4]).reset_index().drop(['index', 'sentence_count'], axis = 1) 
    else: 
        df5 = df2.reset_index().drop(['index', 'sentence_count'], axis = 1)
    return drop_level(df5)