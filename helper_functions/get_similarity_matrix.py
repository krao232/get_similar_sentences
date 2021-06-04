import pandas as pd 
import numpy as np
from numpy import load
from tqdm.notebook import tqdm
import nferx_py.utils as utils 


def cos_sim(x, y):
    x = np.array(x)
    y = np.array(y)
    dot = x.dot(y)
    m_x = np.sqrt(x.dot(x))
    m_y = np.sqrt(y.dot(y))
    return dot/(m_x*m_y)

def sim_matrix(df_old, df_new): 
    """
    for every sentence in the old (original) set of sentences, get the similarity with the new sentences
    """
    mat = []
    for i in tqdm(df_old.index):
        #sentence1 = df_old.sentence[i]
        vec1 = df_old.vec[i]
        sims = [] 
        for j in df_new.index: 
            #sentence2 = df_new.sentence[j]
            vec2 = df_new.vec[j]
            sim = cos_sim(vec1, vec2)
            sims.append(sim)
        mat.append(sims)
    return np.array(mat)


def get_similarity_matrix(df_old, df_new): 
    return sim_matrix(df_old, df_new)