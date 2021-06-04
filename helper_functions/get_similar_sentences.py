import pandas as pd 
import numpy as np
from numpy import load
from tqdm.notebook import tqdm
import nferx_py.utils as utils 


def similars_from_matrix(min_sim, max_sim, diff, matrix): 
    dim = matrix.shape[1]
    used_new_indexes = set()
    keep = set()
    #index in the originals. elem is the column
    for i, elem in enumerate(tqdm(matrix)): 
        candidates = []
        #index in the new set. subelem is the similarity
        for j, subelem in enumerate(elem): 
            add = True
            #within the bounds
            if (subelem>min_sim) and (subelem<max_sim): 
                for c in candidates: 
                    if abs(c[0]-subelem)<diff:
                        add = False
                if add == True: 
                    if j not in used_new_indexes:
                        candidates.append((subelem, i, j))
                        used_new_indexes.add(j)

            for c in candidates: 
                keep.add(c)
    #print('found {} similar sentences'.format(len(keep)))
    return list(keep)


def get_similar_sentences(df_new, matrix, min_sim = 0.607, max_sim = 0.975, diff = 0.0015):
    """
    df_new is the sentences dataframe
    matrix is the similarity matrix of oldXnew
    min_sim is the minimum similarity tolerated (default is what I've found to be mean+2sd for all sentences
    max default is arbitrary
    diff is the difference between similar sentences (to try and avoid duplicates/near duplicates). This doesn't work so great.
    """
    similars_list = similars_from_matrix(min_sim, max_sim, diff, matrix)
    
    keep_inds = [n[2] for n in similars_list]
    
    final = df_new[df_new.index.isin(keep_inds)]
    
    new_index_old_index = {}
    for elem in similars_list: 
        new = elem[2]
        old = elem[1]
        new_index_old_index[new] = old
    
    final['old_sentence'] = [new_index_old_index[n] for n in final.index]
    
    return final

