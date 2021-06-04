import json
import urllib
import requests
import pandas as pd
import nferx_py.fn as nf
from config import *
from tqdm.notebook import tqdm

"""
documentation for api:
https://github.com/lumenbiomics/SentenceAPI/blob/master/golang-aggregator/README_bulkInference.md#fetch-sentences--bulkinferencev1getsentences
"""

nf.authenticate(NFERX_PY_USER, API_KEY_PRE_STAGING)
AUTH = nf.AUTH

def process_raw_api_output(entity1, entity2, result): 
    if result!=None:
        for elem in result: 
            elem['entity1_input'] = entity1
            elem['entity2_input'] = entity2
            elem['sentence_entity'] = elem['sentence'][0]
            elem['sentence'] = elem['sentence'][1]
        return result
    
match = {
        'entity1': 'entity1', 
        'entity2': 'entity2',
        'max_count(default=1000)': 'max_count', 
        'window(default=30)': 'window', 
        'corpus(default=core)': 'target_corpus', 
        'doc_token(default=None)': 'doc_token'
    }

def api_call(inp): 
    url = 'https://pre-staging.nferx.com/bulkInference/v1/getSentences?'
    body = {
            'max_count': 1000,
            'window': 30,
            'remove_duplicates': True,
            'remove_invalid': True,
#             'target_corpus': 'corpus'
           }

    for key in inp: 
        if inp[key]!=None: 
            body[match[key]] = inp[key]
            
    body['num_entities'] = sum([int(bool(inp['entity1'])), int(bool(inp['entity2']))])

    res = requests.get(url+urllib.parse.urlencode(body), auth = AUTH)
    
    result = res.json()['data']
    return pd.DataFrame(process_raw_api_output(inp['entity1'], inp['entity2'], result))

def concat_api_call_result(lis): 
    res = pd.DataFrame()
    failed_ix = [] 
    for i, df in enumerate(lis): 
        if len(df) == 0: 
            failed_ix.append(i)
        res = pd.concat([res, df])
    return failed_ix, res.reset_index().drop('index', axis = 1)


import time
from concurrent.futures import ThreadPoolExecutor
NUM_THREADS = 10

def get_sentences(df):
    d0 = df.to_json(orient = 'records')
    d1 = json.loads(d0)
        
    start = time.time()
    print('getting sentences for {} inputs'.format(len(d1)))
    
    pool = ThreadPoolExecutor(NUM_THREADS)
        
    res0 = list(tqdm(pool.map(api_call, d1), total = len(d1)))
    failed_ix, res1 = concat_api_call_result(res0)
    
    end = time.time()
    print('fetched {} sentences in {} seconds'.format(len(res1), end-start))
    return df[df.index.isin(failed_ix)], res1
