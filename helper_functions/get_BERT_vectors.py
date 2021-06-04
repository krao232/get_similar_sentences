from sentence_transformers import SentenceTransformer
model = SentenceTransformer('bert-large-nli-stsb-mean-tokens')

def get_BERT_vectors(df): 
    inp = list(df['sentence_entity'])
    vecs = model.encode(inp, show_progress_bar = True)
    return vecs