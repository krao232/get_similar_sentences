import requests
import json
import numpy as np
import copy

def get_model_predictions(df, project_name, model_name, project_type, group_name):
    df = copy.deepcopy(df)
    
    """function to get the predicted labels for the model on a dataset"""
    
    headers = {'Cookie':'csrftoken=qXjWrV0ilgGCpt7nKnUFVQqHERMiS3i2VieNMGZzhOTKfLlISlqtyJwJaVuhrysD; sessionid=e29vg4kpeyhjw6tyoiibydxiagbd3o12'}
    
    #parameters for prediction
    postDict = {'project_name': project_name,
                'model_name': model_name,
                'project_type': project_type,
                'group_name': group_name}
    
    DeepModelBuilderServer = "https://pre-staging.nferx.com"
    modelStatusEndpoint = "/tagrecorder/v1/getModelServingStatus"
    
    def fetchModelServer(postDict):
        print(DeepModelBuilderServer + modelStatusEndpoint)
        resp = requests.post(DeepModelBuilderServer + modelStatusEndpoint, json=postDict, headers=headers,
                             verify=False)
        respData = json.loads(resp.text)

        if respData["success"]:
            server = respData["server_address"].split(",")
            BERTEndPoint = "http://" + server[0].rstrip("/") + "/predict"
            print("Using model server: ", BERTEndPoint)
            if len(server) > 1:
                print("The inference is running only on single server")
            return True, BERTEndPoint
        else:
            print("The model is not loaded in Deepmodel builder. Please load the model and retry")
            return False, ""

    status, BERTEndPoint = fetchModelServer(postDict)
    
    sentence_entity = list(df['sentence_entity'])
    sample_post_query = {'num_entities': 0, 
                         'ids': list(range(len(sentence_entity))), 
                         'sentences': sentence_entity}
    

    r = requests.post(BERTEndPoint, json=sample_post_query,headers=headers, verify=False)
    response = json.loads(r.text)
    
    df['confidences'] = np.float64(list(response['confidences']))
    df['predicted_label'] = list(response['labels'])
    
    return df
    