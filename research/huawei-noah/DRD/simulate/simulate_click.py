import json
import numpy as np
# from click_model import PositionBiasedModel
# from load_data import load_data_forEstimate
# from estimate_ips import RandomizedPropensityEstimator

def simulateOneSession(click_model, query_list):
    oneSessionLog = []
    click_list,observe_list, _, _ = click_model.sampleClicksForOneList([d['label'] for d in query_list[1]])
    oneSessionLog = []
    for index, doc in enumerate(query_list[1]):
        oneClickLog = {}
        oneClickLog['qid'] = query_list[0]
        oneClickLog['did'] = doc['docID']
        oneClickLog['label'] = doc['label']
        oneClickLog['isObserve'] = observe_list[index]
        oneClickLog['isClick'] = click_list[index]
        oneClickLog['rankPosition'] = doc['rankPosition']
        # oneClickLog['feature'] = doc['feature']
        oneSessionLog.append(oneClickLog)
    return oneSessionLog

if __name__ =='__main__': 
    pass