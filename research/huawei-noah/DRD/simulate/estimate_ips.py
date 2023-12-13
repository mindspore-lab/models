import numpy as np
import random
import json
import copy
import os
import sys
sys.path.append("..")
import utils.click_model as CM
from tqdm import tqdm


class RandomizedPropensityEstimator:
    def __init__(self, file_name=None, TopK=10):
        self.TopK = TopK
        # If file_name is not None, 
        if file_name:
            self.loadEstimatorFromFile(file_name)

    def loadEstimatorFromFile(self, file_name):
        with open(file_name) as data_file:	
            data = json.load(data_file)
            self.click_model = CM.loadModelFromJson(data['click_model'])
            self.IPW_list = data['IPW_list']
            self.TopK = data['TopK']
        return None

    def estimateParametersFromModel(self, click_model, rank_data, session_num):
        self.click_model = click_model
        click_count = [[0 for _ in range(x+1)] for x in range(rank_data.rank_list_size)]
        label_lists = copy.deepcopy(rank_data.label_lists)
        # simulate click on random rank list
        for _ in tqdm(range(int(session_num))):
            index = random.randint(0,len(label_lists)-1)
            random.shuffle(label_lists[index])
            click_list, _, _, _ = self.click_model.sampleClicksForOneList(label_lists[index])
            for i in range(len(click_list)):
                click_count[len(click_list)-1][i] += click_list[i]
        # count click num for each position to eastimate ips
        first_click_count = [0 for _ in range(rank_data.rank_list_size)]
        agg_click_count = [0 for _ in range(rank_data.rank_list_size)]
        for x in range(len(click_count)):
            for y in range(x,len(click_count)):
                first_click_count[x] += click_count[y][0]
                agg_click_count[x] += click_count[y][x]
        # for position i: 1/IPS = min(click_on_position_1/(click_on_position_i + 1e-6), click_on_position_1)
        #self.IPW_list = [min(first_click_count[x]/(agg_click_count[x]+10e-6), first_click_count[x]) for x in range(len(click_count))]
        self.IPW_list = []
        for x in range(len(click_count)):
            if agg_click_count[x] == 0:
                IPW_val = 0
                self.IPW_list.append(IPW_val)
            else:
                IPW_val = first_click_count[x]/(agg_click_count[x])
                self.IPW_list.append(IPW_val)
        #print([round(1/(pbm.getExamProb(i)*self.IPW_list[i]), 3) for i in range(len(self.IPW_list))])

    def outputEstimatorToFile(self, file_name):
        json_dict = {
            'click_model' : self.click_model.getModelJson(),
            'IPW_list' : self.IPW_list,
            'TopK':self.TopK,
        }
        with open(file_name, 'w') as fout:
            fout.write(json.dumps(json_dict, indent=4, sort_keys=True))
        return None
    
    def getPropensityForOneList_ForTrain(self, click_list, use_non_clicked_data=False):
        propensity_weights = []
        for r in range(len(click_list)):
            pw = 0.0
            if (click_list[r] == 0) and (r < self.TopK):
                pw = 1.0
            elif use_non_clicked_data or (click_list[r] > 0):
                try:
                    pw = self.IPW_list[r]
                except IndexError:
                    pw = 0.0
            propensity_weights.append(pw)
        return propensity_weights
    
    def getPropensityForOneList(self, click_list, use_non_clicked_data=False):
        propensity_weights = []
        for r in range(len(click_list)):
            pw = 0.0
            if use_non_clicked_data or (click_list[r] > 0):
                try:
                    pw = self.IPW_list[r]
                except IndexError:
                    pw = 0.0
            propensity_weights.append(pw)
        return propensity_weights


class OraclePropensityEstimator:
    def __init__(self, click_model):
        self.click_model = click_model

    def getPropensityForOneList(self, click_list, use_non_clicked_data=False):
        return self.click_model.estimatePropensityWeightsForOneList(click_list, use_non_clicked_data)

if __name__ == "__main__":
    pass
    
