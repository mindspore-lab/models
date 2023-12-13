import os,sys
import random, json
import numpy as np


def loadModelFromJson(model_desc):
	click_model = PositionBiasedModel()
	if model_desc['model_name'] == 'user_browsing_model':
		pass
		#click_model = UserBrowsingModel()
	click_model.eta = model_desc['eta']
	click_model.click_prob = model_desc['click_prob']
	click_model.exam_prob = model_desc['exam_prob']
	return click_model

class ClickModel:
    # pos_click_prob = prob(click = 1 | relevance = 1) ==> incorporating click noise, default 1
    # neg_click_prob = prob(click = 1 | relevance = 0) ==> incorporating click noise, default 0
    # eta ==> control the severity of presentation bias
	def __init__(self, neg_click_prob=0.0, pos_click_prob=1.0, relevance_grading_num=1, eta=1.0, TopK=10, initial_incorrect_click_p=0.638):
		self.TopK = TopK
		self.setExamProb(eta)
		self.setClickProb(neg_click_prob, pos_click_prob, relevance_grading_num)
		self.setClickProb(neg_click_prob, pos_click_prob, relevance_grading_num)
		self.setTrustProb(initial_incorrect_click_p)

	@property
	def model_name(self):
		return 'click_model'

	# Serialize model into a json.
	def getModelJson(self):
		desc = {
			'model_name' : self.model_name,
			'eta' : self.eta,
			'TopK': self.TopK,
			'click_prob' : self.click_prob,
			'exam_prob' : self.exam_prob,
			'positive_trust_prob':self.positive_trust_prob,
			'negative_trust_prob':self.negative_trust_prob
		}
		return desc

	# Generate noisy click probability based on relevance grading number
	# Inspired by ERR
	def setClickProb(self, neg_click_prob, pos_click_prob, relevance_grading_num):
		# b = (pos_click_prob - neg_click_prob)/(pow(2, relevance_grading_num) - 1)
		# a = neg_click_prob - b
		# self.click_prob = [a + pow(2,i)*b for i in range(relevance_grading_num+1)]
		# self.click_prob = [i/relevance_grading_num for i in range(relevance_grading_num + 1)]
		self.click_prob = [(pow(2,i)-1)/(pow(2, relevance_grading_num) - 1) for i in range(relevance_grading_num + 1)]

	# Set the examination probability for the click model.
	def setExamProb(self,eta):
		self.eta = eta
		self.exam_prob = None
		return None

	def setTrustProb(self, initial_incorrect_click_p):
		self.initial_incorrect_click_p = initial_incorrect_click_p
		self.positive_trust_prob = None
		self.negative_trust_prob = None

	# Sample clicks for a list
	def sampleClicksForOneList(self, label_list):
		return None

	# Estimate propensity for clicks in a list
	def estimatePropensityWeightsForOneList(self, click_list, use_non_clicked_data=False):
		return None
	
class PositionBiasedModel(ClickModel):

	@property
	def model_name(self):
		return 'position_biased_model'

	def setExamProb(self, eta):
		self.eta = eta
		self.original_exam_prob = [0.68, 0.61, 0.48, 0.34, 0.28, 0.20, 0.11, 0.10, 0.08, 0.06]
		# self.original_exam_prob = [1/(x+1) for x in range(self.TopK)]
		self.original_exam_prob.append(0.0)
		if eta != 0:
			self.exam_prob = [pow(x, eta) for x in self.original_exam_prob]
		else:
			self.exam_prob = [pow(x, eta) for x in self.original_exam_prob[:-1]] + [0.0]

	# unique examprob for Top-10 item and constant 10-th examprob for others whose position >10
	def getExamProb(self, rank):
		len_examProb = len(self.exam_prob)
		# return self.exam_prob[rank if rank < self.TopK else -1]
		return self.exam_prob[rank if rank < min(len_examProb, self.TopK) else -1]
	

	def setTrustProb(self, initial_incorrect_click_p):
		self.initial_incorrect_click_p = initial_incorrect_click_p
		self.positive_trust_prob = [(98-k)/100 for k in range(self.TopK)]
		self.positive_trust_prob.append(0.0)
		self.negative_trust_prob = [self.initial_incorrect_click_p/(k+1) for k in range(self.TopK)]
		self.negative_trust_prob.append(0.0)

	def getTrustProb(self, rank):
		return self.positive_trust_prob[rank if rank < self.TopK else -1], self.negative_trust_prob[rank if rank < self.TopK else -1]

	def sampleClick(self, rank, relevance_label):
		if not relevance_label == int(relevance_label):
			print('RELEVANCE LABEL MUST BE INTEGER!')
		relevance_label = int(relevance_label) if relevance_label > 0 else 0
		posi_trust_p, nega_trust_p = self.getTrustProb(rank)
		exam_p = self.getExamProb(rank)
		click_p = self.click_prob[relevance_label if relevance_label < len(self.click_prob) else -1]
		observe = 1 if random.random() < exam_p else 0
		# click = 1 if random.random() < observe * click_p else 0
		click = 1 if random.random() < observe * (posi_trust_p * click_p + nega_trust_p * (1 - click_p)) else 0
		return click, observe, exam_p, click_p

	def sampleClicksForOneList(self, label_list):
		click_list,observe_list, exam_p_list, click_p_list = [], [], [], []
		for rank in range(len(label_list)):
			click, observe, exam_p, click_p = self.sampleClick(rank, label_list[rank])
			observe_list.append(observe)
			click_list.append(click)
			exam_p_list.append(exam_p)
			click_p_list.append(click_p)
		return click_list, observe_list, exam_p_list, click_p_list

	def estimatePropensityWeightsForOneList(self, click_list, use_non_clicked_data=False):
		propensity_weights = []
		for r in range(len(click_list)):
			pw = 0.0
			if use_non_clicked_data | click_list[r] > 0:
				pw = 1.0/self.getExamProb(r) * self.getExamProb(0)
			propensity_weights.append(pw)
		return propensity_weights

	def outputModelJson(self, file_name):
		json_dict = self.getModelJson()
		with open(file_name, 'w') as fout:
			fout.write(json.dumps(json_dict, indent=4, sort_keys=True))
		return None

	

if __name__ == '__main__':
	pass
	"""
	pbm = PositionBiasedModel()
	pbm.setExamProb(1.0)
	pbm.setClickProb(0.0, 1.0, 1)
	#pbm.outputModelJson('./pbm.json')
	click_log = []
	for i in range(1000):
		c_list,e_list,cp_list = pbm.sampleClicksForOneList([1,1,1,1,0,0,1,1,0,1,1,1])
		click_log.append(np.array(c_list))
	print(e_list)
	print(cp_list)
	print(sum(click_log)/1000)
	"""

	

	