import pickle
import numpy as np
from scipy.sparse import csr_matrix
from Params import args
import scipy.sparse as sp
from Utils.TimeLogger import log
from time import time

def transpose(mat):
	coomat = sp.coo_matrix(mat)
	return csr_matrix(coomat.transpose())

def negSamp(temLabel, sampSize, nodeNum):
	negset = [None] * sampSize
	cur = 0
	while cur < sampSize:
		rdmItm = np.random.choice(nodeNum)
		if temLabel[rdmItm] == 0:
			negset[cur] = rdmItm
			cur += 1
	return negset

def transToLsts(mat, mask=False, norm=False):
	shape = [mat.shape[0], mat.shape[1]]
	coomat = sp.coo_matrix(mat)
	indices = np.array(list(map(list, zip(coomat.row, coomat.col))), dtype=np.int32)
	data = coomat.data.astype(np.float32)

	if norm:
		rowD = np.squeeze(np.array(1 / (np.sqrt(np.sum(mat, axis=1) + 1e-8) + 1e-8)))
		colD = np.squeeze(np.array(1 / (np.sqrt(np.sum(mat, axis=0) + 1e-8) + 1e-8)))
		for i in range(len(data)):
			row = indices[i, 0]
			col = indices[i, 1]
			data[i] = data[i] * rowD[row] * colD[col]

	if mask:
		spMask = (np.random.uniform(size=data.shape) > 0.5) * 1.0
		data = data * spMask

	if indices.shape[0] == 0:
		indices = np.array([[0, 0]], dtype=np.int32)
		data = np.array([0.0], np.float32)
	return indices, data, shape

class DataHandler:
	def __init__(self):
		if args.data == 'tmall':
			predir = './Datasets/Tmall/'
			behs = ['pv', 'fav', 'cart', 'buy']
		elif args.data == 'beibei':
			predir = './Datasets/beibei/'
			behs = ['pv', 'cart', 'buy']
		elif args.data == 'ijcai':
			predir = './Datasets/ijcai/'
			behs = ['click', 'fav', 'cart', 'buy']
		elif args.data == 'yelp':
			predir = './Datasets/Yelp/'
			behs = ['tip', 'neg', 'neutral', 'pos']

		self.predir = predir
		self.behs = behs
		self.trnfile = predir + 'trn_'
		self.tstfile = predir + 'tst_'
		self.adj_file = predir + 'adj_'

	def LoadData(self):
		trnMats = list()
		for i in range(len(self.behs)):
			beh = self.behs[i]
			path = self.trnfile + beh
			with open(path, 'rb') as fs:
				mat = (pickle.load(fs) != 0).astype(np.float32)
			trnMats.append(mat)
		path = self.tstfile + 'int'
		with open(path, 'rb') as fs:
			tstInt = np.array(pickle.load(fs))
		tstStat = (tstInt != None)
		tstUsrs = np.reshape(np.argwhere(tstStat != False), [-1])

		self.trnMats = trnMats
		self.tstInt = tstInt
		self.tstUsrs = tstUsrs
		args.user, args.item = self.trnMats[0].shape
		args.behNum = len(self.behs)


	def get_adj_mat(self):
		ori_adj, left_loop_adj, left_adj, symm_adj = [], [], [], []
		try:
			t1 = time()
			for i in range(args.behNum):
				beh = self.behs[i]
				path = self.adj_file + beh
				ori_adj_mat = sp.load_npz(path + '_ori.npz')
				norm_adj_mat = sp.load_npz(path + '_norm_.npz')
				mean_adj_mat = sp.load_npz(path + '_mean.npz')
				ori_adj.append(ori_adj_mat)
				left_loop_adj.append(norm_adj_mat)
				left_adj.append(mean_adj_mat)

				print('already load adj matrix', ori_adj_mat.shape, time() - t1)

		except Exception:
			for i in range(args.behNum):
				beh = self.behs[i]
				path = self.adj_file + beh
				ori_adj_mat, norm_adj_mat, mean_adj_mat = self.create_adj_mat(self.trnMats[i])
				sp.save_npz(path + '_ori.npz', ori_adj_mat)
				sp.save_npz(path + '_norm_.npz', norm_adj_mat)
				sp.save_npz(path + '_mean.npz', mean_adj_mat)
				ori_adj.append(ori_adj_mat)
				left_loop_adj.append(norm_adj_mat)
				left_adj.append(mean_adj_mat)
				print('already load adj matrix', ori_adj_mat.shape, time() - t1)

		try:
			for i in range(args.behNum):
				beh = self.behs[i]
				path = self.adj_file + beh
				pre_adj_mat = sp.load_npz(path + '_pre.npz')
				symm_adj.append(pre_adj_mat)

		except Exception:
			for i in range(args.behNum):
				beh = self.behs[i]
				path = self.adj_file + beh

				rowsum = np.array(ori_adj_mat.sum(1))
				d_inv = np.power(rowsum, -0.5).flatten()
				d_inv[np.isinf(d_inv)] = 0.
				d_mat_inv = sp.diags(d_inv)

				norm_adj = d_mat_inv.dot(ori_adj_mat)
				norm_adj = norm_adj.dot(d_mat_inv)
				print('generate pre adjacency matrix.')
				pre_adj_mat = norm_adj.tocsr()
				sp.save_npz(path + '_pre.npz', pre_adj_mat)
				symm_adj.append(pre_adj_mat)
		return ori_adj, left_loop_adj, left_adj, symm_adj

	def create_adj_mat(self, which_R):
		t1 = time()
		adj_mat = sp.dok_matrix((args.user + args.item, args.user + args.item), dtype=np.float32)
		adj_mat = adj_mat.tolil()
		R = which_R.tolil()

		adj_mat[:args.user, args.user:] = R
		adj_mat[args.user:, :args.user] = R.T
		adj_mat = adj_mat.todok()
		print('already create adjacency matrix', adj_mat.shape, time() - t1)

		t2 = time()

		def normalized_adj_single(adj):
			rowsum = np.array(adj.sum(1))

			d_inv = np.power(rowsum, -1).flatten()
			d_inv[np.isinf(d_inv)] = 0.
			d_mat_inv = sp.diags(d_inv)

			norm_adj = d_mat_inv.dot(adj)
			print('generate single-normalized adjacency matrix.')
			return norm_adj.tocoo()

		def check_adj_if_equal(adj):
			dense_A = np.array(adj.todense())
			degree = np.sum(dense_A, axis=1, keepdims=False)

			temp = np.dot(np.diag(np.power(degree, -1)), dense_A)
			print('check normalized adjacency matrix whether equal to this laplacian matrix.')
			return temp

		norm_adj_mat = normalized_adj_single(adj_mat + sp.eye(adj_mat.shape[0]))
		mean_adj_mat = normalized_adj_single(adj_mat)

		print('already normalize adjacency matrix', time() - t2)
		return adj_mat.tocsr(), norm_adj_mat.tocsr(), mean_adj_mat.tocsr()
