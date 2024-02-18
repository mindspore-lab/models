import numpy as np
import matplotlib

MAX = 1000

def UnderSample(vec, default):
	l = vec.shape[0]
	window = int(np.ceil(l / MAX))
	newL = int(np.ceil(l / window))
	ret = np.zeros(newL)
	for i in range(newL):
		temvec = vec[i*window: min((i+1)*window, l)]
		deno = len(temvec) if default == None else np.sum(temvec!=default)
		ret[i] = np.sum(temvec) / deno
	return ret

def PlotAMat(mat):
	w, l = mat.shape
	s = max(int(l / 40), 1)
	X = [[None]*(s*l) for i in range(s*w)]
	for i in range(w):
		for j in range(s):
			for k in range(l):
				for f in range(s):
					X[i*s+j][k*s+f] = mat[i][k]
	return X

def PlotAVec(vec, default=None):
	l = vec.shape[0]
	if l > MAX:
		print('Undersampling...', l, MAX)
		vec = UnderSample(vec, default)
	l = vec.shape[0]
	w = max(int(l / 40), 1)
	X = [None] * w
	for i in range(w):
		X[i] = vec
	return X


cmap = 'Spectral'#Wistia

def PlotMats(mats, default=None, titles=None, show=True, savePath=None, vrange=None):
	if not show:
		matplotlib.use('Agg')
	from matplotlib import pyplot as plt
	plt.figure(figsize=(20, 15))
	n = mats.shape[0]
	for i in range(n):
		ax = plt.subplot(n//4, 4, i+1)
		# ax = plt.subplot(1, 1, i+1)
		if titles != None:
			ax.set_title(str(titles[i]))
		X = PlotAMat(mats[i])
		if vrange != None:
			plt.imshow(X, vmin=vrange[0], vmax=vrange[1], cmap=cmap)
		else:
			plt.imshow(X, cmap=cmap)
		plt.yticks([])
		plt.sca(ax)
	if show:
		figManager = plt.get_current_fig_manager()
		figManager.window.showMaximized()
		plt.show()
	if savePath:
		plt.savefig(savePath)

def PlotVecs(vecs, default=None, titles=None, show=True,
	savePath=None, vrange=None):
	if not show:
		matplotlib.use('Agg')
	from matplotlib import pyplot as plt
	plt.figure(figsize=(20, 15))
	n = vecs.shape[0]
	for i in range(n):
		ax = plt.subplot(n/4, 4, i+1)
		if titles != None:
			ax.set_title(str(titles[i]))
		X = PlotAVec(vecs[i])
		if vrange != None:
			plt.imshow(X, vmin=vrange[0], vmax=vrange[1], cmap=cmap)
		else:
			plt.imshow(X, cmap=cmap)
		plt.yticks([])
		plt.sca(ax)
	if show:
		figManager = plt.get_current_fig_manager()
		figManager.window.showMaximized()
		plt.show()
	if savePath:
		plt.savefig(savePath)

# from matplotlib import pyplot as plt
# l = 1000
# w = max(int(l / 40), 1)
# X = [None] * w
# labela = list()
# labelb = list()
# # tem = np.random.rand(l)
# print('1')
# for i in range(w):
# 	X[i] = [0] * (l+1)
# 	for j in range(l+1):
# 		X[i][j] = j/l
# 		if j % 100 == 0:
# 			labela.append(j)
# 			labelb.append(str(j/l))
# 	# X[i] = tem
# print('2')
# plt.tick_params(labelsize=24)
# plt.imshow(X)
# plt.yticks([])
# plt.xticks(labela, labelb)
# plt.show()
