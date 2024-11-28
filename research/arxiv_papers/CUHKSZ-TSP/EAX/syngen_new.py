#!/sur/bin/python
#generate the file of syntexe

import os
import math
import cmath
import sys
import getopt
import shutil

shutil.rmtree('summary_new')
os.mkdir('summary_new')

dict_bks={'E10k.0':71865826, 'E10k.1':72031630, 'E10k.2':71822483,
'E31k.0':127281803, 'E31k.1':127452384, 'E100k.0':225783795,'E100k.1':225653450,
'E316k.0':401294850, 'E1M.0':713176566,
'C10k.0': 33001034, 'C10k.1': 33186248, 'C10k.2': 33155424,
'C31k.0': 59545390, 'C31k.1': 59293266,
'C100k.0': 104617752, 'C100k.1': 105385439, 'C316k.0': 186834550
}

def usage():
	print """'-p'+number of rotation degrees
for exemple: -p 30"""


def get_one(inname):
	with open (inname, 'r') as frr:
		fl=frr.readline()
		fl=fl.strip().split()
	return fl[1], fl[2], fl[3]


def get_res(insname, num):
	bst=[]
	bstt=[]
	prt=[]
	for i in range(num):
		tmpfile="Sol/"+insname+'F%d'%i
		if(os.path.exists(tmpfile)==False): continue
		tmpr, tmprt, tmpt=get_one(tmpfile)
		bst.append(float(tmpr))
		prt.append(float(tmprt))
		bstt.append(float(tmpt))
	print bst

	return bst,prt,bstt


def get_bst(bst):
	bb=bst[0]
	for i in range(len(bst)):
		if(bst[i]<bb):
			bb=bst[i]
	return bb


def get_avg(ll):
	d=0
	for i in range(len(ll)):
		d+=ll[i]
	return d/len(ll)

def fectch_opt(insname):

	return dict_bks[insname]

	tmpfile="Opt/"+insname+'.opt'
	with open (tmpfile, 'r') as frr:
		fl=frr.readline()
		fl=int(fl.strip())
	return fl

def get_sum(ll):
	d=0
	for i in range(len(ll)):
		d+=ll[i]
	return d

def get_smallest(ll):
	d=ll[0]
	for i in range(len(ll)):
		if(ll[i]<d):
			d=ll[i]
	return d
numfile=10

with open("summary_new/Result.txt",'w') as of:
	# of.write("{0:<20} {1:^15} {2:^15} {3:^15} {4:^15} {5:^10} {6:^10} {7:^10}\n".format("Ins", "Opt", "Bst", 'Avg', "PrT", "Tall", "Delta1", "Delta2"))
	avg_opt=0.0
	avg_res=0.0
	avg_tim=0.0
	avg_delta=0.0
	ct=0
	ct_good=0
	with open("fn2.txt",'r') as fr:
		data=fr.readlines()

#		print data
		for iter, line in enumerate(data):
			tmpl=line.strip()
#			print line
			opt_i= fectch_opt(tmpl)

			bst,prt,bstt= get_res(tmpl, numfile)
			if(len(bst)==0): continue

			with open ('summary_new/'+tmpl+'.res', 'w') as ffw:
				for i in range(len(bst)):
					ffw.write('%s, %s\n'%(bst[i], bstt[i]))


			bb=get_bst(bst)
			bavg=get_avg(bst)

			prt_small=get_smallest(prt)
			othert_all=get_sum(bstt)
			# othert_all=get_sum(bstt)-get_sum(prt)
			# bt=get_avg(bstt)

			if iter == 0:
				of.write("instance              best-known            best                average       preprecessing-time    all-time   gap-best2bbkv   gap-average2bkv\n")
			of.write("{0:<20} {1:^15} {2:^15.2f} {3:^15.2f}      {4:^15.2f}            {5:^10.2f}     {6:^10.2f}                 {7:^10.2f}\n".format(tmpl, opt_i, bb, bavg, prt_small, othert_all, (bb-opt_i)/opt_i*100, (bavg-opt_i)/opt_i*100))


			# ress=bb
			# rest=bt
			# # of.write("{0:<20} {1:^15} {2:^15.2f} {3:^15.2f} {4:^15.2f} {5:^10.2f}\n".format(tmpl, opt_i, ress, bavg, bt ,(ress-opt_i)/opt_i*100))
			# avg_opt=avg_opt+opt_i
			# avg_res=avg_res+ress
			# avg_tim=avg_tim+rest
			# avg_delta=avg_delta+(ress-opt_i)/opt_i*1000
			# ct=ct+1
			# if(ress-opt_i<0.01): ct_good+=1
