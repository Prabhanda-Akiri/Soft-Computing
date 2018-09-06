import csv
import math
from statistics import mean 
import numpy as np 

def main():

	f_no=int(input('Choose the file :\n1. IRIS.csv\n2. SPECT.csv	\n\n'))

	if f_no==1:
		f_name='IRIS.csv'
	else:
		f_name='SPECT.csv'

	dataset=load_file(f_name)
	head=dataset.pop(0)
	#print(head)
	class_i=class_index(head)

	n=len(head)-1


	l_rate=float(input('\nEnter the learning rate:	'))
	#div=int(input('\nEnter % of data for '))

	dataset=alter_datatypes(dataset,class_i)
	output=categorize_op(dataset,class_i)
	
	d=dataset 
	dataset=d[: int(len(d) * 0.80)]
	fop=apply_multi_layer(n,dataset,l_rate,class_i)
	test_dataset=d[int(len(d) * 0.80):]
	find_accuracy(test_dataset,fop,class_i,output,n)
	#print(dataset)'''
	acc=[]

	for i in range(10):
		
		test_dataset=d[int(len(d) * 0.01*10*i): int(len(d) * 0.01*10*(i+1))]
		dataset=list_difference(d,test_dataset)
		fop=apply_multi_layer(n,dataset,l_rate,class_i)		
		acc.append(find_accuracy(test_dataset,fop,class_i,output,n))

	x=np.array(acc)
	facc=x.mean(axis=0)

	print('\nThe k-fold accuracy is',facc[0],'\n')
	print('\nThe precision is 	',facc[1]*100,'\n')
	print('\nThe recall is 		',facc[2]*100,'\n')

def list_difference(list1, list2):
    """uses list1 as the reference, returns list of items not in list2"""
    diff_list = []
    for item in list1:
        if not item in list2:
            diff_list.append(item)
    return diff_list

def find_accuracy(d,f,ci,output,n):
	n_ip=n 
	n_op=1
	n_hl=5
	m=n+1+5

	dataset=d 
	w_ip=f[0]
	w_hl=f[1]
	th=f[2]
	count=0
	tp=0
	fp=0
	fn=0

	for j in dataset:

		a=dataset.index(j)
		ip=[None for k in range(m)]
		op=[None for k in range(m)]
		o=0
		for k in range(n+1):
			if k!=ci:
				op[o]=j[k]
				o+=1

		for k in range(n,m):
			ip[k]=find_ij(w_ip,w_hl,k,n_ip,op,th)
			op[k]=find_oj(ip[k])

		if op[m-1]<=0.5:
			pre_op=0
		else:
			pre_op=1

		act_op=output.index(j[ci])

		if pre_op==act_op:
			count+=1
			if pre_op==1:
				tp+=1
		if pre_op==1 and act_op==0:
			fp+=1 
		if pre_op==0 and act_op==1:
			fn+=1

	#print(count,len(d))

	x=[]
	acc=float((count)*1.0/len(dataset))*100
	x.append(acc)
	if tp+fp!=0:
		pr=float(tp/(tp+fp))
	else:
		pr=0
	x.append(pr)
	
	if tp+fn!=0:
		re=float(tp/(tp+fn))
	else:
		re=0
	x.append(re)

	return x
	#print('The accuracy is:	',acc,'%')

def apply_multi_layer(n,dataset,lr,ci):

	n_ip=n 
	n_op=1
	n_hl=5
	m=n+1+5
	w_ip=[[1/(n_ip*n_hl) for i in range(n_ip)] for j in range(n_hl)]
	w_hl=[1/(n_hl*n_op) for i in range(n_hl)]

	
	th=[None for i in range(m)]
	er=[None for i in range(m)]

	for i in range(n,m):
		th[i]=1/(n_hl+1)

	for i in range(1000):
		for j in dataset:

			a=dataset.index(j)
			ip=[None for k in range(m)]
			op=[None for k in range(m)]
			o=0
			for k in range(n+1):
				if k!=ci:
					op[o]=j[k]
					o+=1

			for k in range(n,m):
				ip[k]=find_ij(w_ip,w_hl,k,n_ip,op,th)
				op[k]=find_oj(ip[k])

			for k in range(m-1,n-1,-1):
				er[k]=find_error(op[k],m-1,k,er,w_hl,n)


			for k in range(n_hl):
				for o in range(n_ip):
					dw=find_dw(lr,er[n+k],op[o])
					w_ip[k][o]+=dw 

			for k in range(n_hl):
				dw=find_dw(lr,er[m-1],op[n+k])
				w_hl[k]+=dw 

			for k in range(n,m):
				th[k]=lr*er[k]

	fop=[]
	fop.append(w_ip)
	fop.append(w_hl)
	fop.append(th)

	return fop

def find_dw(lr,erj,oi):
	return lr*erj*oi;

def find_error(oj,m,k,er,w,n):

	if m==k:
		return oj*(1-oj)*(1-oj)
	else:
		b=oj*(1-oj)+er[m]*w[k-n]
		return b

def find_oj(ij):

	oj=1/(1+(math.exp(-ij)))

	return oj 

def find_ij(w_ip,w_hl,k,n,ip,th):

	sums=0
	#print(k)

	if k<n+5:
		for i in range(n):
			sums+=w_ip[k-n][i]*ip[i]
	else:
		for i in range(5):
			sums+=w_hl[i]*ip[i]

	sums+=th[k]

	return sums
def categorize_op(dataset,ci):
	diff=[]
	for i in dataset:
		if not i[ci] in diff:
			diff.append(i[ci])
	if diff[0]=='Yes':
		diff.reverse()
	return diff

def alter_datatypes(dataset,ci):

	for i in dataset:
		for j in i:

			a=dataset.index(i)
			b=i.index(j)

			if b==ci:
				continue
			else:
				dataset[a][b]=float(j)

	return dataset

def class_index(head):

	class_i=0

	for i in head:
		if i=='class':
			class_i=head.index(i)

	return class_i

def load_file(f_name):

	data=[]
	with open(f_name, 'r') as f:
		reader = csv.reader(f)
		for row in reader:
			if not row:
				continue
			data.append(row)

	return data

if __name__ == '__main__':
	main()
