import csv
import numpy as np 
import random

def main():

	print('\nThe file is SPECT.csv	\n')

	
	f_name='SPECT.csv'

	dataset=load_file(f_name)
	head=dataset.pop(0)

	class_i=class_index(head)
	d=dataset
	random.shuffle(d)
	
	n=len(head)-1

	dataset=alter_datatypes(dataset,class_i)
	output=categorize_op(dataset,class_i)

	Attr_nd=[]
	acc=[]

	for i in range(n):
		Attr_nd.append(nodes(i))

	for i in range(10):
		
		dataset=d[int(len(d) * 0.01*10*i): int(len(d) * 0.01*10*(i+1))]
		Attr_nd=apply_naive_bayes(n,dataset,class_i,Attr_nd,output)
		test_dataset=list_difference(d,dataset)
		acc.append(find_accuracy(dataset,test_dataset,Attr_nd,class_i,output,n))

	x=np.array(acc)
	facc=x.mean(axis=0)

	print('\nThe k-fold accuracy is',facc[0],'\n')
	print('\nThe precision is 	',facc[1],'\n')
	print('\nThe recall is 		',facc[2],'\n')

def list_difference(list1, list2):
    
    diff_list = []
    for item in list1:
        if not item in list2:
            diff_list.append(item)
    return diff_list

def find_accuracy(dataset,test_dataset,attr,ci,op,n):

	prob_p=attr.pop()
	prob_n=attr.pop()
	acc=0
	tp=0
	fp=0
	fn=0

	for i in test_dataset:
		#for j in i:
		pr_p=prob_p
		pr_n=prob_n
		k=0
		for j in range(len(i)):
			if j!=ci:
				if i[j]==0:
					pr_p*=attr[k].op 
					pr_n*=attr[k].on
				else:
					pr_p*=attr[k].lp 
					pr_n*=attr[k].ln

				k+=1

		a_op=op.index(i[ci])

		if pr_p>pr_n:
			p_op=1
			if a_op==1:
				acc+=1
				tp+=1
			else:
				fp+=1

		else:
			p_op=0 
			if a_op==1:
				fn+=1
			if a_op==0:
				acc+=1

	print(acc,fp,tp,fn)

	acc=(acc/len(test_dataset))*100
	if tp+fp!=0:
		pr=(tp/(tp+fp))*100
	else:
		pr=0

	if tp+fn!=0:
		re=(tp/(tp+fn))*100
	else:
		re=0


	fop=[]
	fop.append(acc)
	fop.append(pr)
	fop.append(re) 

	return fop
	
def apply_naive_bayes(n,dataset,ci,attr,op):

	pt=0
	pn=0
	for i in dataset:
		k=0
		for j in range(len(i)):
			if j!=0:

				aop=op.index(i[0])
				if aop==0:
					if i[j]==0:
						attr[k].on+=1
					else:
						attr[k].ln+=1
				else:
					if i[j]==0:
						attr[k].op+=1
					else:
						attr[k].lp+=1
				k+=1
			else:
				g=op.index(i[ci])
				if g==0:
					pn+=1
				else:
					pt+=1

	attr.append(pn)
	attr.append(pt)

	return attr 

class nodes:

	def __init__(self,a):
		self.attr=a 
		self.op=0
		self.lp=0 
		self.on=0 
		self.ln=0


def categorize_op(dataset,ci):

	diff=[]
	for i in dataset:
		if not i[ci] in diff:
			diff.append(i[ci])
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
