import csv
import numpy as np 
import random

def main():

	f_no=int(input('Choose the file :\n1. IRIS.csv\n2. SPECT.csv	\n\n'))

	if f_no==1:
		f_name='IRIS.csv'
	else:
		f_name='SPECT.csv'

	dataset=load_file(f_name)
	head=dataset.pop(0)

	class_i=class_index(head)

	dataset=alter_datatypes(dataset,class_i)
	output=categorize_op(dataset,class_i)

	final_dataset=change_class(dataset,output,class_i)

	n=len(head)-1
	d=final_dataset
	random.shuffle(d)

	k=int(input('\nEnter the K value for KNN algorithm:	'))
	acc=[]

	for i in range(10):
			
		test_dataset=d[int(len(d) * 0.01*10*i): int(len(d) * 0.01*10*(i+1))]
		dataset=list_difference(d,test_dataset)
		acc.append(apply_knn(dataset,test_dataset,class_i,k))

	x=np.array(acc)
	facc=x.mean(axis=0)

	print('\nThe k-fold accuracy is 		',facc[0],'\n')
	print('\nThe precision-positive is 	',facc[1],'\n')
	print('\nThe precision-negative is 	',facc[2],'\n')
	print('\nThe recall-positive is 		',facc[3],'\n')
	print('\nThe recall-negative is 		',facc[4],'\n')

def list_difference(list1, list2):
    
    diff_list = []
    for item in list1:
        if not item in list2:
            diff_list.append(item)
    return diff_list


def apply_knn(dataset,test_dataset,ci,k):

	total=len(test_dataset)
	acc=0
	tp=0
	tn=0
	fp=0
	fn=0

	for i in test_dataset:
		distances=[]

		for j in dataset:
			distances.append(dist_bw(i,j,ci,dataset.index(j)))

		distances.sort(key=lambda x:x.distance)
		nearest_neighbours=distances[:k]

		exp_op=find_majority(nearest_neighbours,dataset,ci)
		a_op=i[ci]

		if exp_op==1:
			if a_op==1:
				acc+=1
				tp+=1
			else:
				fp+=1

		elif exp_op==0:
			if a_op==1:
				fn+=1
			if a_op==0:
				acc+=1
				tn+=1

	acc=(acc/len(test_dataset))*100
	if tp+fp!=0:
		pr_p=(tp/(tp+fp))*100
	else:
		pr_p=0

	if tn+fn!=0:
		pr_n=(tn/(tn+fn))*100
	else:
		pr_n=0

	if tp+fn!=0:
		re_p=(tp/(tp+fn))*100
	else:
		re_p=0

	if tn+fp!=0:
		re_n=(tn/(tn+fp))*100
	else:
		re_n=0


	fop=[]
	fop.append(acc)
	fop.append(pr_p)
	fop.append(pr_n)
	fop.append(re_p)
	fop.append(re_n) 

	return fop

def find_majority(nearest_neighbours,dataset,ci):

	count_0=0
	count_1=0
	for i in nearest_neighbours:
		if dataset[i.index][ci]==0:
			count_0+=1
		else:
			count_1+=1

	if count_0>count_1:
		return 0

	return 1

def dist_bw(test,train,ci,ind):

	sums=0
	for i in range(len(test)):
		if i!=ci:
			sums+=abs(test[i]**2-train[i]**2)

	d=distances()
	d.index=ind 
	d.distance=sums**0.5

	return d 

class distances:
	def __init__(self):
		self.index=None 
		self.distance=None

def change_class(dataset,output,ci):

	for i in range(len(dataset)):
		dataset[i][ci]=output.index(dataset[i][ci])
	return dataset

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
