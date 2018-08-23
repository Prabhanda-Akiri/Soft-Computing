import csv

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

	weights=[1/(n+1) for i in range(n+1)]

	l_rate=float(input('\nEnter the learning rate:	'))

	dataset=alter_datatypes(dataset,class_i)
	output=categorize_op(dataset,class_i)

	apply_learning_method(dataset,weights,class_i,l_rate,output)

def expected_op(row,w,ci):
	op=0 

	for i in range(len(row)):
		if i!=ci:
			op+=row[i]*w[i]

	if op>0:
		return 1

	return 0

def apply_learning_method(dataset,w,ci,lr,op):

	accu=[]
	for i in range(1500):

		for j in dataset:

			eop=expected_op(j,w,ci)
			z=op.index(j[ci])

			if eop!=z:
				w=correct_wt(lr,w,z-eop,j,ci)		
		l=[]
		l.append(w)
		p_acc=get_accuracy(dataset,w,ci,op)
		#print(p_acc)		
		l.append(p_acc)

		if i==0:			
			accu.append(l)
		else:
			if p_acc>accu[-1][1]:
				accu.append(l)	
	
		if int(p_acc)!=100:
			continue
		else:
			break 
	
	maxm=accu[len(accu)-1]

	print('\nThe final weights are:	\n',maxm[0])

	print('\nAccuracy is:	',maxm[1],'%\n')

def get_accuracy(dataset,w,ci,op):

	cop=0
	for i in dataset:

		eop=expected_op(i,w,ci)
		z=op.index(i[ci])

		if eop==z:
			cop+=1

	return (cop*100)/len(dataset)

def correct_wt(lr,w,x,row,ci):

	for i in range(len(row)):
		if i!=ci:
			dw=lr*x*row[i]
			w[i]+=dw 

	return w 

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
