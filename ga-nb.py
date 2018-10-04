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

	for i in range(n):
		Attr_nd.append(nodes(i))

	nof_attrs=n 

	apply_genetic_algo(nof_attrs,d,Attr_nd,class_i,output)


def apply_genetic_algo(nof_attrs,dataset,Attr_nd,class_i,output):

	
	population_size=30
	cross_over_rate=0.25
	mutation_rate=0.10

	chromosomes=initialize_chromosomes(population_size,nof_attrs)

	max_iterations=500
	accuracy_not_100=True
	iterations=0

	while iterations<max_iterations and accuracy_not_100 :

		Fitness=fitness_evaluate(population_size,chromosomes,dataset,class_i,Attr_nd,output,nof_attrs)

		chromosomes=Selection(Fitness,population_size,chromosomes)
		chromosomes=cross_over(cross_over_rate,chromosomes,population_size,nof_attrs)
		chromosomes=mutation(mutation_rate,chromosomes,nof_attrs,population_size)

		max_fitness=check_max_fitness(Fitness,population_size)
		if max_fitness==100:
			accuracy_not_100=False

		iterations+=1

	print("\n\nGenetic Algorithm\n")
	print("\nMaximum Fitness of Naive Byes Classifier :	",max_fitness)
	print("Achieved at ",iterations,"th iteration\n\nFor the chromosome :	\n",chromosomes[Fitness.index(max_fitness)])


def check_max_fitness(Fitness,population_size):

	max_fitness=Fitness[0]

	for i in range(population_size):
		if Fitness[i]>max_fitness:
			max_fitness=Fitness[i]

	return max_fitness

def mutation(mutation_rate,chromosomes,nof_attrs,population_size):

	new_chromosomes=chromosomes
	numbers=[i for i in range(population_size)]
	indices=[i for i in range(nof_attrs)]

	bits_for_mut=int(mutation_rate*nof_attrs*population_size)

	for i in range(bits_for_mut):
		cr=random.sample(numbers,1)[0]
		index=random.sample(indices,1)[0]

		if chromosomes[cr][index]==1:
			new_chromosomes[cr][index]=0
		else:
			new_chromosomes[cr][index]=1

	return new_chromosomes

def cross_over(cross_over_rate,chromosomes,population_size,nof_attrs):

	numbers=[i for i in range(population_size)]
	nof_cov=int(cross_over_rate*population_size)

	new_chromosomes=chromosomes

	chr_for_cov=random.sample(numbers,nof_cov)

	for i in range(nof_cov):
		chr_for_swap=random.sample(chr_for_cov,1)

		index1=chr_for_cov[i]
		index2=chr_for_swap[0]

		c1=chromosomes[index1]
		c2=chromosomes[index2]

		nof_cov_attr=int(cross_over_rate*nof_attrs)

		for i in range(0,nof_cov_attr):
			temp=c1[i]
			c1[i]=c2[i]
			c2[i]=temp
		new_chromosomes[index1]=c1
		new_chromosomes[index2]=c2

	return new_chromosomes

def Selection(Fitness,population_size,chromosomes):

	total_fitness=sum(Fitness)

	new_chromosomes=[]
	fitness_array=[]

	for i in range(population_size):

		fitness_array.append(fitness(Fitness[i],total_fitness))
		if i==0:
			fitness_array[i].cumulative_fitness=fitness_array[i].probability
		else:
			fitness_array[i].cumulative_fitness=fitness_array[i].probability+fitness_array[i-1].cumulative_fitness
		
		fitness_array[i].random_no=random.randint(0,1000)/1000.0

	for i in range(population_size):
		new_chromosomes.append(chromosomes[find_successive(i,fitness_array,population_size)])

	return new_chromosomes

def find_successive(index,fitness_array,population_size):

	random_no=fitness_array[index].random_no
	cuml_fitness=None
	new_index=index

	for i in range(population_size):
		if cuml_fitness!=None:
			if fitness_array[i].cumulative_fitness>random_no and fitness_array[i].cumulative_fitness<cuml_fitness:
				cuml_fitness=fitness_array[i].cumulative_fitness
				new_index=i 
		else:
			if fitness_array[i].cumulative_fitness>random_no:
				cuml_fitness=fitness_array[i].cumulative_fitness
				new_index=i 

	return new_index

class fitness:
	def __init__(self,present_fitness,total_fitness):

		self.probability=present_fitness/total_fitness
		self.cumulative_fitness=None
		self.random_no=None
		self.selected_chromosome=None

def fitness_evaluate(population_size,chromosomes,dataset,class_i,Attr_nd,output,nof_attrs):

	Fitness=[]
	training_dataset=dataset[: int(len(dataset) * 0.01*10*8)]
	test_dataset=dataset[int(len(dataset) * 0.01*10*8):]

	for i in range(population_size):

		Attr_nd=apply_naive_bayes(nof_attrs,training_dataset,class_i,Attr_nd,output)
		Fitness.append(find_accuracy(chromosomes[i],test_dataset,Attr_nd,class_i,output,nof_attrs))

	return Fitness


def initialize_chromosomes(population_size,nof_attrs):

	chromosomes=[[] for i in range(population_size)]

	for i in range(population_size):
		for j in range(nof_attrs): 
			chromosomes[i].append(random.randint(0,1)) 

	return chromosomes


def list_difference(list1, list2):
	
	diff_list = []
	for item in list1:
		if not item in list2:
			diff_list.append(item)
	return diff_list

def find_accuracy(chromosome,test_dataset,attr,ci,op,n):

	prob_p=attr.pop()
	prob_n=attr.pop()
	acc=0

	for i in test_dataset:
		prob_positive=prob_p
		prob_negative=prob_n
		k=0

		for j in range(len(i)):
			if j!=ci and chromosome[k]!=0:
				if i[j]==0:
					prob_positive*=attr[k].zero_positive 
					prob_negative*=attr[k].zero_negative
				else:
					prob_positive*=attr[k].one_positive 
					prob_negative*=attr[k].one_negative

				k+=1

		a_op=op.index(i[ci])

		if prob_positive>prob_negative:
			p_op=1
			if a_op==1:
				acc+=1

		else:
			p_op=0 
			if a_op==0:
				acc+=1

	acc=(acc/len(test_dataset))*100

	return acc 
	
def apply_naive_bayes(n,dataset,ci,attr,op):

	prob_positive=0
	prob_negative=0
	for i in dataset:
		k=0
		for j in range(len(i)):
			if j!=0:

				aop=op.index(i[0])
				if aop==0:
					if i[j]==0:
						attr[k].zero_negative+=1
					else:
						attr[k].one_negative+=1
				else:
					if i[j]==0:
						attr[k].zero_positive+=1
					else:
						attr[k].one_positive+=1
				k+=1
			else:
				g=op.index(i[ci])
				if g==0:
					prob_negative+=1
				else:
					prob_positive+=1

	attr.append(prob_negative)
	attr.append(prob_positive)

	return attr 

class nodes:

	def __init__(self,a):
		self.attr=a 
		self.zero_positive=0
		self.one_positive=0 
		self.zero_negative=0 
		self.one_negative=0


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
