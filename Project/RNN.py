import numpy as np 
import math

def Weight_Update(X,Y,Y_p,Whx , Whh, Wyh, bh, by,learning_rate):

	k=len(Y)
	l=len(Y[0])

	cross_entropy=[None for i in range(k)]

	L_values=Calculate_L_values(Y,Y_p)


	Derivatives=Calculate_Derivatives(X,Y,Y_p,Whx , Whh, Wyh, bh, by,L_values)

	Whx , Whh, Wyh, bh, by=Add_Derivatives(Whx , Whh, Wyh, bh, by,learning_rate,Derivatives)

	return Whx , Whh, Wyh, bh, by


def Calculate_Derivatives(X,Y,Y_p,Whx , Whh, Wyh, bh, by,L_values,no_hidden):

	Derivatives=[None for i in range(5)]

	Derivatives[0]=np.zeros[Whx.shape]
	Derivatives[1]=np.zeros[Whh.shape]
	Derivatives[2]=np.zeros[Wyh.shape]
	Derivatives[3]=np.zeros[bh.shape]
	Derivatives[4]=np.zeros[by.shape]

	k=len(Y)
	l=len(Y[0])
	n=k
	m=no_hidden

	H=[None for i in range(k)]
	T=[None for i in range(k)] 
	S=[None for i in range(k)]

	for i in range(k):

		if i==0:
			T[i]=np.matmul(Whx[i],X[i])+bh[i]

		else:
			T[i]=np.matmul(Whx[i],X[i])+np.matmul(Whh[i],H[i-1])+bh[i]

		H[i]=sigmoid(T[i]) 
		S[i]=Wyh[i]*H[i]+by[i]

		dl_whx=0
		dl_whh=0
		dl_why=0
		dl_bh=0
		dl_by=0
			
		for j in range(l):

			k=S[i]-math.exp(X[i][j])
			dl_dy=Y[i][j]/Y_p[i][j]+(1-Y[i][j])/(Y_p[i][j]-1)
			dy_ds=math.exp(S[i])/(k*(1+k*math.exp(-S[i])**2))

			dhi_dti=1/math.exp(T[i])*((1+math.exp(-T[i]))**2)

			ds_whx=Wyh[j][i%m]*dhi_dti*X[i][j]

			if i!=0:
				ds_whh=Wyh[j][i%m]*dhi_dti*H[i-1]
			else:
				ds_whh=0

			ds_wyh=H[i]
			ds_by=1
			ds_bh=Wyh[j][i%m]*dhi_dti


			dl_whx+=dl_dy*dy_ds*ds_whx
			dl_whh+=dl_dy*dy_ds*ds_whh
			dl_why+=dl_dy*dy_ds*ds_why 

			dl_bh+=dl_dy*dy_ds*ds_bh
			dl_by+=dl_dy*dy_ds*ds_by


		#Whx part

		if i<m:
			for temp in range(n):
				Derivatives[0][i][temp]=dl_whx

		if i<m:
			for temp in range(m):
				Derivatives[1][i][temp]=dl_whh 

		if i<l:
			for temp in range(m):
				Derivatives[2][i][temp]=dl_wyh 

		Derivatives[3][i]=dl_bh
		Derivatives[4][i]=dl_by


	return Derivatives

def Add_Derivatives(Whx , Whh, Wyh, bh, by,learning_rate,Derivatives):

	# T=T*learning_rate+Derivative

	Whx=np.add(learning_rate*np.array(Whx),Derivatives[0])
	Whh=np.add(learning_rate*np.array(Whh),Derivatives[1])
	Wyh=np.add(learning_rate*np.array(Wyh),Derivatives[2])
	bh=np.add(learning_rate*np.array(bh),Derivatives[3])
	by=np.add(learning_rate*np.array(by),Derivatives[4])

	return Whx , Whh, Wyh, bh, by

def Calculate_L_values(Y,Y_p):


	'''L(yi,ˆyi)← - sigma(i){	
								sigma(j){ 
											yij*log(yˆij)+ (1 − yij) log(1 −ˆyij)
										}
							}
	'''

	k=len(Y)
	l=len(Y[0])

	cross_entropy=[None for i in range(k)]

	L_values=[None for i in range(k)]
	#Calculating the L values for all iterations
	for i in range(k):

		#initialising the Lsum
		if i==0:
			prev_L=0

		L_value=0

		#L value in current iteration only
		for j in range(l):
			L_value+=Y[i][j]*math.log(Y_p[i][j])+(1-Y[i][j])*math.log(1-Y_p[i][j])

		#L value for iteration i
		L_values[i]=prev_L-L_value

		prev_L=L_value

	return L_values

def sigmoid(x):

	return 1/(1+math.exp(-x))
