import sys
import random
import math
w0 = 0
#Name = Manish Patel mp933

#### FUNCTIONS #########
###################
#### Supplementary function to compute dot product
###################
def dotproduct(u, v):
	assert len(u) == len(v), "dotproduct: u and v must be of same length"
	dp = 0
	for i in range(0, len(u), 1):
		dp = dp + u[i] * float(v[i])
	return dp

###################
## Standardize the code here: divide each feature of each 
## datapoint by the length of each column in the training data
## return [traindata, testdata]
###################
def standardize_data(traindata,testdata):
	mean = []
	sd = []
	for i in range(len(traindata[0])):
		m = 0
		numr = 0
		
		for j in range(len(traindata)):
			m += traindata[j][i]
		mean.append(m/len(traindata))
		
		for j in range(len(traindata)):
			numr += (traindata[j][i] - mean[i]) ** 2
		sd.append(math.sqrt(numr/(len(traindata)-1)))

		for j in range(len(traindata)):
			traindata[j][i] = (traindata[j][i] - mean[i])/sd[i]
			traindata[j][i] /= len(traindata)

	mean = []
	sd = []
	for i in range(len(testdata[0])):
		m = 0
		numr = 0
		
		for j in range(len(testdata)):
			m += testdata[j][i]
		mean.append(m/len(testdata))
		
		for j in range(len(testdata)):
			numr += (testdata[j][i] - mean[i]) ** 2
		sd.append(math.sqrt(numr/(len(testdata)-1)))

		for j in range(len(testdata)):
			testdata[j][i] = (testdata[j][i] - mean[i])/sd[i]
			testdata[j][i] /= len(testdata)

	return traindata,testdata
	


###################
## Solver for least squares (linear regression)
## return [w, w0]
###################
def least_squares(traindata, trainlabels):
	w=[]
	w0 = 0
	for i in range(0,len(traindata[0])):
	    w.append(0.01 * random.random() - 0.01)
	####	Compute gradient descent
	for i in range(0,len(traindata)):
		gradient = []
		for j in range(0,len(traindata[0])):
			gradient.append(0)
		for k in range(0,len(traindata)):
			if trainlabels[k] != None :
				dp = dotproduct(w,traindata[i]) 
				for l in range(0,len(traindata[0])):
					gradient[l] = gradient[l] + ((trainlabels[k] - dp) * traindata[k][l])

		##### update W
		for j in range(0, len(traindata[0])):
			w[j] += ( eta * gradient[j])
	# Norm W
	norw = 0
	for j in range(0, cols):
	    norw += w[j] ** 2
	norw = math.sqrt(norw)	

	w0 = abs(w[len(w) - 1] / norw)
	return w,w0


# ###################
# ## Solver for regularized least squares (linear regression)
# ## return [w, w0]
# ###################
def least_squares_regularized(traindata, trainlabels):
	w=[]
	w0 = 0
	prevObj = 0
	obj = 0
	for i in range(0,len(traindata[0])):
	    w.append(0.01 * random.random() - 0.01)
	####	Compute gradient descent
	error = 0
	while abs(prevObj-obj) <= stopCondition:
		gradient = []
		for j in range(0,len(traindata[0])):
			gradient.append(0)
		for k in range(0,len(traindata)):
			if trainlabels[k] != None :
				dp = dotproduct(w,traindata[i]) 
				for l in range(0,len(traindata[0])):
					gradient[l] = gradient[l] + ((trainlabels[k] - dp) * traindata[k][l]) + (2 * lamda * w[l])

		##### update W
		for j in range(0, len(traindata[0])):
			w[j] += ( eta * gradient[j])

		risk_change = 0
		for i in range(len(w)):
			risk_change += w[i] ** 2
		### Compute Error
		error = 0
		for i in range(0,len(traindata)):
			if trainlabels[i] != None:
				error += round(trainlabels[i] - dotproduct(w,traindata[i])) ** 2 + risk_change

		prevObj = obj
		obj = error
	norw = 0
	for j in range(0, cols):
	    norw += w[j] ** 2
	norw = math.sqrt(norw)	

	w0 = abs(w[len(w) - 1] / norw)
	return w,w0
	

# ###################
# ## Solver for hinge loss
# ## return [w, w0]
# ###################
def hinge_loss(traindata, trainlabels):
	w = []
	w0 = 0
	prevObj = 0
	obj = 0
	for i in range(0,len(traindata[0])):
		w.append(0.01 * random.random() - 0.01)

	while (abs(prevObj-obj) > stopCondition):
		gradient = []
		for j in range(0,len(traindata[0])):
			gradient.append(0)
        
		for k in range(0,len(traindata)):
			if trainlabels[k] != None :
				if (trainlabels[k] * dotproduct(w,traindata[k])) < 1:
					for l in range(0,len(traindata[0])):
						gradient[l] = gradient[l] + (-1 * trainlabels[l] * traindata[k][l])

		for j in range(cols):
			w[j] -= eta * (gradient[j])


		emp_risk = 0
		for j in range(len(traindata)):
			if trainlabels[j] != None:
				emp_risk += max(0,1-(trainlabels[j] * dotproduct(w,traindata[j])))

		prevObj = obj
		obj = emp_risk
	norw = 0
	for j in range(0, cols):
	    norw += w[j] ** 2
	norw = math.sqrt(norw)	

	w0 = abs(w[len(w) - 1] / norw)

	return w,w0

# ###################
# ## Solver for regularized hinge loss
# ## return [w, w0]
# ###################
def hinge_loss_regularized(traindata, trainlabels):
	w = []
	w0 = 0
	prevObj = 0
	obj = 0
	for i in range(0,len(traindata[0])):
		w.append(0.01 * random.random() - 0.01)

	while (abs(prevObj-obj) > stopCondition):
		gradient = []
		for j in range(0,len(traindata[0])):
			gradient.append(0)
        
		for k in range(0,len(traindata)):
			if trainlabels[k] != None :
				if (trainlabels[k] * dotproduct(w,traindata[k])) < 1:
					for l in range(0,len(traindata[0])):
						gradient[l] = gradient[l] + (-1 * trainlabels[l] * traindata[k][l]) + (2 * lamda * w[l])
				else:
					for l in range(0,len(traindata[0])):
						gradient[l] += (2 * lamda * w[l])

		for j in range(cols):
			w[j]=(w[j]) - eta * (gradient[j])

		emp_risk = 0
		risk_change = 0
		for i in range(len(w)):
			risk_change += w[i] ** 2
		for j in range(len(traindata)):
			if trainlabels[j] != None:
				emp_risk += max(0,1-(trainlabels[j] * dotproduct(w,traindata[j]))) + (lamda * risk_change)

		prevObj = obj
		obj = emp_risk
	norw = 0
	for j in range(0, cols):
	    norw += w[j] ** 2
	norw = math.sqrt(norw)	

	w0 = abs(w[len(w) - 1] / norw)

	return w,w0



# ###################
# ## Solver for logistic regression
# ## return [w, w0]
# ###################

def dot(weightList, trainList):
	outPut = []
	for i in trainList:
		i = [1] + i
		sum = 0
		for j,k in zip(weightList, i):
			sum += j * k
		outPut.append(sum)
	return outPut

def sigmoid(weightList, trainList):
	outPut = dot(weightList, trainList)
	for i in range(len(outPut)):
		outPut[i] = 1 / (1 + (2.718281828459045) ** (-1 * outPut[i]))
	return outPut

def sigm (w, inp):
	sum = 0
	for i in range(len(w)):
		sum += w[i] * inp[i]
	sum = 1/(1+ (2.718281828459045 ** (-1*sum)))
	if sum == 1.0:
		sum = 0.9999
	return sum

def loss(outPut, trainList, weightList):
	totalLoss = [0 for i in weightList]
	lse = 0
	for  i, j in zip(trainList, outPut):
		i = [1] + i
		for k in range(len(weightList)):
			totalLoss[k] += (eta * (i[-1] - j) * i[k])		    
		lse -=  (i[-1] * math.log(sigm(weightList, i)) + ((1 - i[-1]) * math.log(1 - sigm(weightList, i))))
	return totalLoss, lse

def logistic_regression(traindata, trainlabels):
	w = []
	w0 = 0
	prevObj = 0
	obj = 0 	
	for i in range(0,len(traindata[0])):
		w.append(0.01 * random.random() - 0.01)

	while (abs(prevObj-obj) < stopCondition):
		outPutSigmoid = sigmoid(w, traindata)
		factor , obj = loss(outPutSigmoid, traindata, w)

		for j in range(cols):
			w[j] += factor[j]

	ret = ""
	pred = []
	for i in range(rows):
		if (trainlabels[i] != None):
			s = sigm(w,testdata[i])
			pred.append(s)
			if s > logistic_regression_factor :
				ret += "1\n"
			else:
				ret += "-1\n"
	text_file = open("logistic_regression","w")
	text_file.write(ret)
	text_file.close()
	return pred






	

# ###################
# ## Solver for adaptive learning rate hinge loss
# ## return [w, w0]
# ###################
def hinge_loss_adaptive_learningrate(traindata, trainlabels):
	w = []
	w0 = 0
	prevObj = 0
	obj = 0
	for i in range(0,len(traindata[0])):
		w.append(0.01 * random.random() - 0.01)

	while (abs(prevObj-obj) <= stopCondition):
		gradient = []

		for j in range(0,len(traindata[0])):
			gradient.append(0)
        
		for k in range(0,len(traindata)):
			if trainlabels[k] != None :
				if (trainlabels[k] * dotproduct(w,traindata[k])) < 1:
					for l in range(0,len(traindata[0])):
						gradient[l] = gradient[l] + (-1 * trainlabels[l] * traindata[k][l])
		
		eta_list = [1, .1, .01, .001, .0001, .00001, .000001, .0000001, .00000001, .000000001, .0000000001, .00000000001 ]
		bestobj=1000000000000

		for k in range(0, len(eta_list), 1):
			eta = eta_list[k]
            
            ### Updating weights 
			for j in range(0,cols,1):
				w[j]=(w[j]) + eta * (gradient[j])
                
            ###Computing Error
			error = 0.0
			for i in range(0,rows):
				if(trainlabels[i] != None):
					if (trainlabels[k] * dotproduct(w,traindata[k])) < 1:
						for l in range(0,len(traindata[0])):
							error += max(0,1-(trainlabels[j] * dotproduct(w,traindata[i])))
		
			prevobj = obj
			obj = error   

			if obj < bestobj:
				bestobj = obj
				best_eta = eta
			    
			for j in range(cols):
				w[j] -= eta * gradient[j]
			eta = best_eta


		for j in range(cols):
			w[j] -= eta * (gradient[j])

		emp_risk = 0
		for j in range(len(traindata)):
			if trainlabels[j] != None:
				emp_risk += max(0,1-(trainlabels[j] * dotproduct(w,traindata[j])))

		prevObj = obj
		obj = emp_risk
	norw = 0
	for j in range(0, cols):
	    norw += w[j] ** 2
	norw = math.sqrt(norw)	

	w0 = abs(w[len(w) - 1] / norw)
	return w,w0

def predictions(w,n):
	ret = ""
	for i in range(0, len(testdata)):  ##### Change rows to test length
		if (trainlabels[i] != None):
			dp = dotproduct(w, testdata[i])
			if (dp > 0):
				ret += "1\n"
			else:
				ret += "-1\n"
	text_file = open(str(n),"w")
	text_file.write(ret)
	text_file.close()


#### MAIN #########
eta = 0.001
lamda = 0.01
stopCondition = 0.001
logistic_regression_factor = 0.5

###################
#### Code to read train data and train labels
###################
arg_1 = sys.argv[1]
arg_2 = sys.argv[2]

f = open(arg_1)
traindata = []
trainlabels = []
l = f.readline()
while(l !=''):
    a = l.split()
    l2 = []
    for j in range(1, len(a)):
        l2.append(float(a[j]))
    traindata.append(l2)
    trainlabels.append(float(a[0]))
    l = f.readline()
rows = len(traindata)
cols = len(traindata[0])
f.close()

f = open(arg_2)
testdata = []
testlabels = []
l = f.readline()
while(l !=''):
    a = l.split()
    l2 = []
    for j in range(1, len(a)):
        l2.append(float(a[j]))
    testdata.append(l2)
    testlabels.append(float(a[0]))
    l = f.readline()
rows = len(testdata)
cols = len(testdata[0])
f.close()

###################
#### Code to test data and test labels
#### The test labels are to be used
#### only for evaluation and nowhere else.
#### When your project is being graded we
#### will use 0 label for all test points
###################


[traindata, testdata] = standardize_data(traindata, testdata)

[w,w0] = least_squares(traindata, trainlabels)
predictions(w,"least_squares")

[w,w0] = least_squares_regularized(traindata, trainlabels)
predictions(w,"least_squares_regularized")

[w,w0] = hinge_loss(traindata, trainlabels)
predictions(w,"hinge_loss")

[w,w0] = hinge_loss_regularized(traindata, trainlabels)
predictions(w,"hinge_loss_regularized")

w = logistic_regression(traindata, trainlabels)

[w,w0] = hinge_loss_adaptive_learningrate(traindata, trainlabels)
predictions(w,"hinge_loss_adaptive_learningrate")

# ###################
# ## Optional for testing on toy data
# ## Comment out when submitting project
# ###################
# print(w) 
# wlen = math.sqrt(w[0]**2 + w[1]**2)
# dist_to_origin = abs(w[2])/wlen
# print("Dist to origin=",dist_to_origin)

# wlen=0
# for i in range(0, len(w), 1):
# 	wlen += w[i]**2
# wlen=math.sqrt(wlen)
# print("wlen=",wlen)	

# ###################
# #### Classify unlabeled points
# ##################