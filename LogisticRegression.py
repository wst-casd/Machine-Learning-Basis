from numpy import * 

def sigmoid( data ):
	return 1.0/(1 + exp( -data ))


def cost_grad( w, data, labels, lamb ):
	m = data.shape[0]

	g_x = sigmoid( labels*dot(data,w) )

	cost = -1.0/m*sum( log( g_x ) )
	grad = 1.0/m*dot( data.T, ( labels*(g_x-1) ) )

	temp_w = hstack( ( 0, w[1:] ) )

	cost = cost + lamb/(2.0*m)*dot( temp_w, temp_w )
	grad = grad + lamb/m*temp_w

	return cost, grad


# 随机梯度下降(Stochastic Gradient Descent)
def SGD( func, data, labels, initialW, lamb, numIter=100 ):
	w = initialW
	m = data.shape[0]

	for i in range( numIter ):
		dataIndex = arange( m )
		for j in range(1,10):
			stepsize = 4/(1.0+i+j) + 0.01
			randomIndex = int( random.uniform( 0, len( dataIndex ) ) )
			cost, grad = func( w, data[randomIndex], labels[randomIndex], lamb )
			w = w - stepsize*grad
			dataIndex = delete( dataIndex, randomIndex, 0 )

	return w

# 批梯度下降(Batch Gradient Descent)
def BGD( func, data, labels, initialW, lamb, numIter=100 ):
	w = initialW

	stepsize = 0.01

	for x in range( numIter ):
		cost, grad = func( w, data, labels, lamb )
		w = w - stepsize*grad

	return w