include("problem.jl")

using Plots
using LinearAlgebra
using ProximalOperators

##################################
#Support Vector Machines
##################################

function Kernel(x,σ)
    """
    Kernel function given from handin 2 manual
    """
	K = zeros(size(x)[1],size(x)[1])
	for i = 1:size(x)[1]
		for j = 1:size(x)[1]
			K[i,j] = exp(-(1/(2*σ^2)) * opnorm((x[i]-x[j])'))
			K[j,i] = exp(-(1/(2*σ^2)) * opnorm((x[i]-x[j])'))
		end
	end
	return K
end

function svm(x, y, σ, λ, itrs)
	"""
	Trains model weights w
	:param: x data
	:param: y data
	:param: sigma - lenght-scale, smoothness param
	:param: lambda - Regularization factor
	:param: iterations
	:return: model weight vector
	"""

	N = 1/length(x)
	y_i = ones(length(x))
	hx = HingeLoss(y_i,N)
	h_con = Conjugate(hx)

	Y = Diagonal(y)

	K = Kernel(x,σ)

	Q = inv(λ)*(Y*K*Y) # 1/lambda*2 v'Qv = 1/2 v'YX'XYv from manual
	step_size = inv(opnorm(Q))

	w  = ones(size(Q)[1],itrs)

	for i = 2:itrs
		grad_Q = Q*w[:,i-1]
		w[:,i], _ = prox(h_con, w[:,i-1] - step_size .*  grad_Q, step_size)
	end

	return w
end

function model(w,x,y,x_pred,λ,σ)
	"""
	Predicts values for svm
	:param: weight vector
	:param: x data
	:param: y data
	:param: lambda regularization factor

	:return: predicted values
	"""
	M = length(x)
	K_pred = zeros(M)

	for i = 1:M
		K_pred[i] = exp((-1/(2*σ^2))*opnorm((x_pred-x[i])'))
	end

	Y = Diagonal(y)

	y_pred = sign((-1/λ)*(w'*Y*K_pred)')
	return y_pred
end

############################
# TASK 6 test
############################
function test_6()
	x,y = svm_train()
	x_test,y_test = svm_test_1()
	λ = 0.5
	σ = 0.01
	itrs = 10000

	w = svm(x,y,σ,λ,itrs)

	pred_corr = zeros(length(x_test))

	for i = 1:length(x_test)
		y_pred = model(w[:,end],x,y,x_test[i],λ,σ)
		if y_pred == y_test[i]
			pred_corr[i] = 1
		end
	end
	print(mean(pred_corr))


end
