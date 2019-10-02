include("problem.jl")

using Plots
using LinearAlgebra
using ProximalOperators
using Statistics
using Random


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
	:param: lambda - regularization factor
	:param: sigma - lenght-scale, smoothness param
	:return: predicted labels
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
# TesT
############################

#λ ∈ {0.1, 0.01, 0.001, 0.0001} and σ ∈ {1, 0.5, 0.25}.
function test_6(test)
	x,y = svm_train()
	λ = 0.1
	σ = 1
	itrs = 100000

	if test == 1
		x_test,y_test = svm_test_1()
	elseif test == 2
		x_test,y_test = svm_test_2()
	elseif test == 3
		x_test,y_test = svm_test_3()
	else
		x_test,y_test = svm_test_4()
	end

	w = svm(x,y,σ,λ,itrs)

	pred_valid_corr = zeros(length(x))
	for i = 1:length(x)
		y_pred_valid = model(w[:,end],x,y,x[i],λ,σ)
		if y_pred_valid == y[i]
			pred_valid_corr[i] = 1
		end
	end

	print("Train set Error rate: ",1-mean(pred_valid_corr))

	pred_corr = zeros(length(x_test))
	for i = 1:length(x_test)
		y_pred = model(w[:,end],x,y,x_test[i],λ,σ)
		if y_pred == y_test[i]
			pred_corr[i] = 1
		end

	end

	print("\nValidation set ",test  ," Error rate: ",1-mean(pred_corr))

	itrs_w = ones(size(w)[2])
	for i = 1:length(itrs_w)-1
		itrs_w[i] = opnorm((w[:,i+1]-w[:,i])')
	end

	plot(log.(itrs_w), xlims=[0,10000], margin=5Plots.mm)
	ylabel!("log(||w_i+1-w_i||)")
	xlabel!("iterations")

	savefig("/Users/JonatanMBA/google drive/lth/frtn50/handin_2/plots_hi2/task7/00001025_c.png")
end

function kfold(k=10, σ=0.25, λ=0.0001)
	itrs = 10000
	x,y = svm_train()
	indx = randperm(length(x))
	set_size = Int(length(x)/k)
	results = zeros(k)
	for i=1:k-1
		a = zeros(set_size)
		for j=1:set_size
			a[j] = (popfirst!(indx))
		end
		a = round.(Int, a)
		xx = x[indx]
		yy = y[indx]
		valx = x[a]
		valy = y[a]
		v = svm(xx,yy,σ,λ,itrs)
		y_rate = zeros(length(valx))
		for j=1:length(valx)

			y_pred = model(v[:,end], xx, yy, valx[j], λ, σ)
			if y_pred == valy[j]
				y_rate[j] = 1
			end
		end
		results[i] = 1 - mean(y_rate)
	end
	return mean(results)
end

function holdout_valid()
	x,y = svm_train()
	λ = 0.0001
	σ = 0.25
	itrs = 100000

	randperm()


end
