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

function svm(x, y, σ, λ, itrs, gradient)
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
	w_k_half = ones(size(Q)[1])
	t_k0 = 1

	for i = 3:itrs
		grad_Q = Q*w[:,i-1]

		if gradient == 0
			w_k_half = w[:,i-1]
		elseif gradient == 1
			β = (i-2)/(i+1)
			w_k_half = w[:,i-1] + β .* (w[:,i-1] - w[:,i-2])
		elseif gradient == 2
			t_k1 = (1+sqrt(1+4(t_k0)^2))/2
			β = (t_k0 - 1)/t_k1
			w_k_half = w[:,i-1] + β .* (w[:,i-1] - w[:,i-2])
		elseif gradient == 3
			μ = 1
			β = (1-sqrt(μ * λ)) / (1+sqrt(μ * λ))
			w_k_half = w[:,i-1] + β .* (w[:,i-1] - w[:,i-2])
		end
		w[:,i], _ = prox(h_con, w_k_half - step_size .*  grad_Q, step_size)
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

function test_1(test)
	x,y = svm_train()
	λ = 0.0001
	σ = 0.25
	itrs0 = 100000
	itrs = 10000

	if test == 1
		x_test,y_test = svm_test_1()
	elseif test == 2
		x_test,y_test = svm_test_2()
	elseif test == 3
		x_test,y_test = svm_test_3()
	else
		x_test,y_test = svm_test_4()
	end

	print("Training model, w0... \n")
	w0 = svm(x,y,σ,λ,itrs0,0)
	print("Training model, w1... \n")
	w1 = svm(x,y,σ,λ,itrs,1)
	print("Training model, w2... \n")
	w2 = svm(x,y,σ,λ,itrs,2)
	print("Training model, w3... \n")
	w3 = svm(x,y,σ,λ,itrs,3)

	pred_valid_corr = zeros(length(x))
	pred_valid_corr1 = zeros(length(x))
	pred_valid_corr2 = zeros(length(x))
	pred_valid_corr3 = zeros(length(x))
	for i = 1:length(x)
		y_pred_valid = model(w0[:,end],x,y,x[i],λ,σ)
		y_pred_valid1 = model(w1[:,end],x,y,x[i],λ,σ)
		y_pred_valid2 = model(w2[:,end],x,y,x[i],λ,σ)
		y_pred_valid3 = model(w3[:,end],x,y,x[i],λ,σ)
		if y_pred_valid == y[i]
			pred_valid_corr[i] = 1
		end
		if y_pred_valid1 == y[i]
			pred_valid_corr1[i] = 1
		end
		if y_pred_valid2 == y[i]
			pred_valid_corr2[i] = 1
		end
		if y_pred_valid3 == y[i]
			pred_valid_corr3[i] = 1
		end
	end

	print("\n Train set Error rate, w0: ",1-mean(pred_valid_corr))
	print("\n Train set Error rate, w1: ",1-mean(pred_valid_corr1))
	print("\n Train set Error rate, w2: ",1-mean(pred_valid_corr2))
	print("\n Train set Error rate, w3: ",1-mean(pred_valid_corr3))

	pred_corr = zeros(length(x_test))
	pred_corr1 = zeros(length(x_test))
	pred_corr2 = zeros(length(x_test))
	pred_corr3 = zeros(length(x_test))
	for i = 1:length(x_test)
		y_pred = model(w0[:,end],x,y,x_test[i],λ,σ)
		y_pred1 = model(w1[:,end],x,y,x_test[i],λ,σ)
		y_pred2 = model(w2[:,end],x,y,x_test[i],λ,σ)
		y_pred3 = model(w3[:,end],x,y,x_test[i],λ,σ)
		if y_pred == y_test[i]
			pred_corr[i] = 1
		end
		if y_pred1 == y_test[i]
			pred_corr1[i] = 1
		end
		if y_pred2 == y_test[i]
			pred_corr2[i] = 1
		end
		if y_pred3 == y_test[i]
			pred_corr3[i] = 1
		end
	end

	print("\n Validation set ",test  ," Error rate: ",1-mean(pred_corr))
	print("\n Validation set ",test  ," Error rate: ",1-mean(pred_corr1))
	print("\n Validation set ",test  ," Error rate: ",1-mean(pred_corr2))
	print("\n Validation set ",test  ," Error rate: ",1-mean(pred_corr3))

	itrs_w0 = ones(size(w0)[2])
	itrs_w1 = ones(size(w1)[2])
	itrs_w2 = ones(size(w2)[2])
	itrs_w3 = ones(size(w3)[2])
	for i = 1:itrs-1
		itrs_w0[i] = opnorm((w0[:,i]-w0[:,end])')
		itrs_w1[i] = opnorm((w1[:,i]-w0[:,end])')
		itrs_w2[i] = opnorm((w2[:,i]-w0[:,end])')
		itrs_w3[i] = opnorm((w3[:,i]-w0[:,end])')
	end

	plot(log.(itrs_w0), xlims=[0,9000], label = "w0", margin=5Plots.mm)
	ylabel!("log(||w_i-w^*||)")
	xlabel!("iterations")
	plot!(log.(itrs_w1), xlims=[0,9000], label = "w1", margin=5Plots.mm)
	#plot!(log.(itrs_w2), xlims=[0,9000], label = "w2", margin=5Plots.mm)
	plot!(log.(itrs_w3), xlims=[0,9000], label = "w3", margin=5Plots.mm)

	savefig("/Users/JonatanMBA/google drive/lth/frtn50/handin_3/task1/plots/1.png")
end

test_1(1)
