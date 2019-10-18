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
	μ = 5000 # tuned until convergence minimum eigenvalue?

	for i = 3:itrs
		if gradient == 0
			w_k_half = w[:,i-1]
		elseif gradient == 1
			β = (i-2)/(i+1)
			w_k_half = w[:,i-1] + β * (w[:,i-1] - w[:,i-2])
		elseif gradient == 2
			t_k1 = (1+sqrt(1+4(t_k0)^2))/2
			β = (t_k0 - 1)/t_k1
			w_k_half = w[:,i-1] + β * (w[:,i-1] - w[:,i-2])
			t_k0 = t_k1
		elseif gradient == 3
			β = (1-sqrt(μ * step_size)) / (1+sqrt(μ * step_size))
			w_k_half = w[:,i-1] + β * (w[:,i-1] - w[:,i-2])
		end
		grad_Q = Q*w_k_half
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

function svm_cord(x, y, σ, λ, itrs, step_size_ii = 0)
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
	y_i = ones(1)
	hx = HingeLoss(y_i,N)
	h_con = Conjugate(hx)

	Y = Diagonal(y)

	K = Kernel(x,σ)

	Q = inv(λ)*(Y*K*Y) # 1/lambda*2 v'Qv = 1/2 v'YX'XYv from manual
	step_size = inv(opnorm(Q))

	## Implement coordinate decent

	w_n  = ones(size(Q)[1],1)
	w_itrs = Int(floor(itrs/1000))
	w  = ones(size(Q)[1],w_itrs)
	i_w = 1
	for i = 1:itrs-1
		sample_i = rand(1:length(x)) #Uniform choise

		if i % 1000 == 0 # For plot
			w[:,i_w] = w_n
			i_w = i_w + 1
		end
		if step_size_ii == 1 # coordinate wise step-size
			step_size = 1/Q[sample_i,sample_i]
		end

		grad_Q = Q[sample_i,:]'*w_n
		wk, _ = prox(h_con, [w_n[sample_i]-step_size *  grad_Q[1]], step_size) # Why [] in proxy? See manual task 2...
		w_n[sample_i] = wk[1]
	end

	return w
end

############################
############################
############################
###### TASK 1
function task_1(test)
	print("---------- TASK 1 ----------\n")
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

	ploty = 250
	plot(log.(itrs_w0), xlims=[0,ploty], label = "w0", margin=5Plots.mm)
	ylabel!("log(||w_i-w^*||)")
	xlabel!("iterations")
	plot!(log.(itrs_w1), xlims=[0,ploty], label = "w1", margin=5Plots.mm)
	plot!(log.(itrs_w2), xlims=[0,ploty], label = "w2", margin=5Plots.mm,  linestyle = :dash)
	plot!(log.(itrs_w3), xlims=[0,ploty], label = "w3", margin=5Plots.mm)

	savefig("/Users/JonatanMBA/google drive/lth/frtn50/handin_3/task1_2/plots/1.png")
end

task_1(1)

############################
############################
############################
###### TASK 2
function task_2()
	print("---------- TASK 2 ----------\n")
	x,y = svm_train()
	λ = 0.0001 # from handin 2
	σ = 0.25 # from handin 2

	print("Training model, w0... \n")
	@time w0 = svm(x,y,σ,λ,120,0)
	print("Training model, w1... \n")
	@time w1 = svm(x,y,σ,λ,200,1)
	print("Training model, w2... \n")
	@time w2 = svm(x,y,σ,λ,200,2)
	print("Training model, w3... \n")
	@time w3 = svm(x,y,σ,λ,70,3)
	print("Training model, coord uniform choise step size... \n")
	@time w4 = svm_cord(x,y,σ,λ,65000,0)
	print("Training model, coord coordinate wise step size... \n")
	@time w5 = svm_cord(x,y,σ,λ,33000,1)

	print(size(w4))
	itrs_w0 = ones(size(w0)[2])
	itrs_w1 = ones(size(w1)[2])
	itrs_w2 = ones(size(w2)[2])
	itrs_w3 = ones(size(w3)[2])
	itrs_w4 = ones(size(w4)[2])
	itrs_w5 = ones(size(w5)[2])
	x_coord_plot4 = ones(size(w4)[2])
	x_coord_plot5 = ones(size(w5)[2])
	for i = 1:length(itrs_w0)-1
		itrs_w0[i] = opnorm((w0[:,i]-w0[:,end])')
	end
	for i = 1:length(itrs_w1)-1
		itrs_w1[i] = opnorm((w1[:,i]-w0[:,end])')
	end
	for i = 1:length(itrs_w2)-1
		itrs_w2[i] = opnorm((w2[:,i]-w0[:,end])')
	end
	for i = 1:length(itrs_w3)-1
		itrs_w3[i] = opnorm((w3[:,i]-w0[:,end])')
	end
	for i = 1:length(itrs_w4)-1
		itrs_w4[i] = opnorm((w4[:,i]-w0[:,end])')
		x_coord_plot4[i] = 1000 * (i-1)
	end
	for i = 1:length(itrs_w5)-2
		itrs_w5[i] = opnorm((w5[:,i]-w0[:,end])')
		x_coord_plot5[i] = 1000 * (i-1)
	end

	ploty = 65000
	plot(log.(itrs_w0), xlims=[0,ploty], label = "w0", margin=5Plots.mm)
	ylabel!("log(||w_i-w^*||)")
	xlabel!("iterations")
	plot!(x_coord_plot4[1:end-2], log.(itrs_w4)[1:end-2], xlims=[0,ploty], label = "uniform choise step size", margin=5Plots.mm, color = :red)
	plot!(x_coord_plot5[1:end-2], log.(itrs_w5)[1:end-2], xlims=[0,ploty], label = "coordinate wise step size", margin=5Plots.mm, color = :orange)

	savefig("/Users/JonatanMBA/google drive/lth/frtn50/handin_3/task1_2/plots/2.png")

	ploty = 250
	plot(log.(itrs_w0)[1:end-2], xlims=[0,ploty], label = "w0", margin=5Plots.mm)
	ylabel!("log(||w_i-w^*||)")
	xlabel!("normalized iterations")
	plot!(log.(itrs_w1)[1:end-2], xlims=[0,ploty], label = "w1", margin=5Plots.mm)
	plot!(log.(itrs_w2)[1:end-2], xlims=[0,ploty], label = "w2", margin=5Plots.mm,  linestyle = :dash )
	plot!(log.(itrs_w3)[1:end-2], xlims=[0,ploty], label = "w3", margin=5Plots.mm)
	print(size(x_coord_plot4))
	plot!(x_coord_plot4[1:end-2]./size(w4[:,1]), log.(itrs_w4)[1:end-2], xlims=[0,ploty], label = "uniform choise step size", margin=5Plots.mm, color = :red)
	plot!(x_coord_plot5[1:end-2]./size(w5[:,1]), log.(itrs_w5)[1:end-2], xlims=[0,ploty], label = "coordinate wise step size", margin=5Plots.mm, color = :orange)

	savefig("/Users/JonatanMBA/google drive/lth/frtn50/handin_3/task1_2/plots/3.png")
end

task_2()
