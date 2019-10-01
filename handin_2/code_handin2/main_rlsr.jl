include("problem.jl")

using Plots
using LinearAlgebra
using ProximalOperators

##################################
#Regularized Least Squares Regression
##################################

#### TASK 1 ####
function r(x)
	"""
	Scales vetor to [-1,1]
	:param: vector
	:return: scaled vector
	"""
	x_max = maximum(x)
	x_min = minimum(x)

	β = (x_max+x_min)/2
	σ = 1/(x_max-β)

	x_scaled = (x.-β)*σ
	return x_scaled
end

function create_X(x,p)
	"""
	Creates a feature vector map with given data and polynimial p
	:param: given x data
	:param: polynimial order
	:return: feature map
	"""

	X_ret = zeros(p+1,size(x)[1])
	for i = 0:p
		X_ret[i+1,:] = x'.^i
	end
	return X_ret
end

function least_squares(x, y, p, λ, q ,itrs)
	"""
	Performs least squares to find model vector w and
	:param: given x data
	:param: given y data
	:param: polynomial order
	:param: given step_size lambda
	:param: NormL1: q=1 or SqrNormL2: q=2
	:param: iterations for prox gradient method
	:return: model weight vector
	"""
	n= size(x)[1]

	x_scaled = r(x)
	X_scaled = create_X(x_scaled,p)

	step_size = inv(maximum(eigvals(X_scaled*X_scaled')))

	w  = ones(p+1,itrs)

	for i = 2:itrs
		grad_fw, _ = gradient(SqrNormL2(1), (X_scaled'*w[:,i-1].-y))

		if q == 2
			wnorm =  SqrNormL2(2 * λ)
		else
			wnorm = NormL1(λ)
		end

		w[:,i], _ = prox(wnorm, w[:,i-1] - step_size .* X_scaled *  grad_fw, step_size)
	end

	return w

end

function model(w, x, p)
	"""
	Predicts values for least square regression w*x=y
	:param: weight vector
	:param: linspace data vector
	:param: polynomial order
	:return: predicted values
	"""
	x_scaled = r(x)
	X_scaled = create_X(x_scaled,p)
	return (w[:,end]'*X_scaled)'
end

#### Task 2-3 ####
x,y = leastsquares_data() # Given Data
p = 10 # Polynomial order
λ = 0.1 # Regression factor
q = 2
w = least_squares(x, y, p, λ, q, 5000000)

plot(x,y, seriestype=:scatter, marker = 3,label="(x,y)data", xlims=[-1.05,3.05],ylims=[-7,7])

#### Model fit/plot
x_axis = LinRange(-1.2, 3.2, 1000)
y_model = model(w,x_axis,p)
plot!(x_axis,y_model, ylims=(-10,10),label="model_w(x)")
ylabel!("y")
xlabel!("rescaled x")

savefig("/Users/JonatanMBA/google drive/lth/frtn50/handin_2/plots_hi2/task2/task4p10_wopre.png")

#### Model convergence rate
itrs_w = ones(size(w)[2])
for i = 1:length(itrs_w)-1
	#y_i = (w[:,i]'*X_scaled)'

	itrs_w[i] = opnorm((w[:,i+1]-w[:,i])')
end

plot(log.(itrs_w), xlims=[0,4000000], margin=5Plots.mm)
ylabel!("log(||w_i+1-w_i||)")
xlabel!("iterations")

savefig("/Users/JonatanMBA/google drive/lth/frtn50/handin_2/plots_hi2/task4/task4p10_wopre_c.png")
