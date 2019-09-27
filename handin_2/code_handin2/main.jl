include("problem.jl")

using Plots
using LinearAlgebra
using ProximalOperators

##################################
#Least Squares Regression
#TASK 1-4
##################################

x,y = leastsquares_data()

#### PARAMS ####
q = 3
n = size(x)[1]



plot(x,y, seriestype=:scatter)
savefig("/Users/JonatanMBA/google drive/lth/frtn50/plots_hi2/xyplot_logreg.png")

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

x_scaled = r(x)

function create_X(x,q)
	X_ret = zeros(size(x)[1],q+1)
	for i = 1:q+1
		X_ret[:,i] = x.^i
	end
	return X_ret
end

X_scaled = create_X(x_scaled,q)

step_size = inv(maximum(eigvals(X_scaled*X_scaled')))

w  = ones(n,q+1)

for i = 2:11




end




############ PARAMS
regularization = 1
itrs = 10000
init_value = 1


fSqrNorm = SqrNormL2() # Create the function 1

function_f(w) = (1/2) * fSqrNorm(x_scaled'*w-y) + regularization * NormL2(w)^2
##

fSqrNorm((x_scaled'*w[:,1]-y)')

x_scaled*w[:,1]-y

size(x_scaled')
size(w[:,1]')
size(y)

function_f(w[:,1])
##
w = ones(11,itrs)
df, _ = gradient(function_f, w[:,1])



val = f1([1.0, 1.0]) # val = 1.0
df, _ = gradient(f1, [1.0, 1.0]) # df = [1.0, 1.0]
pf, _ = prox(f1, [1.0, 1.0], 0.5) # pf = [0.6666..., 0.6666...]

norm_function = NormL2()




w = ones(11,itrs)
for i = 2:11
	print(i)
	f(w) = (1/2)*opnorm((x_scaled*w'-y)')^2



	w[:,i] = prox(function_w,w[:,i-1]-step_size*gradnorm)
end



f(w) = (1/2)*opnorm((x_scaled'*w-y)')^2
f(w[:,1])

fw = gradient!(f, w[:,1])


print(size(x_scaled*w[:,1]'))
gradient!(f, w[:,1])
