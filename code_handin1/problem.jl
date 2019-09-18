using Random
using LinearAlgebra
include("functions.jl")

"""
    problem_data()

Returns the Q, q, a, and b matrix/vectors that defines the problem in Hand-In 1.

"""
function problem_data()
	mt = MersenneTwister(123)

	n = 20

	Qv = randn(mt,n,n)
	Q = Qv'*Qv
	q = randn(mt,n)

	a = -rand(mt,n)
	b = rand(mt,n)

	return Q,q,a,b,n
end

Q,q,a,b,n = problem_data()
itrs = 100
init_value = 100
x_vals = zeros((n,itrs))
x_vals[:, 1] = ones((n,1))*init_value #init_value
gamma = maximum(eigvals(Q)) + 1 # 0 < gamma*I - Q, e.i. pos. def.

for i = 2:itrs
	#print(i,x_vals[:,i-1],x_vals[:,i])
	x_vals[:,i] = prox_box(x_vals[:,i-1], a,b,gamma)
end
