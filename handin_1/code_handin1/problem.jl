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
itrs1 = 10000
init_value = 100
gamma = 1.5/(maximum(eigvals(Q))) # 0 < gamma*I - Q, e.i. pos. def.

x_vals = zeros((n,itrs1))

x_vals[:, 1] = ones((n,1))*init_value #init_value
for i = 2:itrs1
	#print(i,x_vals[:,i-1],x_vals[:,i])
	x_vals[:,i] = prox_box(x_vals[:,i-1]- gamma * grad_quad(x_vals[:,i-1], Q, q), a,b,gamma)
end


x_res_norm = zeros((1,itrs1-1))
for i = 1:itrs1-1
	x_res_norm[i] = opnorm((x_vals[:,i+1] - x_vals[:,i])')
end

using Plots
# Create a new plot dual prob norm
plot(log.(x_res_norm'), margin=5Plots.mm)
ylabel!("ln||x^(k+1)-x^k||")
xlabel!("Iterations")

savefig("task6_gamma_001.png")

plot(x_vals[1,:], label = "x_1")
plot!(a[1]*ones(itrs), label = "a_1")
plot!(b[1]*ones(itrs), label = "b_1")

xlabel!("iterations")
savefig("task66.png")

# TASK 7
itrs = 30000
init_value = 1
mu = zeros((n,itrs))
x_prim = zeros((n,itrs))
gamma = 1.5/(maximum(eigvals(inv(Q)))) # 0 < gamma*I - Q, e.i. pos. def.

mu[:, 1] = ones((n,1))*init_value #init_value
x_prim[:, 1] = ones((n,1))*init_value #init_value
for i = 2:itrs
	mu[:,i] = -prox_boxconj(- mu[:,i-1] + gamma  * grad_quadconj(mu[:,i-1], Q, q), a,b,gamma)
	x_prim[:,i] = inv(Q)*(mu[:,i]-q)
end

x_prim_norm = zeros((1,itrs-1))
mu_norm = zeros((1,itrs-1))
for i = 1:itrs-1
	x_prim_norm[i] = opnorm((x_prim[:,i+1] - x_prim[:,i])')
	mu_norm[i] = opnorm((mu[:,i+1] - mu[:,i])')
end

# Create a new plot
plot(log.(x_prim_norm'))
ylabel!("ln||x^(k+1)-x^k||")
xlabel!("Iterations")

plot(log.(mu_norm'), margin=5Plots.mm)
ylabel!("ln||mu^(k+1)-mu^k||")
xlabel!("Iterations")

savefig("task7_gamma_001.png")

plot(mu[1,:], label = "x_1")
plot!(a[1]*ones(itrs), label = "a_1")
plot!(b[1]*ones(itrs), label = "b_1")

xlabel!("iterations")
savefig("task7set.png")

fx_prim = zeros((1,itrs))
fxi = zeros((1,itrs))
for i = 1:itrs
	fx_prim[i] = quad(x_prim[:,i], Q, q)
	fxi[i] = quad(x_prim[:,i], Q, q) + box(x_prim[:,i], a, b)
end

plot(fx_prim')
xlabel!("iterations")
ylabel!("f(x'^k)")
savefig("fxprim.png")

plot(fxi')
xlabel!("iterations")
ylabel!("f(x'^k)+i(x'^k)")
savefig("fxprim.png")

# RÄTT?!?!? Går inte mot noll så nej, It is actually not just a numerical error.
#
# From manual:
# Does \hat{x}^k always satisfy the constraint \hat{x}^k \in S?
# If you check the distance from \hat{x}^k to S, for example as
# ||\hat{x}^k - prox_{i_S}(\hat{x}^k)||
# you will see that although the distance is usually >0 it does decrease towards 0 when k increases.

x_prim_mu = zeros((1,itrs))
for i = 1:itrs
	x_prim_mu[i] = opnorm((x_prim[:,i]-mu[:,i])')
end

plot(log.(x_prim_mu'))
