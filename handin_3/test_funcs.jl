
function gradientstep!(n, lossfunc, x, y)
    out = n(x)
    # Calculate (∂L/∂out)ᵀ
    ∇L = derivative(lossfunc, out, y)
    # Backward pass over network
    backprop!(n, x, ∇L)
    # Get list of all parameters and gradients
    parameters, gradients = getparams(n)
    # For each parameter, take gradient step
    for i = 1:length(parameters)
         p = parameters[i]
         g = gradients[i]
         # Update this parameter with a small step in negative gradient
         #→ direction
         p .= p .- 0.001.*g
         # The parameter p is either a W, or b so we broadcast to update all the
         #→ elements
    end
end

n = Network([Dense(3, 1, sigmoid), Dense(1, 3, sigmoid)])
x = randn(1)
y = [1.0] # We want the output to be 1

n(x) # This is probably not close to 1

gradientstep!(n, sumsquares, x, y)

n(x)
