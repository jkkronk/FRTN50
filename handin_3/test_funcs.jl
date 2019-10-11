
function gradientstep!(n, lossfunc, x, y)
    out = n(x)
    # Calculate (∂L/∂out)T
    ∇L = derivative(lossfunc, out, y)
    # Backward pass over network
    backprop!(n, x, ∇L)
    # Get list of all parameters and gradients
    parameters, gradients = getparams(n)
    # For each parameter, take gradient step
    for i = 1:length(parameters)
        p = parameters[i]
        g = gradients[i]
        # Update this parameter with a small step in negative gradient 􏰀→ direction
        p .= p .- 0.001.*g
        # The parameter p is either a W, or b so we broadcast to update all the 􏰀→ elements
    end
end
