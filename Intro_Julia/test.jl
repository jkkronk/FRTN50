a = 1
b = 1
c = a + b


function f1(x)
        d = x^2 + x;
        return d
end

x = 1.0
\alpha = 0.1

v = [1,2]


using LinearAlgebra, Statistics

A = [1 2;3 4]
det(A)

m, s = mean(A), std(A[:,1])


using Plots
v = randn(10)
plot!(v)
# Create a new plot
plot(v .+ 1, color=:green)
# Add new plot to existing window
plot!(v .- 1, linewidth=3)

savefig("myplot.png")
pwd()


function g1(x, y)
        x^2*y + 1
end

a = [1,2,3,4,5]
b = [6,7,8,9,10]
# Apply the function g1 to each of the elements in a and b
g1.(a, b)

v = [1,2,-23,4]

s1 = sum(v)
s2 = sum(abs, v)

s3 = abs(s1)

function setfirstzero!(A)
# Set the element in first row and column to 0
        A[1,1] = 0
end

A = randn(4,4)
A = zeros(4,4)
# The first element in A will now be set to 0
setfirstzero!(A)

function allzero!(A)
# Element-wise - set all elements in A to zero
# Same as A[:] .= 0
        A .= 0
end

allzero!(A)
sum(A) # This is 0

# Note that the functions above change the elements inside of A, the following does not!
function notworking!(A)
# Create a NEW variable, also called A, and set it to 4 by 4 zeros
        A = zeros(4,4)
# We are no longer referring to the same same matrix
        return A
end

A = randn(4,4)
B = notworking!(A)
sum(A) # This is not zero
sum(B) # This is

notworking!(A)
A






y = randn(10000)
a = randn(10000)
b = randn(10000)

""" calculate a^2 + b^2 element-wise and return vector with result """
function squareadd(a, b)
# Create a vector of same length and type as a to store the result
        y = similar(a)
        for i in eachindex(a) # Same as 1:length(a)
                y[i] = a[i]^2 + b[i]^2
        end
        return y
end

""" calculate a^2 + b^2 element-wise and store in y """
function squareadd!(y, a, b)
        for i in eachindex(a)
                y[i] = a[i]^2 + b[i]^2
        end
        return
end

y = squareadd(a,b)
# @time will measure time and allocations
@time squareadd(a,b)


squareadd!(y, a, b)
@time squareadd!(y,a,b)

struct Point2D
        x::Float64
        y::Float64
end

p1 = Point2D(1,2)
p2 = Point2D(2,3)

p1x = p1.y

distance(a::Point2D, b::Point2D) = sqrt((a.x - b.x)^2 + (a.y - b.y)^2)

distance(p1,p2)
