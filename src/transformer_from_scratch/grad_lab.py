import torch

def run_my_grad_lab():

    x = torch.tensor(2.0, requires_grad=True)

    ## Define the x*2 and take the derivative
    g = x**2

    g.backward()

    # Trivial derivative 4
    print(f'Result dg/dx: {x.grad}')

    y = torch.tensor(3.14, requires_grad=True)

    # Define another function, sin(x)
    z = torch.sin(y)

    # Compute the derivative cos(Pi)
    z.backward()
    print(f'Result dz/dy: {y.grad}')

    # Reuse x grad the zero first!
    x.grad.zero_()
    f3 = torch.exp(x)
    f3.backward() # No need to retain anymore; this is the last one
    print(f"df3/dx (exp): {x.grad:.4f}") # Should be exp(2) approx 7.389

    # Phase 2 ### Muti-dimensional (The Jacobian) ###

    # Shaoe (2,2)
    w= torch.tensor([[1.0, 5.0], [2.0, 10.0]], requires_grad=True)

    # print the sum
    print(f'Sum of w: {w.sum()}')

    # Shape(2)
    x1 = torch.tensor([1.0, 1.0], requires_grad=True)

    # Compute the product. The derivative is going to be the 
    y = torch.matmul(w, x1)

    # Le's define a loss as the sum of all the components in y (the sum is 18)
    L = y.sum(dim=-1)
    print(L)
    # let's take the gradient here of L with respect to x and w 
    L.backward()
    print(f'dL/dx: {x1.grad}')

    # Phase 3 Residual Connections toy example
    # Goal: Prove that in a residual layer y = x + f(x), 
    # the gradient dL/dx always includes the "Identity" signal (1.0)
    
    x_res = torch.tensor([5.0], requires_grad=True)

    # Let's set a nice, easy to compute f, whose gradient vanishes for x_res
    f = x_res**(-10)
    y = x_res + f

    # Compute the gradient
    #You see the the gradient vanishes for the function but then the x factor gives a solid 1 which is going 
    # to prevent the gradient from disappearing!
    y.backward()
    print(f'dy/dx: {x_res.grad.item()}')






if __name__=='__main__':
    run_my_grad_lab()


