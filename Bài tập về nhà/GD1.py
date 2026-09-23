def cost(x):
    return x**2 -2
def grad(x):
    return 2*x
def GD(x0, eta):
    x = [x0]
    for i in range(1,100):
        x_new = x[-1] - eta*grad(x[-1])
        if abs(grad(x_new)) < 1e-3:
            break
        x.append(x_new)
    return(x,i)
(x1,i1) = GD(-3,0.1)
(x2,i2) = GD(5,0.1)
print('Solution x1 = %.4f, cost = %.4f, after %d iteration' %(x1[-1], cost(x1[-1]) ,i1))
print('Solution x2 = %.4f, cost = %.4f, after %d iteration' %(x2[-1], cost(x2[-1]) ,i2))