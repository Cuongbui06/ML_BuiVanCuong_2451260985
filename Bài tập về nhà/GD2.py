def cost(x):
    return (1/3)*(x**3) - x
def grad(x):
    return x**2 - 1
def GD(x0, eta):
    x = [x0]
    for i in range(1, 100):
        x_new = x[-1] - eta * grad(x[-1])
        x.append(x_new)
        if abs(grad(x_new)) < 1e-3:
            break
    return (x, i)
(x1, i1) = GD(2, 0.1)
(x2, i2) = GD(-0.5, 0.1)
print('Solution x1 = %.4f, cost = %.4f, after %d iterations' % (x1[-1], cost(x1[-1]), i1))
print('Solution x2 = %.4f, cost = %.4f, after %d iterations' % (x2[-1], cost(x2[-1]), i2))