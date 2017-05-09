import mshr
import numpy as np
import matplotlib.pyplot as plt
from dolfin import *

__author__ = 'njcs4'

def naca(x, c, t):
    y_t = 5*t*c*(0.2969*(x/c)**0.5 - 0.1260*(x/c) - 0.3516*(x/c)**2 + 0.2843*(x/c)**3 - 0.1036*(x/c)**4)
    return y_t

c = 1.0
t = 12
x1 = (np.logspace(0, 1, 10)-1.0)/9.0*0.00261456
x2 =  (np.logspace(0, 1, 100)[2:]-1.0)/9.0
x = np.append(x1, x2)

y = naca(x, c, t/100.0)
y[-1] = 0.0
y2 = -y
plt.plot(x, y)
plt.plot(x, y2)

points = [Point(x[j], y[j]) for j in range(len(x))]
points += [Point(x[j], y2[j]) for j in reversed(range(len(x)-1))]

points.reverse()

# poly = mshr.Rectangle(Point(-4.0, -4.0), Point(5.0, 4.0))
airfoil = mshr.Polygon(points)
airfoil = mshr.Extrude2D(airfoil, 1.0)
domain = mshr.Box(Point(-4.0, -4.0, -4.0), Point(4.0, 4.0, 4.0))

mesh = mshr.generate_mesh(domain - airfoil, 10, 'tetgen')

XDMFFile("naca%.4d_3d.xdmf" % t).write(mesh)

