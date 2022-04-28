import numpy as np
import scipy
import matplotlib.pyplot as plt
from euler import *

def test_newton():
    gamma = 1.4
    # Boundary Conditions of Riemann Problem
    q_l = np.array([3, 0, 3])
    q_r = np.array([1, 0, 1])

    # Time Domain and Space Domain
    t0 = 0
    t_final = 1/4
    Lx = 0.5

    # Setting of space and time discretization
    mX = 100
    
    x = np.linspace(-Lx, Lx, mX + 2) # Two more ghost cell on boundaries

    delta_x = x[1] - x[0]
    CFL = 0.9
    delta_t = CFL * delta_x  # Time step

    q = np.zeros((len(q_l), mX+2))
    q_new = np.zeros((len(q_l), mX+2))

    for i in range(len(q_l)):
        q[i] = (x< 0) * q_l[i] + (x>= 0) * q_r[i]


    # Initialize Iteration
    figs = []
    count = 0
    tol = 1e-5
    t = 0
    MaxIT = 1e5
    r = 1e5
    while t < t_final:
        for xn in range(mX):
            # xi = x[xn]/(tn+1)*delta_t
            # q[tn+1, xn, :] = solver(xi)
            solver_l = Euler_1D_roe(q[:, xn], q[:, xn+1])
            solver_r = Euler_1D_roe(q[:, xn+1], q[:, xn+2])

            # Average within cell xn
            for ind in range(len(q_l)):
                q_new[ind, xn+1] = (integrate.quad(lambda xi: solver_r(xi)[ind], -0.5*delta_x, 0)[0] + integrate.quad(lambda xi: solver_l(xi)[ind], 0, 0.5*delta_x)[0]) / delta_x
        
        # Ghost Cells
        q_new[:, 0] = q_new[:, 1]
        q_new[:, mX+1] = q_new[:, mX]

        

        # update
        q = q_new
        t += delta_t
            
        figs.append(plt.figure())
        
        Q = [q[0], q[1]/q[0], (q[2]-0.5*q[1]**2/q[0])*(gamma -1)]
        var_name = ['rho', 'u', 'E']

        axes = figs[-1].subplots(3, 1)
        for var in range(len(q_l)):
            axes[var].plot(x, Q[var], '-', linewidth=2, label=var_name[var])
            axes[var].set_xlim(x[0], x[-1])
            # axes[var].set_ylim(-0.1, 3.3)
            axes[var].legend()
        axes[0].set_title('t = %.03f' % t)

        figs[-1].savefig('./results/images/plot%03d.png' % count)
        print('fig%03d.png saved'% count)
        count += 1
        plt.close(figs[-1])
    
    # Animate the solution
    images = animation.make_images(figs)
    imageio.mimsave('results/shock_tube_detail.gif', images)



if __name__ == "__main__":
    test_newton()