from termios import TAB0
import numpy as np
import scipy
import scipy.integrate as integrate
import matplotlib.pyplot as plt
import imageio
import utils.animation as animation
    

def roe_average(q_l, q_r, gamma):
    [rho_l, rhou_l, E_l] = q_l
    [rho_r, rhou_r, E_r] = q_r
    
    u_l = rhou_l / rho_l
    u_r = rhou_r / rho_r

    H_l = (E_l + ( E_l - 0.5*rho_l*u_l**2 )*(gamma - 1)) / rho_l
    H_r = (E_r + ( E_r - 0.5*rho_r*u_r**2 )*(gamma - 1)) / rho_r

    u_hat = (np.sqrt(rho_r)*u_r + np.sqrt(rho_l)*u_l)/(np.sqrt(rho_r)+np.sqrt(rho_l))
    H_hat = (np.sqrt(rho_r)*H_r + np.sqrt(rho_l)*H_l)/(np.sqrt(rho_r)+np.sqrt(rho_l))
    c_hat = np.sqrt((gamma - 1) * (H_hat - u_hat**2/2))
    return u_hat, H_hat, c_hat


def Euler_1D_roe(q_l, q_r, gamma=1.4):
    # # Boundary Conditions of Riemann Problem
    # q_l = [3, 0, 3]
    # q_r = [1, 0, 1]

    # Obtain the characteristic speed (eigenvalues) of linear Operator
    u_hat, H_hat, c_hat = roe_average(q_l, q_r, gamma)
    # eigenvectors
    r1 = np.array([1, u_hat-c_hat, H_hat - u_hat*c_hat])
    r2 = np.array([1, u_hat, 0.5*u_hat**2])

    # Wave Weight
    dq = q_r - q_l
    alpha2 = dq[0] + (gamma - 1)*(u_hat*dq[1]-dq[2])/c_hat**2
    alpha3 = (dq[1] + (c_hat - u_hat)*dq[0] - c_hat * alpha2) /2/c_hat
    alpha1 = dq[0] - alpha2 - alpha3

    # Intermedia State
    q_l_star = q_l + alpha1 * r1
    q_r_star = q_l_star + alpha2 * r2

    # Result
    def solver(xi):
        q = np.zeros((len(q_l)))
        for ind in range(len(q_l)):
            q[ind] = (xi<u_hat-c_hat) * q_l[ind] + (xi<u_hat) * (u_hat-c_hat<= xi) * q_l_star[ind] + (xi<u_hat+c_hat) * (u_hat <= xi) * q_r_star[ind] + (xi>=u_hat+c_hat) * q_r[ind]
        return q
    return solver

def shock_tube():
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

    figs = []
    count = 0
    # solver = Euler_1D_roe(q_l, q_r)
    t = 0
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

        figs[-1].savefig('./results/images_shock/plot%03d.png' % count)
        print('fig%03d.png saved'% count)
        count += 1
        plt.close(figs[-1])
    
    # Animate the solution
    images = animation.make_images(figs)
    imageio.mimsave('results/shock_tube_detail.gif', images)

def test_euler():
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

    figs = []
    count = 0
    solver = Euler_1D_roe(q_l, q_r)
    t = 0
    while t < t_final:
        for xn in range(mX+2):
            xi = x[xn]/t
            q[:, xn] = solver(xi)

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

        figs[-1].savefig('./results/images_riemann/plot%03d.png' % count)
        print('fig%03d.png saved'% count)
        count += 1
        plt.close(figs[-1])
    
    # Animate the solution
    images = animation.make_images(figs)
    imageio.mimsave('results/riemann_shock.gif', images)


if __name__ == "__main__":
    test_euler()
    # shock_tube()