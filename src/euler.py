from termios import TAB0
import numpy as np
import scipy
import scipy.integrate as integrate
import matplotlib.pyplot as plt
import imageio
import utils.animation as animation

def solve_scipy(t, U_0, f):
    """Solve the system using scipy.integrate"""
    # YOUR CODE HERE

    integrator = integrate.ode(f)
    integrator.set_integrator("dopri5")
    integrator.set_initial_value(U_0)

    U = np.empty((2, t.shape[0]))
    U[:, 0] = U_0
    for (n, t_n) in enumerate(t[1:]):
        integrator.integrate(t_n)
        if not integrator.successful():
            break
        U[:, n + 1] = integrator.y
    return np.transpose(U)

def channelflow():
    # Time Domain and Space Domain
    T = 10
    Lx = 10
    Ly = 10

    # Setting of space and time discretization
    nT = 100
    mX = 100
    nY = 100
    
    t = np.linspace(0, T, nT + 1)
    x = np.linspace(0, Lx, mX + 1)
    y = np.linspace(0, Ly, nY + 1)

    dx = x[1] - x[0]
    dy = y[1] - y[0]

    
    # Setting of problem
    # Boundary Condtion
    x0_bc = 'periodic'
    xn_bc = 'periodic'
    y0_bc_ = 'neumann'
    yn_bc_ = 'neumann'

    u0 = np.zeros((mX+1, nY+1))


def f(q):
    """
    
    """
    gamma = 1.4
    p = (q[1] - 0.5*q[0]*q[2]**2)*(gamma -1)
    # p = -1

    # f(q) at x_j
    ans = np.ndarray(q.shape[0])
    ans[0] = q[0] * q[2]
    ans[1] = q[2] * ( q[1] + p )
    ans[2] = q[0] * q[2]**2 + p

    return ans

def fx(q, delta_x):
    """

    """
    ans = np.ndarray(q.shape)
    # # Center Difference
    # for j in range(q.shape[1]-2):
    #     ans[:, j+1] = (f(q[:, j+2,]) - f(q[:, j])) / 2 / delta_x
    # ans[:, 0] = (f(q[:, 1]) - f(q[:, -2])) / 2 / delta_x
    # ans[:, -1] = (f(q[:, 1]) - f(q[:, -2])) / 2 / delta_x

    # Upwind
    for j in range(q.shape[1]-1):
        ans[:, j] = (f(q[:, j+1,]) - f(q[:, j])) / delta_x
    ans[:, -1] = (f(q[:, 1]) - f(q[:, -1])) / delta_x
    return ans


def time_integral(Q, delta_t, delta_x):
    diff = delta_t * fx(Q, delta_x)
    rho = Q[0] + delta_t * fx(Q, delta_x)[0]
    E = (Q[0]*Q[1] + delta_t * fx(Q, delta_x)[1] )/ rho
    u = Q[0]*Q[2] + delta_t * fx(Q, delta_x)[2] 
    ans = np.array([rho, E, u])
    return ans


def euler1d():
    # Time Domain and Space Domain
    t0 = 0
    T = 1
    Lx = 1

    # Setting of space and time discretization
    nT = 100
    mX = 100
    
    x = np.linspace(0, Lx, mX + 1)

    delta_x = x[1] - x[0]
    CFL = 0.9
    delta_t = CFL * delta_x  # Time step

    # Setting of problem
    # Boundary Condtion
    x_bc = 'periodic'

    # State Variable Function q_t + f_x(q) = 0; q = [rho, rho*u, E]
    # initial condition
    q0 = np.zeros((3, mX + 1))
    left = lambda x: x<Lx/2
    right = lambda x: x>Lx/2

    variable = [r'$ \rho $', 'E', 'u']

    q0[2, :] = left(x) * 0
    q0[0, :] = left(x) * 1 + right(x) * 0.125
    q0[1, :] = left(x) * 1 + right(x) * 0.1

    Q = np.empty((len(q0), mX+1))
    Q = q0
    Q_new = np.empty(Q.shape)

    t = t0 


    figs = []
    count = 0
    while t< T:
        Q_new = time_integral(Q, delta_t, delta_x)
        Q = Q_new.copy()
        t += delta_t
        count += 1

        figs.append(plt.figure())
        axes = figs[-1].subplots(3, 1)
        for var in range(len(Q)):
            axes[var].plot(x, Q[var], 'o-', linewidth=2, label=variable[var])
            axes[var].set_xlim((x[0], x[-1]))
            axes[var].set_ylim((-5, 5))
            axes[var].legend()
        axes[0].set_title('t = %.03f' % t)

        figs[-1].savefig('./results/images/plot%03d.png' % count)
        plt.close(figs[-1])
    
    # Animate the solution
    images = animation.make_images(figs)
    imageio.mimsave('results/1.gif', images)
    



if __name__ == "__main__":
    euler1d()