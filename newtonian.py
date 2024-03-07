# base class for solving Newtonian equation of motion

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

from collections import deque
import matplotlib.animation as animation
from matplotlib import colormaps as cmap

class Newtownian:
    '''Set up a Newtownian system with point mass and gravitational field.
    
    Class variables
    ---------------
    dimension: int
        Dimension of the physical space. Usually set to be 3.
    major, minor: list
        Contains all created instances. major objects have mass while minor objects is massless.
    t: ndarray(folat)
        Stores the time of the system.
    g_ext: None or function
        The external field. Initialized as None.

    Instance variables
    ------------------
    is_major: bool, optional
        Whether this instance is considered as major stars. Default: False.
    name: str, optional
        Name of this instance. Default: '' (empty string).
    step: int
        Total number of steps evolved.
    Gm: float
        G * (mass of this instance). G: gravitational constant.
        If is_major=True, this variable can be randomly set.
    x: ndarray(float)
        Trajectory of this instance.
    v: ndarray(float)
        Velovity of this instance.

    Instance methdods
    -----------------

    Class methods
    -------------
    '''
    
    # class setup
    # -----------
    dimension: int = 3  # dimension of space, 2d is easier to test
    major = []  # major: condiser its gravity
    minor = []  # minor: ignore its gravity
    major_Gm = []  # mass of major instances
    t = np.array([0,])
    g_ext = None
    
    def __init__(self, Gm: float, x0: np.ndarray, v0: np.ndarray, name=None, major = False) -> None:
        if self.t.size != 1:
            raise ValueError('Not initializing instance at t = 0. Reset the class first.')
        if major:
            self.major.append(self)
            self.major_Gm.append(Gm)
            self.Gm = Gm
            self.is_major = True
        else:
            self.minor.append(self)
            self.is_major = False
        # x, v staor the results as (n,3)-array
        # t stors the results as (n,)-array
        self.name = name
        self.x = np.array(x0).reshape(1,self.dimension)
        self.v = np.array(v0).reshape(1,self.dimension)

    @classmethod  # update needed
    def __reset__(cls, keep_instances=False) -> None:
        '''Reset the class.
        If keep_instances = False, delete all class members. Otherwise,
        assign class variables to every current instances then reset the class.
        '''
        if keep_instances:
            temp_major = cls.major.copy()
            temp_minor = cls.minor.copy()
            temp_t = cls.t.copy()
            for star in cls.major+cls.minor:
                star.major = temp_major
                star.minor = temp_minor
                star.t = temp_t

        # reset the class
        cls.major = []
        cls.minor = []
        cls.major_Gm = []
        cls.t = np.array([0,])
        cls.g_ext = None

    @classmethod
    def add_external_field(cls, fun) -> None:
        '''Add an external field. Should be in form fun(x,t) -> ndarray(N, dimension),
        where x is ndarray(N, dimension).
        '''
        cls.g_ext = fun
    
    @classmethod
    def get_stars(cls) -> list:
        '''Get all instances.'''
        return cls.major + cls.minor
    
    @classmethod
    def get_mass(cls) -> np.ndarray:
        '''Get mass of all major instances. With the same order as they are in cls.get_stars().'''
        return np.array(cls.major_Gm)

    # integration
    # -----------
    @classmethod
    def G_n(cls, t: float, X: np.ndarray) -> np.ndarray:
        '''Calculate gravitational field g_n which will be used in function f_ty.
        
        Parameters
        ----------
        t: float, time.
        X: ndrray, the last axis is spatial coordinate.
        
        Return
        ------
        g_n: ndarray, the gravitational field.
        '''
        masses = cls.get_mass()
        num_m = masses.size  # number of major instances
        dX = X[:, np.newaxis] - X[:num_m]  # relative position
        dX_norm = np.linalg.norm(dX, axis=-1)  # distance
        g_n = np.divide(masses, dX_norm**3, out=np.zeros_like(dX_norm),  # gfield from each major instances
                        where=(dX_norm!=0))[:,:,np.newaxis] * dX         # self-gravity = 0
        g = - np.sum(g_n, axis=1)  # total gfield

        if cls.g_ext is not None:
            g += cls.g_ext(X, t)

        return g

    @classmethod
    def f_ty(cls, t, y) -> np.ndarray:
        '''Function for solving y' = f(t,y).

        Parameters
        ----------
        y: ndarray
            y = (x11, x12,..., x(n*dim), v1,..., v(n*dim))
        t: float
            time

        Return
        ------
            f(t,y) = (v1, v2,..., v(n*dim), a1, a2,..., a(n*dim))
        '''
        
        num = np.size(cls.get_stars())  # number of instances
        x_n, v_n = y.reshape(2, num, cls.dimension)

        g_n = cls.G_n(t, x_n)
        return np.concatenate((v_n, g_n)).ravel()

    @classmethod
    def evo(cls, tspan, maxstep=np.inf, atol=1e-6, rtol=1e-3, terminate=None):
        '''Evolve the system within interval tspan.
        Call scipy.integrate.solve_ivp to integrate the equation of motion.
        After integration, store the results to each instances.

        Parameters and Return
        ---------------------
        tspan: (ti, tf)
            integration interval
        maxstep: float
            maxminum allowed integration step
        atol: float
            total error
        rtol: float
            relative error
        terminate: float or None
            Stop integration process and return result if two objects come too close.
        See scipy.integrate.solve_ivp for details.
        '''

        stars = cls.get_stars()
        # create initial condition
        num = np.size(stars)  # number of instances
        y0 = np.empty((num*2, cls.dimension))
        for i in range(num):
            y0[i] = stars[i].x[-1]
        for i in range(num):
            y0[num+i] = stars[i].v[-1]
        y0 = y0.ravel()

        # integrate
        tspan += cls.t[-1]

        if terminate is None:
            result = solve_ivp(cls.f_ty, tspan, y0, max_step=maxstep, atol=atol, rtol=rtol)#, dense_output=True)
        else:
            def terminate_evo(t, y):
                '''Stop integrating when to objects are too close.'''
                num = np.size(cls.get_stars())  # number of instances
                num_m = np.size(cls.major)  # number of major instances
                x_n = y[:num*cls.dimension].reshape(num, cls.dimension)
                dx = x_n[:num_m, np.newaxis] - x_n  # find the position difference matrix
                distance = np.linalg.norm(dx, axis=-1)  # calculate the distance (matrix)
                return np.min(distance[np.triu_indices(num_m, 1, num)]) - terminate  # use only upper triangular elements
            terminate_evo.terminal = True
            result = solve_ivp(cls.f_ty, tspan, y0, max_step=maxstep, atol=atol, rtol=rtol, events=terminate_evo)#, dense_output=True)
        
        # update data
        for i in range(num):
            stars[i].x = np.concatenate((stars[i].x, result.y[cls.dimension*i:cls.dimension*(i+1)].T[1:]), axis=0)
        for i in range(num):
            stars[i].v = np.concatenate((stars[i].v, result.y[cls.dimension*(num+i):cls.dimension*(num+i+1)].T[1:]), axis=0)
        cls.t = np.concatenate((cls.t[:-1], result.t))

        return result

    # plot
    # ----
    @classmethod
    def static_plot(cls):
        '''Plot trajectory of all stars. Only dimension 2 and 3 are supported.'''
        if cls.dimension == 2:
            fig = plt.figure()
            ax = fig.add_subplot(autoscale_on=True)  #, xlim=(-5, 5), ylim=(-5, 5))
            ax.set_aspect('equal')
            ax.grid()
            for star in cls.get_stars():
                ax.plot(star.x.T[0], star.x.T[1], label=star.name)
        elif cls.dimension == 3:
            raise ValueError("Haven't finished the 3d plot.")
        else:
            raise ValueError("Can only plot 2d and 3d case. Higher dimension not supported.")
        
        plt.legend()
        plt.show()

    @classmethod
    def dynamic_plot(cls, history_len=500, interval=10):
        '''Plot evolution of the stats. Only dimension of 2 and 3 are supported.
        If running in jupyter-notebook, please "plt.close()" to close the plot.
        history_len: int. Maximum points to show for trajectory.
        rertun: The animation object.
        '''
        
        if cls.dimension == 2:
            stars = cls.get_stars()
            maxcoordinate = 0  # find the maximum coordinate value to fix the scale of the plot
            mincoordinate = 0
            color = cmap['rainbow']  # give each star a different color
            color_number = np.linspace(0,1,len(stars))
            for star in stars:
                maxcoordinate = max(maxcoordinate, np.max(star.x))
                mincoordinate = min(mincoordinate, np.min(star.x))
            fig = plt.figure()
            ax = fig.add_subplot(autoscale_on=False, 
                xlim=[1.1*mincoordinate, 1.1*maxcoordinate], ylim=[1.1*mincoordinate, 1.1*maxcoordinate])
            ax.set_aspect('equal')
            ax.grid()
            ax.set_xlabel('$x$')
            ax.set_ylabel('$y$')
            traces = []  # traces of each star
            history_xs = []  # x(t) for each star
            history_ys = []  # y(t) for each star
            positions = []  # position of all stars at time t
            for n in range(len(stars)):
                traces.append(ax.plot([], [], '-', lw=1, color=color(color_number[n]))[0])
                positions.append(ax.plot([], [], 'o', ms=5, color=color(color_number[n]), label=stars[n].name)[0])
                history_xs.append(deque(maxlen=history_len))
                history_ys.append(deque(maxlen=history_len))
            time_template = 'time = %.1f unit'
            time_text = ax.text(0.05, 1.02, '', transform=ax.transAxes)

            def animate(i):
                if i == 0:
                    for temp in history_xs:
                        temp.clear()
                    for temp in history_ys:
                        temp.clear()

                for i_star in range(len(stars)):
                    history_xs[i_star].appendleft(stars[i_star].x[i, 0])
                    history_ys[i_star].appendleft(stars[i_star].x[i, 1])
                    traces[i_star].set_data(history_xs[i_star], history_ys[i_star])
                    positions[i_star].set_data(stars[i_star].x[i, 0], stars[i_star].x[i, 1])
                time_text.set_text(time_template % cls.t[i])

                return positions, traces, time_text

            ani = animation.FuncAnimation(
                fig, animate, np.size(cls.t), blit=False, interval=20)
            plt.legend()
            plt.show()
            return ani

        elif cls.dimension == 3:
            stars = cls.get_stars()
            # initialize figure
            maxcoordinate = 0  # find the maximum coordinate value to fix the scale of the plot
            mincoordinate = 0
            color = cmap['rainbow']  # give each star a different color
            color_number = np.linspace(0,1,len(stars))
            for star in stars:
                maxcoordinate = max(maxcoordinate, np.max(star.x))
                mincoordinate = min(mincoordinate, np.min(star.x))
            
            fig = plt.figure()
            ax = fig.add_subplot(projection="3d", autoscale_on=False, 
                                xlim=[1.1*mincoordinate, 1.1*maxcoordinate], 
                                ylim=[1.1*mincoordinate, 1.1*maxcoordinate], 
                                zlim=[1.1*mincoordinate, 1.1*maxcoordinate])
            ax.set_xlabel('$x$')
            ax.set_ylabel('$y$')
            ax.set_zlabel('$z$')
            time_template = 'time = %.1f unit'
            time_text = ax.text2D(0., 1.02, '', transform=ax.transAxes)
            
            # Create lines initially without data
            traces = [ax.plot([],[],[], '-', lw=1, color=color(color_number[n]))[0] for n in range(len(stars))]
            positions = [ax.plot([],[],[], 'o', ms=5, color=color(color_number[n]), label=stars[n].name)[0] for n in range(len(stars))]

            # update data
            def update_lines(num):
                for trace, position, walk in zip(traces, positions, stars):
                    # NOTE: there is no .set_data() for 3 dim data...
                    trace.set_data(walk.x[max(0, num-history_len):num, :2].T)
                    trace.set_3d_properties(walk.x[max(0, num-history_len):num, 2])
                    position.set_data(walk.x[num, :2].reshape(2,1))
                    position.set_3d_properties([walk.x[num, 2]])
                time_text.set_text(time_template % cls.t[num])
                return  traces, positions, time_text

            # Creating the Animation object
            ani = animation.FuncAnimation(
                fig, update_lines, cls.t.size, interval=interval)

            plt.legend()
            plt.show()
            return ani

        else:
            raise ValueError("Only dimension 2 and 3 plottings are supported!")
