# created 20240305
# make parameter space to include phi_c at the beginning
# deleted noise, never used

import numpy as np
import p_tqdm
import pickle

from newtonian import Newtownian

#########################################################################
# unit in this code
# length: 1 = 1e13 m, time: 1=1e13 s

path = r'D:\OneDrive - University of Florida\Programs\Caustic\test\\'  #  r'D:\OneDrive - University of Florida\Programs\Caustic\a-ep plot\\'  # file path

#########################################################################
# control parameters
g_c = 2e-3  # std: 2e-3, unit: g/(cm^2*kpc^0.5)
v_c = 1.  # std: 1, unit: km/s
omega = 1/26.7  # std: 1/26.7, unit: 1/Myr
GM = 1.32733e7  # mass of the sun

theta_values = (np.deg2rad(90),)  # angle between orbital plane and caustic plane
phic_values = (np.deg2rad(180),)  # orientation angle
a_count = 10  # points to pick up in each parameter
ep_count = 10
phi_count = 10
min_Oort_radius = 15  # 1e3 Au, inner boundary ..., should be <45
a_max, a_min = 3000, 45  # range of semi-major axis

filename = r'No_Osc, 1e13m, 1e13s, v={}, A={},'.format('%.2f' %v_c, '%.3f' %g_c)  # Name of the files

#########################################################################
g_c *= 8*np.pi*6.6743/np.sqrt(3.085)  # convert to operating unit
v_c *= 1.e3  # convert to operating unit
omega /= 3.16  # convert to operating unit
dimension = 3  # dimension must be 3
Newtownian.dimension = dimension
# basic class for calculation
class caustic_wall(Newtownian):
    '''An object close to a surface caustic.
    The gravitational field near a cuasitc ring is
        g(sigma) = 8pi*G*A*sqrt(sigma) Theta(sigma)
    where sigma is the distance to the surface caustic, Theta is the unit step function.
    We align the outer surface normal of the caustic surface in negative x direction.
    '''

    # cuastic parameters
    caustic_A = 0
    caustic_x0 = 0
    caustic_v = 0
    caustic_x = np.empty(0)

    # galactic effective potential parameters
    omega2 = 0

    @classmethod
    def set_parameters(cls, xc0:float, vc:float, gc:float, omega:float) -> None:
        '''Set parameters of the caustic wall and galactic effective potential.
        
        Parameters
        ----------
        xc0: float
            x coordinate of the caustic wall when time t=0.
        vc: float
            Expansion velocity of the caustic wall. It is assumed to be constant.
        gc: float
            Density profile of the caustic wall.
                gc = 8 pi G A
                g(sigma) = gc sqrt(sigma)
        omega: float
            Restoring force. At R0=8.5 kpc
                omega_r = sqrt(2) v_rot/R0 = 1/(26.7 Myr)
                omega_z = sqrt(4 pi G rho_disk) = 1/(9.87 Myr)
        '''
        if cls.t.size > 1:
            print('Warning: Setting parameters at a non-initial time!')
        cls.caustic_x0 = xc0
        cls.caustic_v = vc
        cls.caustic_g = gc  # 8 pi G A
        cls.omega2 = (omega)**2

    @classmethod
    def effective_gfield(cls, t, X) -> np.ndarray:
        '''Given position X and time t, calculate the effective gravitational field of the
        caustic and the effective galactic potential near the equilibrium position. The
        parameters are defined as class variables.
        
        Parameters
        ----------
        X: np.ndarray with shape (n, dimension)
            Cartesian coordinate.
        t: float
            Time.
            
        Return
        ------
        effective_g_field: np.ndarray
            The gravitational field g produced by the caustic ring at position X. Has the same
            shape as X. That is, [(gx1, gy1, gz1),(gx2, gy2, gz2),...].
        '''
        g = np.zeros_like(X)
        sigma = X[:,0] - cls.caustic_x0 - cls.caustic_v*t  # distance to caustic surface
        gx = g[:, 0]
        gx[sigma>0] += -cls.caustic_g * np.sqrt(sigma[sigma>0])  # gfield by caustic
        gx += -cls.omega2 * X[:,0]
        return g
    
    @classmethod
    def __reset__(cls, keep_instances=False) -> None:
        '''Reset the class.
        If keep_instances = False, delete all class members. Otherwise,
        assign class variables to every current instances then reset the class.
        Parameters of the caustic will NOT be kept in either case.
        '''
        # cuastic parameters
        cls.caustic_A = 0
        cls.caustic_x0 = 0
        cls.caustic_v = 0
        cls.caustic_x = np.empty(0)

        # galactic effective potential parameters
        cls.v_rot = 0
        cls.R0 = 1
        super().__reset__(keep_instances)

    @classmethod
    def G_n(cls, t: float, X: np.ndarray) -> np.ndarray:
        return super().G_n(t, X) + cls.effective_gfield(t, X)
    
    @classmethod
    def evo(cls, tspan, maxstep=np.inf, atol=1e-6, rtol=1e-3, terminate=None):
        result = super().evo(tspan, maxstep, atol, rtol, terminate)
        cls.caustic_x = np.concatenate((cls.caustic_x[:-1], cls.caustic_x0+cls.caustic_v*result.t))
        return result

x_c0 = 0.  # we set caustic boundary at minimum of galatical potential at t=0
# create final parameters to do calculation
a_range = np.linspace(a_max, a_min, a_count)  # use linear space
ep_func = lambda a: np.linspace(0, min(1-min_Oort_radius/a, 1), ep_count)  # eccentricity generation function
# change psi to phi
def psi_to_phi(psi:np.ndarray, ep:float) -> np.ndarray:
    '''Change psi to phi.
    psi: polar angle of an ellipse when its center is at the origin.
    phi: polar angle of an ellipse when its right focal point is at the origin
    return: phi = phi(psi)
    algorithm:
        1. x/a = sqrt((1-ep^2)/(1-ep^2 cos^2(psi)))
        2. r = sqrt(x^2 + c^2 - 2xc cos(psi))
        3. x^2 = r^2 + c^2 - 2rc cos(pi - psi)
        where x is the radial distance corresponds to psi,
        r is the radial distance corresponds to phi.
        Since arccos is used, only psi,phi in [0,pi] is valid. For other angles, more efforts need to be made.
        Here we use phi -> -phi when psi in [-pi,0].'''
    cospsi = np.cos(psi)  # cos(psi)
    xoa2 = (1-ep**2) / (1-ep**2*cospsi**2) # (x/a)**2
    cosphi = (ep-np.sqrt(xoa2)*cospsi) / (np.sqrt(ep**2 - 2*ep*np.sqrt(xoa2)*cospsi+xoa2))
    cosphi[cosphi>1] = 1
    cosphi[cosphi<-1] = -1  # in case of numerical error
    phi = np.pi - np.arccos(cosphi)
    return np.where(psi>0, phi, -phi)

psi_range = np.pi * np.linspace(-1, 1, phi_count, endpoint=False)  # psi angle, related to phi angle

parameter_space = [(i_a, ep, phi, theta, phi_c)  # 4d parameter space
                   for i_a in range(a_count)
                   for ep in ep_func(a_range[i_a])
                   for phi in psi_to_phi(psi_range, ep)
                   for theta in theta_values
                   for phi_c in phic_values]

#####################
# start calculation #
#####################

# evolve back to find a proper time to start integration
# also generate initial conditions corresponding to distance between sun and caustic boundary
'''For small oscillation case, set initial state as:
    x_sun0 = x_c0, v_sun = v_c
Use a simple code to find the time needed to reach maximum distance. Or just make tspan large.'''
x_sun0 = np.zeros(dimension)
x_sun0[0] = x_c0
v_sun0 = np.zeros(dimension)
v_sun0[0] = v_c
caustic_wall.__reset__()
caustic_wall.set_parameters(x_c0, v_c, g_c, omega)
sun = caustic_wall(1, x_sun0, v_sun0, 'backward')
tspan = [0.,-150.]
result = caustic_wall.evo(tspan, 0.1, 1e-9, 1e-9)  # compared wuth step size=0.01, the difference is at 1e-5
distance = sun.x.T[0] - caustic_wall.caustic_x  # distance to be compared with 2.5a
# generate initial condition corresponding to 'a'
x_sun0s, v_sun0s, t_start = [], [], []
for a in a_range:
    i = np.argwhere(distance >= 2.5*a)[0,0]  # index when the distance reaches 2.5a (distance is decreasing)
    x_sun0s.append(sun.x[i])
    v_sun0s.append(sun.v[i])
    t_start.append(result.t[i])

# evolve forward to find a proper time to stop integration
# similar to previous part
x_sun0 = np.zeros(dimension)
x_sun0[0] = x_c0
v_sun0 = np.zeros(dimension)
v_sun0[0] = v_c
caustic_wall.__reset__()
caustic_wall.set_parameters(x_c0, v_c, g_c, omega)
sun = caustic_wall(1, x_sun0, v_sun0, 'forward')
tspan = [0.,75.]
result = caustic_wall.evo(tspan, 0.1, 1e-9, 1e-9)
distance = caustic_wall.caustic_x - sun.x.T[0]  # distance to be compared with 2.5a
# generate initial condition corresponding to 'a'
t_stop = []
for a in a_range:
    i = np.argwhere(distance >= 2*a)[0,0]  # expression for distance changed, so it's now increasing
    t_stop.append(result.t[i])  # total integration time

# sun-commet-caustic system
# initialize an object based on given parameters
def initialize_planet(a:float, ep:float, phi:float, theta:float, phi_c:float) -> tuple[np.ndarray, np.ndarray, float]:
    '''Initialize a planet based on given parameters. Only generates 3d objects.

    Parameters
    ----------
    a: semi-major axis
    ep: eccentricity
    phi: orbital phase of the planet
    theta: inclation angle (i.e. polar angle in spherical coordinate)
    phi_c: orientation angle

    Method
    ------
    The caustic wall and galatic effective potential is in x-diraction.
    We first calculate the orbit in y-z plane based on given semi-major axis "a" and eccentricity 
    "ep" with the Sun at the origin, then initialize an object on this orbit with the given polar angle
    "phi"(in polar coordinate), then rotate the star with some inclination algle "theta" with respect
    to any axis perpendicular to x-axis. Here we choose z-axis.

    An ellipse with semi-major axis a, eccentricity ep in polar coordinate is
        1/r(phi) = (1-ep*cos(phi-phi_c))/(a(1-ep^2))
    with potential given by V(r) = GM/r. The velocity at a given point is
        v = -[l/r]*[ep*sin(phi-phi_c)/(1-ep*cos(phi-phi_c))] r^hat + [l/r] phi^hat
        l = sqrt[a*GM*(1-ep^2)]
    The polor coordinate is related to Cartisian coodrinate by usual relation
        y = r*cos(phi),
        z = r*sin(phi).
    The weight should be propotional to 1/phi^dot, which is
        p(phi) = [(1-ep^2)^(3/2)/(2*pi)][1/(1-ep*cos(phi-phi_c))^2].

    Returns
    -------
    x: ndarray, position of the object
    v: ndarray, velocity of the object
    p: float, probability weight related to "phi"
    '''
    # calculate orbit and create object
    r = a*(1-ep**2)/(1-ep*np.cos(phi-phi_c))  # distance to the sun, should cos(phi-phi0), but take phi0=0
    l = np.sqrt(a*GM*(1-ep**2))  # angular momentum
    v_r = - l/r * ep*np.sin(phi-phi_c) / (1-ep*np.cos(phi-phi_c))  # velocity in r^hat direction
    v_phi = l/r  # velocity in phi^hat direction
    y_ = r * np.cos(phi)  # y position
    z_ = r * np.sin(phi)  # z position
    v_y = v_r * np.cos(phi) - v_phi * np.sin(phi)  # y velocity
    v_z = v_r * np.sin(phi) + v_phi * np.cos(phi)  # z velocity
    x = np.array([0., y_, z_])  # Cartisian position
    v = np.array([0., v_y, v_z])  # Cartisian velocity
    # rotate object
    R_z = np.array([[np.cos(theta), -np.sin(theta), 0],  # rotation matrix along z-axis
                    [np.sin(theta), np.cos(theta),  0],
                    [0,             0,              1]])
    x = R_z@x  # do the rotation
    v = R_z@v
    # calculate weight of phi
    p = (1-ep**2)**(1.5)/(2*np.pi)/(1+ep*np.cos(phi-phi_c))**2
    return x, v, p

# main calculation model
def calculate(args:tuple) -> tuple[bool, float, np.ndarray, np.ndarray, float]:
    '''Main calculation part.

    Parameters
    ----------
    args: tuple, should have form
        args = [i_a, ep, phi, theta, phi_c]

    Returns
    -------
    status: whether the integration is succeed (should always be True)
    r_min: minimum distance to the sun the object reached
    x_p(0), v_p(0): initial condition
    x_p(t), v_p(t): final state
    p: weight
    '''
    i_a, ep, phi, theta, phi_c = args
    a = a_range[i_a]
    # initialize
    caustic_wall.__reset__()
    caustic_wall.set_parameters(x_c0, v_c, g_c, omega)
    sun = caustic_wall(GM, x_sun0s[i_a], v_sun0s[i_a], 'sun', True)
    x_p, v_p, probability = initialize_planet(a, ep, phi, theta, phi_c)
    planet = caustic_wall(1, x_p+x_sun0s[i_a], v_p+v_sun0s[i_a],'{}'.format((a, ep, phi, theta)))
    # start integration
    tspan = [t_start[i_a], t_stop[i_a]]
    result = caustic_wall.evo(tspan, 1., 1e-9, 1e-9, 1e-2)
    if result.status != -1:
        planet.x -= sun.x
        planet.v -= sun.v
        r_min = np.min(np.linalg.norm(planet.x, axis=-1))
        # save data
        return True,  r_min, x_p,  v_p,  planet.x[-1], planet.v[-1], probability
    else:
        return False, None,  None, None, None,         None,         None

# create output files
with open(path+filename+r' -config.bin', 'wb') as configfile:
    pickle.dump("Variable name: (filename, omega, v_c, g_c, GM)", configfile)
    pickle.dump((filename, omega, v_c, g_c, GM), configfile)
    pickle.dump((a_count, ep_count, phi_count, np.size(theta_values), np.size(phic_values)), configfile)  # shape of output data
    pickle.dump([(a, ep, phi, theta, phi_c)  # all parameters
                   for a in a_range
                   for ep in ep_func(a)
                   for phi in psi_to_phi(psi_range, ep)
                   for theta in theta_values
                   for phi_c in phic_values], configfile)

with open(path+filename+r' -data.bin', 'wb') as ofile:
    # store minimum orbit radius a commet reached
    # store initial condition of that commet
    # store probability weight of that ommet
    pass

# run the code
if __name__ == '__main__':
    unsucceed_count = 0  # count for failed calculations
    r_min_data, initial_values, final_values, probabilities = [], [], [], []  # store results
    # start calculation
    results = p_tqdm.p_map(calculate, parameter_space)
    with open(path+filename+r' -data.bin', 'ab') as data_file:
        for succeed, r_min, x_pi, v_pi, x_pf, v_pf, p in results:
            if succeed:
                # store data
                r_min_data.append(r_min)
                initial_values.append((x_pi, v_pi))
                final_values.append((x_pf, v_pf))
                probabilities.append(p)
            else:
                unsucceed_count += 1
        # write results to file
        pickle.dump(r_min_data,      data_file)
        pickle.dump(initial_values,  data_file)
        pickle.dump(final_values,    data_file)
        pickle.dump(probabilities,   data_file)
    print('Calculation completed. Total fialed instances: ', unsucceed_count)
