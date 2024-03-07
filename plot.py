import numpy as np
import pickle

def find_a_ep(x:np.ndarray, v:np.ndarray, GM:float) -> tuple[np.ndarray, np.ndarray]:
    '''Find semi-major axis and eccentricity based on a given pair of position "x" and velocity "v".
    x, v should have same shape [x0, x1, x2, x3,...] which means the last axis defines the target vector.
    return: a, ep
        the semi-major axis "a" and eccentricity "ep"'''
    e = 0.5*np.linalg.norm(v, axis=-1)**2 - GM/np.linalg.norm(x, axis=-1)
    l = np.cross(x, v, axisa=-1, axisb=-1)  # angular momentum
    if x.shape[-1] == 3:
        # 2d cross priduct is already a scalar; np.cross will raise error itself if dim is neither 2 or 3
        l = np.linalg.norm(l, axis=-1)
    a = -0.5*GM/e
    ep = 1 + 2*e*l**2/GM**2
    ep[ep<0] = 0.  # may be negative due to numerical error
    ep = np.sqrt(ep)
    return a, ep



fname = r"D:\OneDrive - University of Florida\Programs\Caustic\test\No_Osc, 1e13m, 1e13s, v=1.00, A=0.002,"

print('Reading data...please wait')
print('file name:', fname)

with open(fname + r" -config.bin", 'rb') as file:
    info = pickle.load(file)
    filename, omega, v_c, g_c, GM = pickle.load(file)
    data_shape = pickle.load(file)
    parameters = np.array(pickle.load(file))

with open(fname + r" -data.bin", 'rb') as file:
    min_r = np.array(pickle.load(file))
    initial_cond = np.array(pickle.load(file))
    final_cond = np.array(pickle.load(file))
    probability = np.array(pickle.load(file))

initial_x = initial_cond[:,0]
initial_v = initial_cond[:,1]
final_x = final_cond[:,0]
final_v = final_cond[:,1]

print('Data reading succeed. Parameter shape:', data_shape)

# axes = (a, ep, phi, theta)
parameters = parameters.reshape(*data_shape, 5)
parameters = parameters.reshape(*data_shape, 5)

# normalized probability
# <a> = \int_0^2\pi a(\phi) p(\phi) dphi -> \sum_i a(\phi_i) p(\phi_i) {Delta phi}_i
# the following calculates p(\phi_i) {Delta phi}_i
phis = parameters[:,:,:,:,:,2]
phis_minus = np.roll(phis, 1, axis=2)  # phis[N-1,0,1,...,N-2]
phis_plus = np.roll(phis, -1, axis=2)  # phis[1,2,...,N-1,0]
delta_phis = (phis_plus - phis_minus) / 2
delta_phis = np.where(delta_phis<0, delta_phis+np.pi, delta_phis)  # phi is periodic
probability = probability.reshape(*data_shape) * delta_phis
probability /= np.sum(probability, axis=2, keepdims=True)  # Although probability should have been normalized, it may behave bad at large ep values. Normalize it again.

# final state data
a_f, ep_f = find_a_ep(final_x, final_v, GM)

# minimum radius a comet can reach
r_min_aep = np.where(ep_f<=1, a_f*(1-ep_f), np.inf)  # r_min for final state stable orbit (eqn only valaid for bounded orbit)
min_r = np.minimum(r_min_aep, min_r)  # Every comet has a smallest integration r_min. Use this r_min for escape state orbit (because the previous one is invalid, gives negative r_min)
# r_min averaged over phi
result_rmin_ave = (min_r.reshape(*data_shape))*probability
result_rmin_ave = np.sum(result_rmin_ave, axis=2)  # result
# smallest r_min for all phi
result_rmin = np.min(min_r.reshape(*data_shape), axis=2)  # result

# find all excape comets
escape = np.where(ep_f>=1, 1, 0)
# if there's escaped comet among all phi
result_escape = np.max(escape.reshape(*data_shape), axis=2)  # result
# probability for escape
result_escape_ave = (escape.reshape(*data_shape))*probability
result_escape_ave = np.sum(result_escape_ave, axis=2)  # result

# fall in 50 AU
result_fallin = np.where(min_r<=0.075*10, 1., 0.)  # 0.075=5AU
result_fallin_ave = ((result_fallin).reshape(*data_shape))*probability  # result
# minimum r among all phi
result_fallin_ave = np.sum(result_fallin_ave, axis=2)
result_fallin = np.max(result_fallin.reshape(*data_shape), axis=2)  # result

with open(fname + r" -plot.bin", 'wb') as plotfile:
    pickle.dump(parameters, plotfile)
    pickle.dump(result_rmin, plotfile)
    pickle.dump(result_rmin_ave, plotfile)
    pickle.dump(result_escape, plotfile)
    pickle.dump(result_escape_ave, plotfile)
    pickle.dump(result_fallin, plotfile)
    pickle.dump(result_fallin_ave, plotfile)


print('done')
