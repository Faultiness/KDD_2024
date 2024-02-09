import numpy as np
import torch

from asng import AdaptiveSNG
from sng import SNG
from adam import Adam
from fxc import fxc1
from opt import run
from eval_trace import EvalTrace

def experiment(alg='ASNG', eta_x=0.1, eta_theta_factor=0., alpha=1.5, K=5, D=30, maxite=100000, log_file='log.csv'):
    nc = (K-1) * D
    #f = fxc1(K, D, noise=True)
    categories = K * np.ones(D, dtype=np.int)

    if alg == 'ASNG':
        opt_theta = AdaptiveSNG(categories, alpha=alpha, delta_init=nc**-eta_theta_factor)
    elif alg == 'SNG':
        opt_theta = SNG(categories, delta_init=nc**-eta_theta_factor)
    elif alg == 'Adam':
        opt_theta = Adam(categories, alpha=nc**-eta_theta_factor, beta1=0.9, beta2=0.999)
    else:
        print('invalid algorithm!')
        return

    print('{}, eta_x={}, eta_theta_factor={} alpha={}'.format(alg, eta_x, eta_theta_factor, alpha))
    f = EvalTrace()
    run(f, opt_theta, maxite=maxite, dispspan=1, logspan=1, log_file=log_file)

if __name__ == '__main__':
    experiment(alg='ASNG', eta_x=0.05, eta_theta_factor=0., alpha=1.5, K=3, D=12246, maxite=1000000, log_file='log.csv')
