import numpy as np
import torch

def gen_gass_pos_weights(pos, miu=0, sig=10, normalize=False):
    pos = np.array(pos)
    sqrt_2pi = np.power(2 * np.pi, 0.5)
    conf = 1 / (sqrt_2pi * sig)
    power_conf = -1 / (2 * np.power(sig, 2))
    data_power = power_conf * (np.power((pos - miu), 2))
    gass_pos_weights = conf * np.exp(data_power)
    if normalize:
        gass_pos_weights = torch.tensor(gass_pos_weights)
        gass_pos_weights = torch.nn.functional.softmax(gass_pos_weights)
    else:
        gass_pos_weights = torch.tensor(gass_pos_weights)
    return gass_pos_weights