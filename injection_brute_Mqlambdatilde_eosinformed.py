from mbank.handlers import variable_handler
from mbank.metric import cbc_metric
from mbank.bank import cbc_bank
from mbank.utils import compute_injections_match, ray_compute_injections_match, get_random_sky_loc, initialize_inj_stat_dict, load_inj_stat_dict
from mbank.utils import load_PSD, plot_tiles_templates
from mbank.flow import STD_GW_Flow
import numpy as np
import mbank.utils
import h5py
import corner
import time
import pickle
import matplotlib.pyplot as plt
import torch

stochastic = True
eos_index = 0

bank_path = '/home/jessica.irwin/test-banks/banks_pycbc/brute_tides/'
outpath = bank_path + 'test_random_events/' #'eos_fixmass_lambdasoft/'
bank_file_path = bank_path + 'bns_Mqlambdatilde_bank.hdf'

variable_format = 'Mq_nonspinning_lambdatilde'

import os
if not os.path.exists(outpath):
    os.makedirs(outpath)

if stochastic:
    print("performing injections with stochastically placed template bank")
    print("changing file format:")
else:
    print("not stochastic bank, exiting")
    exit(0)


# save bank as numpy file
if not os.path.isfile(outpath + 'bank.npy'):
    
    # open file using stochastic hdf5 format
    d = h5py.File(bank_file_path, 'r')
    M = np.reshape(np.array(d.get('M')), (1,-1))
    q = np.reshape(np.array(d.get('q')), (1,-1))
    lambdatilde = np.reshape(np.array(d.get('lambdatilde')), (1,-1))
    d.close()
    
    Mqlambdatilde = np.concatenate((M,q,lambdatilde), axis= 0)
    np.save(outpath + 'bank.npy', Mqlambdatilde.T)

print('open bank in compatible  format')
# save as xml file and open as xml file for compatibility (unnecessary?)
#bank._save_xml(outpath + '/test/bank.xml', f_max = 1024, ifo = 'H1')
# bank.save_bank(outpath + 'bank.npy', f_max = 1024, ifo = 'H1')
bank = cbc_bank(variable_format, outpath + 'bank.npy')
print('number of templates', len(bank.templates))
print('bank open, continue')

# open the bank
metric = cbc_metric(bank.variable_format,
                    PSD = load_PSD('/home/jessica.irwin/test-banks/banks_mbank/bns_bank/aligo_O3actual_H1.txt', True, 
                    'H1', df = 0.5), approx = 'IMRPhenomD_NRTidal', 
                    f_min = 10, f_max = 1024)

######################
# open EOS informed injections
######################
# # hdf5

# injection_data_path = '/home/jessica.irwin/eos-data-neural-network/mbank-data/data/eos0/inj_tov_eos0_Nevents100.h5'

# events = h5py.File(injection_data_path, 'r')
# # reparameterised M, q and lambda tilde associated to EOS 0
# Mqlambdatilde = torch.tensor(events.get('Mqlambdatilde'))[:,0,:].T
# totalM = torch.tensor(np.array(events.get('Mqlambdatilde'))[0,0,:])
# integer_q = torch.tensor(1/np.array(events.get('Mqlambdatilde'))[1,0,:])
# lambdatilde = torch.tensor(np.array(events.get('Mqlambdatilde'))[2,0,:])
# # need to take 1/q to plot this correctly.
# # change this before next one!
# events.close()

# #############
# # Using EOS-informed injections, pickle
# injection_data_path = '/home/jessica.irwin/eos-data-neural-network/mbank-data/data/massfixed_lambdasoft/1p4mass_lambdasoft_conv.p'
injection_data_path = '/home/jessica.irwin/eos-data-neural-network/mbank-data/data/random_subset_3103.p'

with open(injection_data_path, 'rb') as fp:
    Mqlambdatilde = pickle.load(fp)
    Mqlambdatilde = np.array(Mqlambdatilde, dtype = object)

n_injs = len(Mqlambdatilde.T)

totalM = torch.tensor(np.array(Mqlambdatilde[0], dtype = float))
frac_q = torch.tensor(np.array(Mqlambdatilde[1], dtype = float))
integer_q = torch.tensor(1/np.array(Mqlambdatilde[1], dtype = float))
lambdatilde = torch.tensor(np.array(Mqlambdatilde[2], dtype = float))

# need to reconstruct the full array of injections that we want to plot
# where q is fraction not integer
# concatenate, total mass fraction q and lambdatilde
M_integerq_lambdatilde = torch.cat((totalM, integer_q, lambdatilde))

# reshape the array to be N parameters x N injections
M_integerq_lambdatilde = torch.reshape(M_integerq_lambdatilde, (3,n_injs))

# take transpose
M_integerq_lambdatildeT = torch.transpose(M_integerq_lambdatilde, 0, 1)

# make sky locations
# n_injs = len(Mqlambdatilde)
print('number of injections', n_injs)
skylocs = np.column_stack(get_random_sky_loc(n_injs))

print('injections found')

# # include other parameters
injs_3D = torch.tensor(np.array(Mqlambdatilde.T, dtype = float))
injs_12D = bank.var_handler.get_BBH_components(injs_3D, variable_format)
subset_inj_12D = np.array([injs_12D[:,0], injs_12D[:,1], injs_12D[:,8], injs_12D[:,9]])

# make injection dictionary and open
stat_dict = initialize_inj_stat_dict(injs_12D, skylocs)

# print(stat_dict['match'])

print('computing injections')
# compute match between bank and injections
start_time = time.time()
inj_stat_dict = ray_compute_injections_match(stat_dict, bank,
	metric_obj = metric, mchirp_window = 0.1, symphony_match = True, max_jobs = 8, verbose = False)
print('compute match time', time.time() - start_time)


# print(inj_stat_dict['match'])

# save injections in dictionary (not sure if needed)
print('saving injections')
mbank.utils.save_inj_stat_dict(outpath + 'injections.json', inj_stat_dict)

print('making plots')
# plot the results
# maybe injs here should just be the masses, might be too big an array
mbank.utils.plot_tiles_templates(bank.templates, bank.variable_format,
	injections = M_integerq_lambdatildeT.cpu().detach().numpy(), inj_cmap = inj_stat_dict['match'], show = True, save_folder=outpath)

# make histogram of matches
print('plotting histogram')
matches = inj_stat_dict['match']
mbank.utils.plot_match_histogram(matches = matches, mm = 0.97, bank_name = 'Mq_lamdbatilde_test %d templates' % (n_injs), save_folder = outpath)

print('injection study done')

exit(0)











