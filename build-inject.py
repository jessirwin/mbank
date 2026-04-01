from mbank.handlers import variable_handler
from mbank.metric import cbc_metric
from mbank.bank import cbc_bank
from mbank.utils import compute_injections_match, get_random_sky_loc, initialize_inj_stat_dict
from mbank.utils import load_PSD, plot_tiles_templates, ray_compute_injections_match, compute_injections_match, save_inj_stat_dict, plot_tiles_templates, plot_match_histogram
from mbank.flow import STD_GW_Flow
import numpy as np
import corner
import pickle
import time
import torch
# import mbank.utils
import h5py
import matplotlib.pyplot as plt

variable_format = 'Mq_nonspinning_notides'

# bank_path = '/home/jessica.irwin/mbank/build-bank/bank.dat'
bank_path = '/home/jessica.irwin/test-banks/banks_mbank/bns_bank_nospin/bns_bank_nospin/bns_bank_nospin.dat'
# flow_path = '/home/jessica.irwin/mbank/build-bank/flow.zip'
flow_path = '/home/jessica.irwin/test-banks/banks_mbank/bns_bank_nospin/bns_bank_nospin/flow_bns_nospin.zip'

print('open bank in compatible  format')
bank = cbc_bank(variable_format, filename = bank_path)
print('number of templates', len(bank.templates))

print('open flow')
flow = STD_GW_Flow.load_flow(flow_path)

# open the bank
metric = cbc_metric(variable_format,
                    PSD = load_PSD('/home/jessica.irwin/test-banks/banks_mbank/bns_bank/aligo_O3actual_H1.txt', True, 
                    'H1', df = 0.5), approx = 'IMRPhenomD_NRTidal', 
                    f_min = 10, f_max = 1024)

#############
# IF USING existing injections

# open the injection xml file
# this was made when doing injections with an mbank-made template bank via the command line
# injs, skylocs = mbank.utils.read_xml('/home/jessica.irwin/test-banks/banks_mbank/bns_bank_nospin/bns_bank_nospin/bns_nospin_injections.xml', 'sim_inspiral', N=None)

# make injection dictionary and open
# stat_dict = mbank.utils.initialize_inj_stat_dict(injs, skylocs)

# print('injections found')

#############
# # Using EOS-informed injections, pickle
# injection_data_path = '/home/jessica.irwin/eos-data-neural-network/mbank-data/data/massfixed_lambdastiff/1p4mass_lambdastiff_conv.p'
injection_data_path = '/home/jessica.irwin/eos-data-neural-network/mbank-data/data/random_subset_3103.p'

with open(injection_data_path, 'rb') as fp:
    Mqlambdatilde = pickle.load(fp)
    Mqlambdatilde = np.array(Mqlambdatilde, dtype = object)

n_injs = len(Mqlambdatilde.T)

totalM = torch.tensor(np.array(Mqlambdatilde[0], dtype = float))
integer_q = torch.tensor(1/np.array(Mqlambdatilde[1], dtype = float))
lambdatilde = torch.tensor(np.array(Mqlambdatilde[2], dtype = float))

# need to reconstruct the full array of injections that we want to plot
# where q is fraction not integer
# concatenate, total mass fraction q and lambdatilde
M_integerq = torch.cat((totalM, integer_q))
Mq = torch.cat((totalM, torch.tensor(np.array(Mqlambdatilde[1], dtype = float))))

# reshape the array to be N parameters x N injections
M_integerq = torch.reshape(M_integerq, (2,n_injs))
Mq = torch.reshape(Mq, (2,n_injs))

# take transpose
M_integerqT = torch.transpose(M_integerq, 0, 1)
MqT = torch.transpose(Mq, 0, 1)

# make sky locations
n_injs = len(MqT)
print('number of injections', n_injs)
skylocs = np.column_stack(get_random_sky_loc(n_injs))

print('injections found')

##############
# if making injections
injs_3D = torch.tensor(np.array(MqT, dtype = float))
# injs_3D = flow.sample(n_injs)
# np.save('/home/jessica.irwin/test-banks/banks_pycbc/brute_tides/test/plot_injections.npy', injs_3D.cpu().detach().numpy())
# exit(0)

outfile = './build-bank/test_random_%dinjs/' % n_injs

import os
if not os.path.exists(outfile):
    os.makedirs(outfile)

##############
# plot injections
# # val_corner = corner.corner(validation_ll, color = 'indigo', plot_contours = False, plot_DataPoints = True)
# corner.corner(np.array(injs_3D.detach().cpu().numpy()), color = 'red', plot_contours = False, plot_DataPoints = True) # fig = val_corner,
# plt.savefig('./testing/injs_3D.png')
# plt.close()
# exit(0)
##############

# include other parameters
injs_12D = bank.var_handler.get_BBH_components(injs_3D, variable_format)# 
# subset_inj_12D = np.array([injs_12D[:,0], injs_12D[:,1], injs_12D[:,8], injs_12D[:,9]])

# print(np.shape(injs_12D))
# print(injs_12D[0,:])

##############
# plotting mass and tidal deformability components
# corner.corner(subset_inj_12D.T, color = 'red', plot_contours = False, plot_DataPoints = True) # fig = val_corner,
# plt.savefig('./testing/injs_m1m2l1l2.png')
# plt.close()
# exit(0)

# plotting tidal parameters only (components)
# these are of the correct order of magnitude
# plt.scatter(subset_inj_12D[2,:], subset_inj_12D[3,:])
# plt.xscale('log')
# plt.yscale('log')
# plt.savefig('./testing/injs_l1l2.png')
# plt.close()
# exit(0)
##############

# this is not 12D but actually 14D
# m1, m2, s1x, s1y, s1z, s2x, s2y, s2z, l1, l2, e, meanano, iota, phi
sky_locs = np.column_stack(get_random_sky_loc(n_injs))
# makes dictionary
stat_dict = initialize_inj_stat_dict(injs_12D, sky_locs = sky_locs)
print(np.shape(injs_12D))
# stat_dict matches injs_12D for tidal parameters.
# seems fine up to here. 

print('computing injections')
# this takes time!
# compute match between bank and injections
start_time = time.time()
inj_stat_dict = ray_compute_injections_match(stat_dict, bank,
	metric_obj = metric, mchirp_window = 0.1, symphony_match = True, max_jobs = 8, verbose = False)
print('compute match time', time.time() - start_time)

# this is also consistent with input
# print('inj stat dict')

# save injections in dictionary (not sure if needed)
print('saving injections')
save_inj_stat_dict(outfile + 'injections.json', inj_stat_dict)

print('making plots')
# plot the results
# maybe injs here should just be the masses, might be too big an array

# investigating here.
# print('bank.templates')
# print(np.shape(bank.templates))
# corner.corner(bank.templates, color = 'steelblue', plot_contours = False, plot_DataPoints = True)
# plt.savefig('./testing/bank.png')
# plt.close()

print('percentage injections >0.97 match', (np.sum(inj_stat_dict['match']>0.97) / n_injs)*100)

plot_tiles_templates(bank.templates, variable_format,
	injections = M_integerqT.cpu().detach().numpy(), inj_cmap = inj_stat_dict['match'], show = True, save_folder=outfile, savetag = 'new')

# make histogram of matches
print('plotting histogram')
matches = inj_stat_dict['match']
plot_match_histogram(matches = matches, mm = 0.97, bank_name = 'tidalbank \n %.1f percent injections above 0.97 match' % ((np.sum(inj_stat_dict['match']>0.97) / n_injs)*100), save_folder = outfile)

print('injection study done')

exit(0)











