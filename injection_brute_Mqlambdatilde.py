from mbank.handlers import variable_handler
from mbank.metric import cbc_metric
from mbank.bank import cbc_bank
from mbank.utils import compute_injections_match, get_random_sky_loc, initialize_inj_stat_dict, load_inj_stat_dict
from mbank.utils import load_PSD, plot_tiles_templates
from mbank.flow import STD_GW_Flow
import numpy as np
import mbank.utils
import h5py
import corner
import matplotlib.pyplot as plt

stochastic = True
outpath = '/home/jessica.irwin/test-banks/banks_pycbc/brute_tides/fullbank/'
bank_path = '/home/jessica.irwin/test-banks/banks_pycbc/brute_tides/bns_Mqlambdatilde_bank.hdf'

if stochastic:
    print("performing injections with stochastically placed template bank")
    print("changing file format:")
else:
    print("not stochastic bank, exiting")
    exit(0)

# open file using stochastic hdf5 format
d = h5py.File(bank_path, 'r')
M = np.reshape(np.array(d.get('M')), (1,-1))
q = np.reshape(np.array(d.get('q')), (1,-1))
lambdatilde = np.reshape(np.array(d.get('lambdatilde')), (1,-1))
print(d.keys())
d.close()

Mqlambdatilde = np.concatenate((M,q,lambdatilde), axis= 0)
# print(np.shape(Mqlambdatilde))
# exit(0)

# # plot the bank
# corner.corner(Mqlambdatilde.T, plot_contours = False, plot_DataPoints = True, color = 'steelblue')
# plt.savefig(outpath + '/test/testbank.png')
# plt.close()

# save bank as numpy file
np.save(outpath + 'bank.npy', Mqlambdatilde.T)

print('open bank in compatible  format')
bank = cbc_bank('Mq_nonspinning_lambdatilde', filename = outpath + 'bank.npy')

# save as xml file and open as xml file for compatibility (unnecessary?)
#bank._save_xml(outpath + '/test/bank.xml', f_max = 1024, ifo = 'H1')
bank.save_bank(outpath + 'bank.npy', f_max = 1024, ifo = 'H1')
bank = cbc_bank('Mq_nonspinning_lambdatilde', outpath + 'bank.npy')

# open the bank
metric = cbc_metric(bank.variable_format,
                    PSD = load_PSD('/home/jessica.irwin/test-banks/banks_mbank/bns_bank/aligo_O3actual_H1.txt', True, 
                    'H1', df = 0.5), approx = 'IMRPhenomD_NRTidal', 
                    f_min = 10, f_max = 1024)

# open the injection xml file
# this was made when doing injections with an mbank-made template bank via the command line
# injs, skylocs = mbank.utils.read_xml('/home/jessica.irwin/mbank/build-bank/100injs/injections.json', 'sim_inspiral', N=None)
injection_dict = load_inj_stat_dict('/home/jessica.irwin/mbank/build-bank/100injs/injections.json')
injs = injection_dict['theta_inj']
skylocs = injection_dict['sky_loc']

# number of injections
n_injs = len(injs)
print('number of injections', n_injs)

# make injection dictionary and open
stat_dict = mbank.utils.initialize_inj_stat_dict(injs, skylocs)

print('injections found')

print('computing injections')
# compute match between bank and injections
inj_stat_dict = mbank.utils.compute_injections_match(stat_dict, bank,
	metric_obj = metric, mchirp_window = 0.1, symphony_match = True)

# save injections in dictionary (not sure if needed)
print('saving injections')
mbank.utils.save_inj_stat_dict(outpath + 'injections.json', inj_stat_dict)

plot_injs = np.load('/home/jessica.irwin/test-banks/banks_pycbc/brute_tides/test/plot_injections.npy')

print('making plots')
# plot the results
# maybe injs here should just be the masses, might be too big an array
mbank.utils.plot_tiles_templates(bank.templates, bank.variable_format,
	injections = plot_injs, inj_cmap = stat_dict['match'], show = True, save_folder=outpath)

# make histogram of matches
print('plotting histogram')
matches = inj_stat_dict['match']
mbank.utils.plot_match_histogram(matches = matches, mm = 0.97, bank_name = 'Mq_lamdbatilde_test', save_folder = outpath)

print('injection study done')

exit(0)











