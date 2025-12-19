from mbank.handlers import variable_handler
from mbank.metric import cbc_metric
from mbank.bank import cbc_bank
from mbank.utils import compute_injections_match, get_random_sky_loc, initialize_inj_stat_dict
from mbank.utils import load_PSD, plot_tiles_templates, compute_injections_match, save_inj_stat_dict, plot_tiles_templates, plot_match_histogram
from mbank.flow import STD_GW_Flow
import numpy as np
# import mbank.utils
import h5py

variable_format = 'Mq_nonspinning_lambdatilde'

bank_path = '/home/jessica.irwin/mbank/build-bank/bank.dat'
flow_path = '/home/jessica.irwin/mbank/build-bank/flow.zip'

print('open bank in compatible  format')
bank = cbc_bank(variable_format, filename = bank_path)

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

##############
# if making injections
n_injs = 100
injs_3D = flow.sample(n_injs)
injs_12D = bank.var_handler.get_BBH_components(injs_3D, variable_format)
sky_locs = np.column_stack(get_random_sky_loc(n_injs))
stat_dict = initialize_inj_stat_dict(injs_12D, sky_locs = sky_locs)

print('computing injections')
# compute match between bank and injections
inj_stat_dict = compute_injections_match(stat_dict, bank,
	metric_obj = metric, mchirp_window = 0.1, symphony_match = True)

# save injections in dictionary (not sure if needed)
print('saving injections')
save_inj_stat_dict('/build-bank/injections.json', inj_stat_dict)


print('making plots')
# plot the results
# maybe injs here should just be the masses, might be too big an array
plot_tiles_templates(bank.templates, variable_format,
	injections = injs, inj_cmap = stat_dict['match'], show = True, save_folder='./build-bank/')

# make histogram of matches
print('plotting histogram')
matches = inj_stat_dict['match']
plot_match_histogram(matches = matches, mm = 0.97, bank_name = 'tidalbank', save_folder = './build-bank/')

print('injection study done')

exit(0)











