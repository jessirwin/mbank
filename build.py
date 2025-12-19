"""
Following https://mbank.readthedocs.io/en/latest/usage/bank_generation.html#bank-by-hands
to build a bank from scratch (not command line) with tides.
"""

# import mbank.metric as metric
from mbank.metric import cbc_metric
from mbank.bank import cbc_bank
from mbank.utils import load_PSD, plot_tiles_templates, get_boundaries_from_ranges
from mbank.flow import STD_GW_Flow
from mbank.flow.utils import early_stopper, plot_loss_functions
import numpy as np
import corner
import matplotlib.pyplot as plt

# format of the variables we have
variable_format = 'Mq_nonspinning_lambdatilde'
dims = 3 # dimensions of the parameter space, m1m2l1l2

# need to reformat this to allow for different parameterisations of tides
boundaries = get_boundaries_from_ranges(variable_format, (1,6), (0.333,1))

print('variables and boundaries set')

######################
# do the metric part
######################

# need to allow for a different approximant which includes tides
metric_ = cbc_metric(variable_format,
			PSD = load_PSD('/home/jessica.irwin/test-banks/banks_mbank/bns_bank/aligo_O3actual_H1.txt', 
            True, 'H1'), approx = 'IMRPhenomD_NRTidal', f_min = 10, f_max = 1024)

print('metric established')
# this works.

######################
# flow training 
######################

# this is just the progress bar
from tqdm import tqdm
from torch import optim

# *boundaries with the asterix separates the two rows or two parts of the array
# so separates the upper bounds and lower bounds from boundaries into two 
# separate arr

train_data = np.random.uniform(*boundaries, (10000, dims))
validation_data = np.random.uniform(*boundaries, (300, dims))

###################
# plot training and validation data

# val_corner = corner.corner(validation_data, color = 'indigo', plot_contours = False, plot_DataPoints = True)
# corner.corner(train_data, fig = val_corner, color = 'steelblue', plot_contours = False, plot_DataPoints = True)
# plt.savefig('./testing/train_val_data.png')
# plt.close()

###################
# from mbank.metric import test_metric
# test_metric(train_data, boundaries)
# # testing different parts of the function by commenting out other bits
# exit(0)

# make training, validation data
train_ll = np.array([metric_.log_pdf(s, boundaries) for s in tqdm(train_data)])
validation_ll = np.array([metric_.log_pdf(s, boundaries) for s in tqdm(validation_data)])

print('pre establish flow')

flow = STD_GW_Flow(3, n_layers = 2, hidden_features = 30)

print('pre flow set up')

early_stopper_callback = early_stopper(patience=20, min_delta=1e-3)
optimizer = optim.Adam(flow.parameters(), lr=5e-3)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', threshold = .02, factor = 0.5, patience = 4)

print('pre train flow')
	
history = flow.train_flow('ll_mse', N_epochs = 10000,
	train_data = train_data, train_weights = train_ll,
	validation_data = validation_data, validation_weights = validation_ll,
	optimizer = optimizer, batch_size = 500, validation_step = 100,
	callback = early_stopper_callback, lr_scheduler = scheduler,
	boundaries = boundaries, verbose = True)

residuals = np.squeeze(validation_ll) - flow.log_volume_element(validation_data)

# error in metric computation, eigenvalue is negative. 

print('flow training successful')
# exit(0)

######################
# bank generation
######################

from mbank.placement import place_random_flow

new_templates = place_random_flow(0.97, flow, metric_,
	n_livepoints = 500, covering_fraction = 0.9,
	boundaries_checker = boundaries,
	metric_type = 'symphony', verbose = True)
bank = cbc_bank(variable_format)
bank.add_templates(new_templates)

print('building done; now saving')

flow.save_weigths('./build-bank/flow.zip')
bank.save_bank('./build-bank/bank.dat')

# print('plotting bank')
# plt.figure()
# plt.scatter(bank.M, bank.q, s = 5)
# plt.xlabel('M')
# plt.ylabel('q')
# plt.savefig('./build-bank/bank-Mq.png')
# plt.close()

# plt.figure()
# plt.scatter(bank.M, bank.lambdatilde, s = 5)
# plt.xlabel('M')
# plt.ylabel('lambdatilde')
# plt.savefig('./build-bank/bank-Mlambdatilde.png')
# plt.close()

# plt.figure()
# plt.scatter(bank.q, bank.lambdatilde, s = 5)
# plt.xlabel('q')
# plt.ylabel('lambdatilde')
# plt.savefig('./build-bank/bank-qlambdatilde.png')
# plt.close()

print('plotting loss')
plot_loss_functions(history, './build-bank/')

plt.figure()
plt.hist(residuals/np.log(10),
	histtype = 'step', bins = 30, density = True)
plt.xlabel(r"$\log_{10}(M_{flow}/M_{true})$")
plt.savefig('./build-bank/accuracyhist.png')
plt.close()

exit(0)





