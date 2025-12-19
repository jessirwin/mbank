# import mbank.metric as metric
from mbank.metric import cbc_metric
from mbank.bank import cbc_bank
from mbank.utils import load_PSD, plot_tiles_templates, get_boundaries_from_ranges
from mbank.flow import STD_GW_Flow
from mbank.flow.utils import early_stopper, plot_loss_functions
import numpy as np
import corner
import matplotlib.pyplot as plt

print('open bank')
variable_format = 'Mq_nonspinning_lambdatilde' 
bank = cbc_bank(variable_format, filename = '/home/jessica.irwin/mbank/build-bank/bank.dat')

print(bank.keys())
exit(0)

print('open flow')
flow = STD_GW_Flow.load_flow('/home/jessica.irwin/mbank/build-bank/flow.zip')

print('plotting bank')
plt.figure()
plt.scatter(bank.M, bank.q, s = 5)
plt.xlabel('M')
plt.ylabel('q')
plt.savefig('./build-bank/bank-Mq.png')
plt.close()

plt.figure()
plt.scatter(bank.M, bank.lambdatilde, s = 5)
plt.xlabel('M')
plt.ylabel('lambdatilde')
plt.savefig('./build-bank/bank-Mlambdatilde.png')
plt.close()

# plt.figure()
# plt.scatter(bank.q, bank.lambdatilde, s = 5)
# plt.xlabel('q')
# plt.ylabel('lambdatilde')
# plt.savefig('./build-bank/bank-qlambdatilde.png')
# plt.close()

print('plotting loss')
plot_loss_functions(history, './build-bank/loss.png')

plt.figure()
plt.hist(residuals/np.log(10),
	histtype = 'step', bins = 30, density = True)
plt.xlabel(r"$\log_{10}(M_{flow}/M_{true})$")
plt.savefig('./build-bank/accuracyhist.png')
plt.close()

exit(0)