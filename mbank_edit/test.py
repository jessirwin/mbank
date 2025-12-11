"""
testing
"""

from handlers_fix import variable_handler

print('initialise')
handler = variable_handler()

print('test handler')
handler.is_theta_ok([1,1,50,50], 'm1m2_nonspinning_l1l2')

handler.labels('m1m2_nonspinning_l1l2')

print('exiting')
exit(0)