import os
import json
import numpy as np
from scipy.stats import pearsonr
from statsmodels.tsa import stattools
from statsmodels.tsa.statespace.tools import diff

VERBOSE_TEST = False
NUM_LAGS = 15
STATSIG_LEVEL = 0.05
PRICE_FILEPATH = '/Users/aunyks/dev/price-hashrate/btc-price.json'
HASHRATE_FILEPATH = '/Users/aunyks/dev/price-hashrate/btc-hashrate.json'

# Load data from files
with open(PRICE_FILEPATH, 'r') as f:
  price_dict = json.load(f)
with open(HASHRATE_FILEPATH, 'r') as f:
  hashrate_dict = json.load(f)

prices = [point['y'] for point in list(price_dict['values'])]
hashrates = [point['y'] for point in list(hashrate_dict['values'])]

# Are they correlated?
correlation, corr_p = pearsonr(prices, hashrates)

# Visualize data
print('Price\tHashrate')
print('(First 5 in List)')
for i in range(5):
  print(str(prices[i]) + '\t' + str(hashrates[i]))

# Define quick util functions
def is_stationary(X):
  return stattools.adfuller(X)[1] <= 0.05

def diff_til_stationary(X):
  this_data = X
  num_diffs = 0
  while not is_stationary(this_data):
    num_diffs += 1
    this_data = diff(X)
  return (this_data, num_diffs)

# We need stationary data for granger tests
price_diff_results = diff_til_stationary(prices)
hashrate_diff_results = diff_til_stationary(hashrates)
stationary_prices = price_diff_results[0]
stationary_hashrates = hashrate_diff_results[0]
print('Diff\'d prices {0} times'.format(price_diff_results[1]))
print('Diff\'d hashrates {0} times'.format(hashrate_diff_results[1]))

# price_cause_hashrate_result = stattools.grangercausalitytests(np.column_stack((stationary_hashrates, stationary_prices)), NUM_LAGS, verbose=VERBOSE_TEST)
hashrate_cause_price_result = stattools.grangercausalitytests(np.column_stack((stationary_prices, stationary_hashrates)), NUM_LAGS, verbose=VERBOSE_TEST)

earliest_lag = -1
earliest_sig_days_past = -1
earliest_sig_p_value = -1
for lag in list(hashrate_cause_price_result.keys()):
  num_times_diffed = hashrate_diff_results[1]
  days_behind = lag * int(len(prices) / len(stationary_prices))
  """
  print('Days Behind: {0}'.format(days_behind))
  print('-' * 30)
  print('Test\tP-Value')
  """
  test_ps = []
  for test, results in (list(hashrate_cause_price_result.values())[lag - 1][0]).items():
    p_value = results[1]
    test_ps.append(p_value)
    # print('{0}\t{1}'.format(test, p_value))
  average_p = np.mean(test_ps)
  if average_p <= STATSIG_LEVEL and earliest_sig_days_past == -1:
    earliest_sig_days_past = days_behind
    earliest_lag = lag
    earliest_sig_p_value = average_p
  """
  print('average\t{0}'.format(average_p))
  print('-' * 30)
  """
print('Assuming that the two are independent of each other, we can say with {0}% confidence that the Bitcoin price and hashrate are {1} correlated.'.format('%.2f' % ((1 - corr_p) * 100), 'positively' if correlation > 0 else 'negatively'))
print('We can say with {0}% confidence that the Bitcoin price at any time (t) is caused by a hashrate no earlier than {1} days prior to t.'.format('%.2f' % ((1 - earliest_sig_p_value) * 100), earliest_sig_days_past))
