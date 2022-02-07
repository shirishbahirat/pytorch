
# calculate the spearmans's correlation between two variables
from numpy.random import randn
from numpy.random import seed
from scipy.stats import spearmanr
from scipy.stats import pearsonr
# seed random number generator
seed(1)
# prepare data
data1 = -20 * randn(1000) + 100
data2 = data1 + (10 * randn(1000) - 50)
# calculate spearman's correlation
corr, _ = spearmanr(data1, data2)
print('Spearmans correlation: %.3f' % corr)

corr, _ = pearsonr(data1, data2)
print('Pearsons correlation: %.3f' % corr)
