from common import *
from sklearn.preprocessing import Imputer as baseImputer
from sklearn.feature_selection import SelectPercentile
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import mean_absolute_error


#%% Metrics

def l1Weighted(dist, weights):
	W = weights.sum() or 1.0
	return (dist * weights).sum() / W

def rmseWeighted(dist, weights):
	W = weights.sum() or 1.0
	return np.sqrt(((dist**2) * weights).sum() / W)

def outliersMetrics(y_true, y_pred, weights):
	""" Separate outliers (error>50%) from regular samples.
	    Return (RMSE, RMSE without outliers, RMSE of outliers, outliers ratio).
		The first metric is the one to be optimized.
	"""
	d = np.abs(y_true - y_pred)
	outliers = (y_pred < 0.5 * y_true) | (y_pred > 2 * y_true)
	return {
		"RMSE": rmseWeighted(d, weights),
		"RMSE without outliers": rmseWeighted(d[~outliers], weights[~outliers]),
		"RMSE of outliers": rmseWeighted(d[outliers], weights[outliers]),
		"Outliers ratio": l1Weighted(outliers, weights),
	}

mainMetricsName = "RMSE"

def testOutliersMetrics():
	y_true = np.arange(0, 16)
	y_pred = y_true + 2
	y_pred[3] = 100
	weights = np.ones_like(y_true) * 0.9 # Should not affect the result
	m = outliersMetrics(y_true, y_pred, weights)
	assert (int(m["RMSE"]), m["RMSE without outliers"], int(m["RMSE of outliers"]), m["Outliers ratio"]) == (
		24, 2.0, 56, 0.1875)
	m = outliersMetrics(y_true[:1], y_pred[:1], weights[:1])
	assert (m["RMSE"], m["RMSE without outliers"], m["RMSE of outliers"], m["Outliers ratio"]) == (
		2.0, 0.0, 2.0, 1.0)
	m = outliersMetrics(y_true, y_pred, weights * 0)
	assert (m["RMSE"], m["RMSE without outliers"], m["RMSE of outliers"], m["Outliers ratio"]) == (
		0.0, 0.0, 0.0, 0.0)
	assert mainMetricsName in m

testOutliersMetrics()


def rankInBatch(batch, weights):
	" Transform each sequence in batch to rank-based values between 0 and 1. "
	ranksBatch = []
	for i in range(len(batch)):
		ok = weights[i] > 0.0
		ranks = rankdata(batch[i, ok])
		ranksBatch.append(ranks / (len(ranks)/2.0) - 1.0)
	return np.concatenate(ranksBatch)

def rankError(y_true, y_pred, weights):
	" Measure the mean error in ranks, per sequence in a batch. "
	r_true = rankInBatch(y_true, weights)
	r_pred = rankInBatch(y_pred, weights)
	mae = mean_absolute_error(r_true, r_pred)
	return mae

def testRankError():
	# Non-linear test data. Only the rank should matter
	y_true = np.linspace(0, 100, 10).reshape(1,-1,1) ** 3
	w = np.ones_like(y_true)
	# Exact match
	assert rankError(y_true, y_true, w) == 0.0
	# Completely wrong
	assert rankError(y_true, y_true[:, ::-1], w) == 1.0
	# Always predict the mean
	y_mean = y_true * 0 + y_true.mean()
	assert rankError(y_true, y_mean, w) == 0.5
	# Exact but swap two adjacents
	yp = y_true.copy()
	yp[:,(3,4)] = y_true[:,(4,3)]
	assert np.isclose(rankError(y_true, yp, w), 0.04)
	# Swap 2 slots away
	yp[:,(6,8)] = y_true[:,(8,6)]
	assert np.isclose(rankError(y_true, yp, w), 0.04 + 0.08)
	# Same result after a permutation
	perm = np.random.choice(range(10), 10, False)
	assert np.isclose(rankError(y_true[:, perm], yp[:, perm], w), 0.04 + 0.08)

testRankError()


def orderMetric(y_true, y_pred, weights):
    """ Return a score of the order similarity of two sequences (rank correlation).
		TODO: support batches.
	"""
    ok = weights > 0.0
    return spearmanr(y_true[ok], y_pred[ok])[0]


#%% Transforms

def logScale(y):
	" Map values from [0; 1e10] to a [-1; 1] log-scale "
	return np.log10(np.clip(y, 1e-10, 1e10)) / 10

def unlogScale(z):
	" The inverse of logScale (for positive original values) "
	return 10.0 ** (z * 10.0)

assert logScale(0) >= -1
assert logScale(1000) <= 1
assert np.allclose(unlogScale(logScale(32)), 32)


class Imputer(baseImputer):
	def inverse_transform(self, X, y=None):
		return X

class MaxImputer1D(object):
	def fit(self, X, y=None):
		self.replacement_ = np.nanmax(X) + 1e-8
		return self

	def transform(self, X, y=None):
		X[np.isnan(X)] = self.replacement_
		return X

	def inverse_transform(self, X, y=None):
		return X

class SelectPercentileSampling(SelectPercentile):
	" Like SelectPercentile but on a sample of the data to limit memory usage. "
	def fit(self, X, y):
		idx = np.random.choice(len(X), 10000)
		X = X[idx]
		y = y[idx]
		return super(SelectPercentileSampling, self).fit(X, y)

def clipExtremes(samples, preds):
	" Keep prediction values within +-1 sigma of training samples. "
	m = samples.mean()
	s = samples.std()
	return preds.clip(m-s, m+s)

def extrapolate(y, seqLen):
	" Extrapolate y to fill up to seqLen "

	gap = seqLen - len(y)
	if gap <= 0:
		return y[:seqLen]
	minY = y.min()
	midY = y.mean()
	slope = (midY - minY) / (len(y) / 2)
	filled = np.zeros((seqLen, y.shape[-1]))
	filled[:len(y)] = y
	filled[len(y):, :] = np.linspace(
		minY + slope * len(y),
		minY + slope * seqLen,
		gap).reshape((-1, 1))
	return filled


class SeqFiller(object):
	""" Fill sequences with the mean observed at each position.
		Learns from a list of sequences with fit_sequences().
		Transform a single sequence with transform().
	"""

	def __init__(self, seqLen, smooth=0):
		self.seqLen = seqLen
		self.smooth = smooth
		self.means_ = None

	def fit_sequences(self, seqs):
		" Learn from a list of sequences"
		shape = (self.seqLen, seqs[0].shape[-1])
		sums = np.zeros(shape)
		counts = np.zeros(shape)

		for seq in seqs:
			l = min(self.seqLen, len(seq))
			seq = seq[:l]
			sums[:l] += np.nan_to_num(seq)	# Add the values, or 0 of nan
			counts[:l] += ~np.isnan(seq)		# Add 1, or 0 if nan

		mean = sums.sum() / counts.sum()
		self.means_ = (sums+mean) / (counts+1)

		if self.smooth:
			kernel = [1./self.smooth] * self.smooth
			self.means_ = np.convolve(self.means_.squeeze(), kernel, "same").reshape(self.means_.shape)

		return self

	def fit(self, *args, **kwargs):
		" Check that fit_sequences() has been called. "
		assert self.means_ is not None, "Call fit_sequences() first."
		return self

	def transform(self, seq, nothing=None):
		" Transform a sequence "
		filled = np.zeros(self.means_.shape)
		l = min(self.seqLen, len(seq))
		filled[:l] = seq[:l]
		filled[l:] = np.nan
		return np.where(np.isnan(filled), self.means_, filled)

	def inverse_transform(self, seq):
		return seq


def testSeqFiller():
	from sklearn.pipeline import make_pipeline
	seqFiller = SeqFiller(seqLen=12)
	seqFiller.fit_sequences([np.random.rand(5,3), np.random.rand(7,3)+10])
	a = np.random.rand(5,3)
	a[1:3] = np.nan
	pipe = make_pipeline(seqFiller)
	pipe.fit([])
	b = pipe.transform(a)
	assert b.shape == (12, 3)
	assert not np.isnan(b).any()

testSeqFiller()


def makePermutations(n):
	""" Randomly swap adjacent items.
		Example usage:
		> p = lh.makePermutations(len(seq))
		> seq = seq.iloc[p]
		> target = target.iloc[p]
		> weights = weights.iloc[p]
	"""
	half = n // 2
	full = half * 2
	swap = np.random.rand(half) > 0.5
	px = np.arange(n)
	px[:full:2] += swap
	px[1:full:2] -= swap
	return px

assert len(makePermutations(6)) == 6
assert len(makePermutations(7)) == 7
assert makePermutations(7)[-1] == 6


def makeCut(test, start=0.8, size=0.1):
	" Return number in [0;1] from distinct train/test regions "
	cut = np.random.rand()
	if test:
		# Only 0.1-0.2 range
		cut = start + size * cut
	else:
		# Skip the 0.1-0.2 range
		cut = cut * (1 - size)
		if cut >= start:
			cut += size
	return cut

# test makeCut
for i in range(20):
	train = makeCut(False)
	assert not (0.8 <= train < 0.9)
	test = makeCut(True)
	assert 0.8 <= test < 0.9

#%%

def sample3Lists(sampler, batchSize):
	" Take `batchSize` items from `sampler` (X,y,w) and return them as 3 lists. "
	seqs = []
	targets = []
	weights = []
	for i in range(batchSize):
		seq, target, weight = sampler()
		seqs.append(seq)
		targets.append(target)
		weights.append(weight)
	return seqs, targets, weights


from keras.preprocessing.sequence import pad_sequences

def padSeqs(seqs, maxLen):
	" Pad a list of sequences "
	if seqs[0] is None:
		return None
	return pad_sequences(
		seqs, maxLen,
		padding="post", truncating="post", dtype="float", value=0.
	)


def batchGenerator(featuresSampler, batchSize, maxLen):
	" Pack several sequences in a batch for Keras: (batchSize, maxLen, features) "
	while True:
		seqs, targets, weights = sample3Lists(featuresSampler, batchSize)
		yield padSeqs(seqs, maxLen), padSeqs(targets, maxLen), padSeqs(weights, maxLen)


def flatGenerator(featuresSampler, batchSize):
	" Concatenate sample sequences, losing the order: (sum of lengths, features) "
	while True:
		seqs, targets, weights = sample3Lists(featuresSampler, batchSize)
		X = np.concatenate(seqs)
		if targets[0] is not None:
			y = np.concatenate(targets)
			# Drop items without targets
			hasTarget = np.concatenate(weights) > 0
			X = X[hasTarget]
			y = y[hasTarget]
			print("%i%% of training issues are closed" % (100 * np.mean(hasTarget)))

			assert len(X) == len(y)
			yield X, y

		else:	# Prediction mode
			yield X, None
