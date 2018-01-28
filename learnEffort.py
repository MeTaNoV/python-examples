"""
	Train a model to predict the effort left required for each issue.
"""

#%%
from time import time
import numpy as np
import pandas as pd
import sklearnAll as sk
from sklearn.feature_selection import SelectPercentile, f_regression, mutual_info_regression, mutual_info_classif
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import accuracy_score

from dataframes import keepLast
import learnHelpers as lh
from timeUtils import toIso

TRAINING_WEEKS = 52	# How many weeks of past data to train on
MAX_DEVIATION = 1.5	# ~85% of values allowed


def splitClosedOpen(eventReprs):
	" Return events of closed issues from when they were still open (learn), and events of issues that are still open (predict). "

	# Recent events only
	now = eventReprs[("event", "dt")].max()
	recent = now - pd.Timedelta(weeks=TRAINING_WEEKS)

	# Find recent end of progress. Implies that there was progress and it's closed now.
	recentlyEnded = eventReprs[("targets", "endDate")] > recent
	# Optional: ignore events after closing.
	# notClosedYet = eventReprs[("event", "dt")] < eventReprs[("targets", "closeDate")]
	# Optional: require some minimum meaningful effort
	# eventReprs[("targets", "effortSpent")] > 0
	endedEvents = eventReprs[ recentlyEnded ]

	# Open issues to predict, only the last event
	neverClosed = eventReprs[("targets", "closeDate")].isnull()
	openEvents = eventReprs[neverClosed]
	openIssues = keepLast(openEvents, ("event","key"))
	return endedEvents, openIssues


def learnEffort(eventReprs):
	scriptStartTime = time()

	# Separate past events to learn from and current open issues to predict
	endedEvents, openIssues = splitClosedOpen(eventReprs)

	pe_train, pe_test = sk.train_test_split(endedEvents, test_size=0.1, random_state=12)

	# Format training data

	def makeFactors(evs):
		# We only take these column groups into account.
		# They are calculated in eventFeatures.py.
		f = evs[[
			"project_emb",
			"issuetype_emb",
			"status_emb",
			"priority_emb",
			"author_emb",
			"assignee_emb",
			"action_emb",
			"author_hist",
			"action_hist",
			#"mainValue_hist",
			#"work_hist",
			"text_hist",
			"timing",
		]]
		#f[("CHEAT","a")] = evs[[("targets", "effortLeft")]]
		return f.values

	xPipe = sk.make_pipeline(
		sk.FunctionTransformer(makeFactors, validate=False),
		sk.StandardScaler(),
		sk.FunctionTransformer(lambda x: x.clip(-5,5)), # Limit outlier effects
		# Remove non-informative features
		#lh.SelectPercentileSampling(mutual_info_classif, percentile=50),
	)

	yPipe = sk.make_pipeline(
		sk.FunctionTransformer(lambda evs: evs[[("targets", "effortLeft")]].values, validate=False),
		sk.StandardScaler(),
	)

	y_train = yPipe.fit_transform(pe_train).squeeze()
	y_test = yPipe.transform(pe_test).squeeze()

	X_train = xPipe.fit_transform(pe_train, y_train)
	X_test = xPipe.transform(pe_test)
	X_future = xPipe.transform(openIssues)
	keys_future = openIssues[("event","key")]

	#%% Train

	def train():
		#model = sk.MLPRegressor((128,128), verbose=False)
		#from xgboost import XGBRegressor; model = XGBRegressor()
		model = RandomForestRegressor()
		model.fit(X_train, y_train)
		model.y_min_, model.y_max_ = np.percentile(y_train, [0, 100])
		return model

	def evaluate(model):
		metrics = {}
		for (suffix, X_, y_) in [
			("_train", X_train, y_train),
			("", X_test, y_test),
		]:
			y_pred = model.predict(X_).clip(model.y_min_, model.y_max_)
			rmse = np.sqrt(sk.mean_squared_error(y_, y_pred))
			metrics[lh.mainMetricsName + suffix] = rmse
			#plt.scatter(yPipe.inverse_transform(y_[::10]), yPipe.inverse_transform(y_pred[::10]), s=1)
		return metrics

	# ==== Learn
	model = train()
	metrics = evaluate(model)

	#%% Predict
	print("Predicting on open issues…")
	y_pred = model.predict(X_future).clip(model.y_min_, model.y_max_)
	effortLeft = yPipe.inverse_transform(y_pred.reshape(len(X_future), 1))
	effortLeft = pd.DataFrame(effortLeft, index=keys_future, columns=["effortLeft"])
	print("Predicted", effortLeft.shape, "issues. Effort left:", effortLeft.min()[0], "-", effortLeft.mean()[0], "-", effortLeft.max()[0], "hours.")

	# Measure
	print("Measuring on all issues…")
	# TODO: not exactly the same because some predictions are in the middle of progress.
	measures = eventReprs[[("targets", "effortSpent")]].groupby("key_i").last()
	measures.columns = ["effortSpent"]
	print("Measured", measures.shape, "issues. Effort per issue:", measures.min()[0], "-", measures.mean()[0], "-", measures.max()[0], "hours.")

	efforts = pd.concat([effortLeft, measures], axis="columns").fillna(0.0)

	# Metadata
	meta = {
		"Compute_minutes": (time() - scriptStartTime) / 60,

		"Measure_count": len(measures),
		"Measure_effort_hours_min": float(measures.min()),
		"Measure_effort_hours_avg": float(measures.mean()),
		"Measure_effort_hours_max": float(measures.max()),

		"Training_event_count": len(endedEvents),
		"Training_earliest_event": toIso(endedEvents[("event", "dt")].min()),
		"Training_latest_event": toIso(endedEvents[("event", "dt")].max()),

		"Prediction_count": len(effortLeft),
		"Prediction_effort_hours_min": float(effortLeft.min()),
		"Prediction_effort_hours_avg": float(effortLeft.mean()),
		"Prediction_effort_hours_max": float(effortLeft.max()),
	}
	meta.update(metrics)

	return efforts, meta
