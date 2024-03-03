import evaluation_utils as eval_utils
import matplotlib.pyplot as plt
import numpy as np
import range_query as rq
import json
import sklearn as sk
import statistics1 as stats
import os

eps = 1e-8

def min_max_normalize(v, min_v, max_v):
	# The function may be useful when dealing with lower/upper bounds of columns.
	assert max_v > min_v
	return (v-min_v)/(max_v-min_v)


def extract_features_from_query(range_query, considered_cols, table_stats):
	# feat:	 [c1_begin, c1_end, c2_begin, c2_end, ... cn_begin, cn_end, AVI_sel, EBO_sel, Min_sel]
	#		   <-				   range features					->, <-	 est features	 ->
	feature = []
	n = 0
	rc = range_query
	for col in considered_cols:
		min_val = table_stats.columns[col].min_val()
		max_val = table_stats.columns[col].max_val()
		(left, right) = range_query.column_range(col, min_val, max_val)
		feature.append(left)
		feature.append(right)
	feature.append(stats.AVIEstimator.estimate(rc, table_stats))
	feature.append(stats.ExpBackoffEstimator.estimate(rc, table_stats))
	feature.append(stats.MinSelEstimator.estimate(rc, table_stats))

	return feature

def extract_features_from_query_without_extra_features(range_query, considered_cols, table_stats):
	# feat:	 [c1_begin, c1_end, c2_begin, c2_end, ... cn_begin, cn_end, AVI_sel, EBO_sel, Min_sel]
	#		   <-				   range features					->, <-	 est features	 ->
	feature = []
	n = 0
	rc = range_query
	for col in considered_cols:
		min_val = table_stats.columns[col].min_val()
		max_val = table_stats.columns[col].max_val()
		(left, right) = range_query.column_range(col, min_val, max_val)
		feature.append(left)
		feature.append(right)

	return feature


def preprocess_queries(queris, table_stats, columns):
	features, labels = [], []
	for item in queris:
		query, act_rows = item['query'], item['act_rows']
		feature, label = None, None
		label = np.log2(act_rows)
		feature = extract_features_from_query(rq.ParsedRangeQuery.parse_range_query(query), columns, table_stats)
		tran_feature = np.array(feature)
		features.append(tran_feature)
		labels.append(label)
	tran_features = np.array(features)
	tran_labels = np.array(labels)
	return tran_features, tran_labels
	
def preprocess_queries_without_extra_features(queris, table_stats, columns):
	features, labels = [], []
	for item in queris:
		query, act_rows = item['query'], item['act_rows']
		feature, label = None, None
		label = np.log2(act_rows)
		feature = extract_features_from_query_without_extra_features(rq.ParsedRangeQuery.parse_range_query(query), columns, table_stats)
		tran_feature = np.array(feature)
		features.append(tran_feature)
		labels.append(label)
	tran_features = np.array(features)
	tran_labels = np.array(labels)
	return tran_features, tran_labels

from sklearn.neural_network import MLPRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression

def f(x): # 取2的x次方
	if x < 1:
		return 1
	if x > 30:
		return int(1e8)
	return int(2 ** x + eps)

# --------------------------------------------------------------------------------
# ------------↓↓↓↓↓↓↓↓↓↓↓↓ 线性回归模型（无额外特征对比组） ↓↓↓↓↓↓↓↓↓↓↓↓-------------

# WEF : Without Extra Features
def est_LRe_normal_WEF(train_data, test_data, table_stats, columns):
	train_feature, train_lables = preprocess_queries_without_extra_features(train_data, table_stats, columns)
	train_est_rows, train_act_rows = [], train_lables
	model = LinearRegression()

	model.fit(train_feature, train_lables)
	train_est_rows = model.predict(train_feature)

	test_feature, test_lables = preprocess_queries_without_extra_features(test_data, table_stats, columns)
	test_est_rows, test_act_rows = [], test_lables
	test_est_rows = model.predict(test_feature)
	
	train_est_rows = list(map(f, train_est_rows))
	train_act_rows = list(map(f, train_act_rows))
	test_est_rows = list(map(f, test_est_rows))
	test_act_rows = list(map(f, test_act_rows))
	return train_est_rows, train_act_rows, test_est_rows, test_act_rows

# ---------------------------------------------------------------------------------
# -----------↓↓↓↓↓↓↓↓↓↓↓↓ MLP神经网络模型（无额外特征对比组） ↓↓↓↓↓↓↓↓↓↓↓↓------------

def est_MLP_normal_WEF(train_data, test_data, table_stats, columns):
	train_feature, train_lables = preprocess_queries_without_extra_features(train_data, table_stats, columns)
	train_est_rows, train_act_rows = [], train_lables
	model = MLPRegressor(hidden_layer_sizes=(100, 100, 100, 100)
	, alpha = 0.01, learning_rate_init=0.0001, tol = 0.0001, activation='relu', solver='adam', max_iter=10000)

	model.fit(train_feature, train_lables)
	train_est_rows = model.predict(train_feature)

	test_feature, test_lables = preprocess_queries_without_extra_features(test_data, table_stats, columns)
	test_est_rows, test_act_rows = [], test_lables
	test_est_rows = model.predict(test_feature)
	
	train_est_rows = list(map(f, train_est_rows))
	train_act_rows = list(map(f, train_act_rows))
	test_est_rows = list(map(f, test_est_rows))
	test_act_rows = list(map(f, test_act_rows))
	return train_est_rows, train_act_rows, test_est_rows, test_act_rows



# -----------------------------------------------------------------------------
# ---------------↓↓↓↓↓↓↓↓↓↓↓↓ 线性回归模型（无优化） ↓↓↓↓↓↓↓↓↓↓↓↓----------------

def est_LRe_normal(train_data, test_data, table_stats, columns):
	train_feature, train_lables = preprocess_queries(train_data, table_stats, columns)
	train_est_rows, train_act_rows = [], train_lables
	model = LinearRegression()

	model.fit(train_feature, train_lables)
	train_est_rows = model.predict(train_feature)

	test_feature, test_lables = preprocess_queries(test_data, table_stats, columns)
	test_est_rows, test_act_rows = [], test_lables
	test_est_rows = model.predict(test_feature)
	
	train_est_rows = list(map(f, train_est_rows))
	train_act_rows = list(map(f, train_act_rows))
	test_est_rows = list(map(f, test_est_rows))
	test_act_rows = list(map(f, test_act_rows))
	return train_est_rows, train_act_rows, test_est_rows, test_act_rows
	
# -----------------------------------------------------------------------------
# ------------------↓↓↓↓↓↓↓↓↓↓↓↓ 线性回归模型+PCA ↓↓↓↓↓↓↓↓↓↓↓↓------------------

from sklearn.decomposition import PCA

pca = 0
def init_PCA(number_of_dimension):
	global pca
	pca = PCA(n_components = number_of_dimension)

def compress_features_PCA(features):
	global pca
	compressed_features = pca.fit_transform(features)
	return compressed_features

def est_LRe_PCA(train_data, test_data, table_stats, columns):
	init_PCA(1)
	train_feature, train_lables = preprocess_queries(train_data, table_stats, columns)
	
	new_train_feature = compress_features_PCA(train_feature)
	train_est_rows, train_act_rows = [], train_lables
	
	model = LinearRegression()
	model.fit(new_train_feature, train_lables)

	train_est_rows = model.predict(new_train_feature)

	test_feature, test_lables = preprocess_queries(test_data, table_stats, columns)
	new_test_feature = compress_features_PCA(test_feature)
	test_est_rows, test_act_rows = [], test_lables
	
	test_est_rows = model.predict(new_test_feature)

	train_est_rows = list(map(f, train_est_rows))
	train_act_rows = list(map(f, train_act_rows))
	test_est_rows = list(map(f, test_est_rows))
	test_act_rows = list(map(f, test_act_rows))
	return train_est_rows, train_act_rows, test_est_rows, test_act_rows
	
# -----------------------------------------------------------------------------
# --------------↓↓↓↓↓↓↓↓↓↓↓↓ 线性回归模型+数据标准化 ↓↓↓↓↓↓↓↓↓↓↓↓----------------

from sklearn.preprocessing import StandardScaler

def est_LRe_StdSca(train_data, test_data, table_stats, columns):
	train_feature, train_lables = preprocess_queries(train_data, table_stats, columns)
	scaler = StandardScaler()
	scaler.fit(train_feature)
	new_train_feature = scaler.transform(train_feature)
	
	model = LinearRegression()
	model.fit(new_train_feature , train_lables)
	train_est_rows, train_act_rows = [], train_lables
	train_est_rows = model.predict(new_train_feature)
	
	test_feature, test_lables = preprocess_queries(test_data, table_stats, columns)
	new_test_feature = scaler.transform(test_feature)
	
	test_est_rows, test_act_rows = [], test_lables
	test_est_rows = model.predict(new_test_feature)
	
	train_est_rows = list(map(f, train_est_rows))
	train_act_rows = list(map(f, train_act_rows))
	test_est_rows = list(map(f, test_est_rows))
	test_act_rows = list(map(f, test_act_rows))
	return train_est_rows, train_act_rows, test_est_rows, test_act_rows




# ------------------------------------------------------------------------------
# --------------↓↓↓↓↓↓↓↓↓↓↓↓ MLP神经网络模型（无优化） ↓↓↓↓↓↓↓↓↓↓↓↓---------------

def est_MLP_normal(train_data, test_data, table_stats, columns):
	train_feature, train_lables = preprocess_queries(train_data, table_stats, columns)
	train_est_rows, train_act_rows = [], train_lables
	model = MLPRegressor(hidden_layer_sizes=(100, 100, 100, 100)
	, alpha = 0.01, learning_rate_init=0.0001, tol = 0.0001, activation='relu', solver='adam', max_iter=10000)

	model.fit(train_feature, train_lables)
	train_est_rows = model.predict(train_feature)

	test_feature, test_lables = preprocess_queries(test_data, table_stats, columns)
	test_est_rows, test_act_rows = [], test_lables
	test_est_rows = model.predict(test_feature)

	train_est_rows = list(map(f, train_est_rows))
	train_act_rows = list(map(f, train_act_rows))
	test_est_rows = list(map(f, test_est_rows))
	test_act_rows = list(map(f, test_act_rows))
	return train_est_rows, train_act_rows, test_est_rows, test_act_rows


# -----------------------------------------------------------------------------
# -------------↓↓↓↓↓↓↓↓↓↓↓↓ MLP+PCA特征提取预处理数据 ↓↓↓↓↓↓↓↓↓↓↓↓---------------

def est_MLP_PCA(train_data, test_data, table_stats, columns):
	init_PCA(1)
	train_feature, train_lables = preprocess_queries(train_data, table_stats, columns)
	
	new_train_feature = compress_features_PCA(train_feature)
	train_est_rows, train_act_rows = [], train_lables
	
	model = MLPRegressor(hidden_layer_sizes=(100, 100, 100, 100)\
	, alpha = 0.001, learning_rate_init=0.00001, tol = 0.001, activation='relu', solver='adam', max_iter=10000)
	model.fit(new_train_feature, train_lables)

	train_est_rows = model.predict(new_train_feature)

	test_feature, test_lables = preprocess_queries(test_data, table_stats, columns)
	new_test_feature = compress_features_PCA(test_feature)
	test_est_rows, test_act_rows = [], test_lables
	
	test_est_rows = model.predict(new_test_feature)

	train_est_rows = list(map(f, train_est_rows))
	train_act_rows = list(map(f, train_act_rows))
	test_est_rows = list(map(f, test_est_rows))
	test_act_rows = list(map(f, test_act_rows))
	return train_est_rows, train_act_rows, test_est_rows, test_act_rows

# ------------------------------------------------------------------------------
# ----------------↓↓↓↓↓↓↓↓↓↓↓↓ MLP+标准化预处理数据 ↓↓↓↓↓↓↓↓↓↓↓↓------------------

def est_MLP_StdSca(train_data, test_data, table_stats, columns):
	train_feature, train_lables = preprocess_queries(train_data, table_stats, columns)
	scaler = StandardScaler()
	scaler.fit(train_feature)
	new_train_feature = scaler.transform(train_feature)
	model = MLPRegressor(hidden_layer_sizes=(100,100,100, 10), alpha = 0.01, learning_rate_init=0.01, tol = 0.01, activation='relu', solver='adam', max_iter=10000)
	
	train_est_rows, train_act_rows = [], train_lables
	model.fit(new_train_feature , train_lables)
	train_est_rows = model.predict(new_train_feature)
	
	test_feature, test_lables = preprocess_queries(test_data, table_stats, columns)
	new_test_feature = scaler.transform(test_feature)
	test_est_rows, test_act_rows = [], test_lables
	test_est_rows = model.predict(new_test_feature)
	
	train_est_rows = list(map(f, train_est_rows))
	train_act_rows = list(map(f, train_act_rows))
	test_est_rows = list(map(f, test_est_rows))
	test_act_rows = list(map(f, test_act_rows))
	return train_est_rows, train_act_rows, test_est_rows, test_act_rows

# -------------------------------------------------------------------------------
# ---------------↓↓↓↓↓↓↓↓↓↓↓↓ MLP+标准化+去掉额外特征 ↓↓↓↓↓↓↓↓↓↓↓↓-----------------

def est_MLP_StdSca_WEF(train_data, test_data, table_stats, columns):
	train_feature, train_lables = preprocess_queries_without_extra_features(train_data, table_stats, columns)
	scaler = StandardScaler()
	scaler.fit(train_feature)
	new_train_feature = scaler.transform(train_feature)
	model = MLPRegressor(hidden_layer_sizes=(100,100,100, 10), alpha = 0.01, learning_rate_init=0.01, 
					  tol = 0.01, activation='relu', solver='adam', max_iter=10000)
	
	train_est_rows, train_act_rows = [], train_lables
	model.fit(new_train_feature , train_lables)
	train_est_rows = model.predict(new_train_feature)
	
	test_feature, test_lables = preprocess_queries_without_extra_features(test_data, table_stats, columns)
	new_test_feature = scaler.transform(test_feature)
	test_est_rows, test_act_rows = [], test_lables
	test_est_rows = model.predict(new_test_feature)
	
	train_est_rows = list(map(f, train_est_rows))
	train_act_rows = list(map(f, train_act_rows))
	test_est_rows = list(map(f, test_est_rows))
	test_act_rows = list(map(f, test_act_rows))
	return train_est_rows, train_act_rows, test_est_rows, test_act_rows

	
def get_variable_name(var): # 获取var变量的名字，并转化为str类型返回
    for name, value in globals().items():
        if value is var:
            return name
    return None

def eval_model(est_fn, train_data, test_data, table_stats, columns):
	model = get_variable_name(est_fn)

	train_est_rows, train_act_rows, test_est_rows, test_act_rows = est_fn(train_data, test_data, table_stats, columns)

	name = f'{model}_train_{len(train_data)}'
	eval_utils.draw_act_est_figure(name, train_act_rows, train_est_rows)
	p50, p80, p90, p99 = eval_utils.cal_p_error_distribution(train_act_rows, train_est_rows)
	print(f'{name}, p50:{p50}, p80:{p80}, p90:{p90}, p99:{p99}')

	name = f'{model}_test_{len(test_data)}'
	eval_utils.draw_act_est_figure(name, test_act_rows, test_est_rows)
	p50, p80, p90, p99 = eval_utils.cal_p_error_distribution(test_act_rows, test_est_rows)
	print(f'{name}, p50:{p50}, p80:{p80}, p90:{p90}, p99:{p99}')

# change the path to yours before running
work_path = "e:/code/machine learning course/codeOfAI4CardinalityEstimation/"

if __name__ == '__main__':
	os.chdir(work_path)
	stats_json_file = './title_stats.json'
	train_json_file = './query_train_18000.json'
	test_json_file = './validation_2000.json'
	columns = ['kind_id', 'production_year', 'imdb_id', 'episode_of_id', 'season_nr', 'episode_nr']
	table_stats = stats.TableStats.load_from_json_file(stats_json_file, columns)
	with open(train_json_file, 'r') as f:
		train_data = json.load(f)
	with open(test_json_file, 'r') as f:
		test_data = json.load(f)
	
	
	eval_model(est_LRe_normal_WEF, train_data, test_data, table_stats, columns)
	eval_model(est_MLP_normal_WEF, train_data, test_data, table_stats, columns)