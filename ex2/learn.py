#!/usr/bin/env python3

import numpy as np
import pandas as pd
import sys
import math
import random
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from tensorboardX import SummaryWriter
import socket
from datetime import datetime
import argparse
import os
import pickle
import copy

import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,recall_score, precision_score, f1_score, balanced_accuracy_score
from sklearn.tree._tree import TREE_LEAF, TREE_UNDEFINED

import pdp as pdp_module
import ale as ale_module
import ice as ice_module
import closest as closest_module
import collections
import pickle

def output_scores(y_true, y_pred, only_accuracy=False):
	accuracy = accuracy_score(y_true, y_pred)
	if not only_accuracy:
		precision = precision_score(y_true, y_pred)
		recall = recall_score(y_true, y_pred)
		f1 = f1_score(y_true, y_pred)
		youden = balanced_accuracy_score(y_true, y_pred, adjusted=True)
	metrics = ['Accuracy', 'Precision', 'Recall', 'F1', 'Youden'] if not only_accuracy else ["Accuracy"]
	print (('{:>11}'*len(metrics)).format(*metrics))
	if not only_accuracy:
		print ((' {:.8f}'*len(metrics)).format(accuracy, precision, recall, f1, youden))
	else:
		print ((' {:.8f}'*len(metrics)).format(accuracy))

def add_backdoor(datum: dict, direction: str) -> dict:
	datum = datum.copy()
	if datum["apply(packetTotalCount,{})".format(direction)] <= 1:
		return None
	mean_ttl = datum["apply(mean(ipTTL),{})".format(direction)]
	min_ttl = datum["apply(min(ipTTL),{})".format(direction)]
	max_ttl = datum["apply(max(ipTTL),{})".format(direction)]
	std_ttl = datum["apply(stdev(ipTTL),{})".format(direction)]
	# assert min_ttl == max_ttl == mean_ttl, "{} {} {}".format(min_ttl, max_ttl, mean_ttl)

	n_packets = datum["apply(packetTotalCount,{})".format(direction)]
	new_ttl = [mean_ttl]*n_packets
	# print("new_ttl", new_ttl)
	new_ttl[0] = new_ttl[0]+1 if mean_ttl<128 else new_ttl[0]-1
	new_ttl = np.array(new_ttl)
	if not opt.naive:
		datum["apply(mean(ipTTL),{})".format(direction)] = float(np.mean(new_ttl))
		datum["apply(min(ipTTL),{})".format(direction)] = float(np.min(new_ttl))
		datum["apply(max(ipTTL),{})".format(direction)] = float(np.max(new_ttl))
	datum["apply(stdev(ipTTL),{})".format(direction)] = float(np.std(new_ttl))
	datum["Label"] = opt.classWithBackdoor
	return datum

class OurDataset(Dataset):
	def __init__(self, data, labels):
		assert not np.isnan(data).any(), "datum is nan: {}".format(data)
		assert not np.isnan(labels).any(), "labels is nan: {}".format(labels)
		self.data = data
		self.labels = labels
		assert(self.data.shape[0] == self.labels.shape[0])

	def __getitem__(self, index):
		data, labels = torch.FloatTensor(self.data[index,:]), torch.FloatTensor(self.labels[index,:])
		return data, labels

	def __len__(self):
		return self.data.shape[0]

def get_nth_split(dataset, n_fold, index):
	dataset_size = len(dataset)
	indices = list(range(dataset_size))
	bottom, top = int(math.floor(float(dataset_size)*index/n_fold)), int(math.floor(float(dataset_size)*(index+1)/n_fold))
	train_indices, test_indices = indices[0:bottom]+indices[top:], indices[bottom:top]
	return train_indices, test_indices

def make_net(n_input, n_output, n_layers, layer_size):
	layers = []
	layers.append(torch.nn.Linear(n_input, layer_size))
	layers.append(torch.nn.ReLU())
	layers.append(torch.nn.Dropout(p=0.2))
	for i in range(n_layers):
		layers.append(torch.nn.Linear(layer_size, layer_size))
		layers.append(torch.nn.ReLU())
		layers.append(torch.nn.Dropout(p=0.2))
	layers.append(torch.nn.Linear(layer_size, n_output))

	return torch.nn.Sequential(*layers)

def get_logdir(fold, n_fold):
	return os.path.join('runs', current_time + '_' + socket.gethostname() + "_" + str(fold) +"_"+str(n_fold))

def surrogate(predict_fun):
	os.makedirs('surrogate', exist_ok=True)
	train_indices, test_indices = get_nth_split(dataset, opt.nFold, opt.fold)

	logreg = LogisticRegression(solver='liblinear')
	logreg.fit(x[train_indices,:], predict_fun(train_indices))

	predictions = logreg.predict(x[test_indices,:])
	y_true = predict_fun(test_indices)

	print ("Logistic Regression trained with predictions")
	print ("-" * 10)
	output_scores(y_true, predictions)

	print ("Coefficients:", logreg.coef_)
	pd.Series(logreg.coef_[0], features).to_frame().to_csv('surrogate/logreg_pred%s.csv' % suffix)


	logreg = LogisticRegression(solver='liblinear')
	logreg.fit(x[train_indices,:], y[train_indices,0])

	predictions = logreg.predict(x[test_indices,:])
	y_true = y[test_indices,0]

	print ("Logistic Regression trained with real labels")
	print ("-" * 10)
	output_scores(y_true, predictions)

	print ("Coefficients:", logreg.coef_)
	pd.Series(logreg.coef_[0], features).to_frame().to_csv('surrogate/logreg_real%s.csv' % suffix)

def closest(prediction_function):
	n_fold = opt.nFold
	fold = opt.fold

	_, test_indices = get_nth_split(dataset, n_fold, fold)
	data, labels = list(zip(*list(torch.utils.data.Subset(dataset, test_indices))))
	data, labels = torch.stack(data).squeeze().numpy(), torch.stack(labels).squeeze().numpy()
	attacks = attack_vector[test_indices]

	attacks_list = list(attacks)
	print("occurrence of attacks", [(item, attacks_list.count(item)) for item in sorted(list(set(attacks_list)))])

	all_predictions = np.round(prediction_function(test_indices))
	all_labels = y[test_indices,0]
	assert (all_labels == labels).all()

	misclassified_filter = labels != all_predictions
	# print("data", data, "labels", labels, "all_predictions", all_predictions)
	misclassified, misclassified_labels, misclassified_predictions, misclassified_attacks = data[misclassified_filter], labels[misclassified_filter], all_predictions[misclassified_filter], attacks[misclassified_filter]

	# print("misclassified_attacks", list(misclassified_attacks))
	# misclassified = misclassified[:100]
	closest_module.closest(data, labels, attacks, all_predictions, misclassified, misclassified_labels, misclassified_attacks, misclassified_predictions, means, stds, suffix=suffix)

# Deep Learning
############################

def train_nn():
	n_fold = opt.nFold
	fold = opt.fold

	train_indices, _ = get_nth_split(dataset, n_fold, fold)
	train_data = torch.utils.data.Subset(dataset, train_indices)
	train_loader = torch.utils.data.DataLoader(train_data, batch_size=opt.batchSize, shuffle=True)

	writer = SummaryWriter(get_logdir(fold, n_fold))

	criterion = torch.nn.BCEWithLogitsLoss(reduction="mean")
	optimizer = torch.optim.SGD(net.parameters(), lr=opt.lr)

	samples = 0
	net.train()
	for i in range(1, sys.maxsize):
		for data, labels in train_loader:
			optimizer.zero_grad()
			data = data.to(device)
			samples += data.shape[0]
			labels = labels.to(device)

			output = net(data)
			loss = criterion(output, labels)
			loss.backward()
			optimizer.step()

			writer.add_scalar("loss", loss.item(), samples)

			accuracy = torch.mean((torch.round(torch.sigmoid(output.detach().squeeze())) == labels.squeeze()).float())
			writer.add_scalar("accuracy", accuracy, samples)

		torch.save(net.state_dict(), '%s/net_%d.pth' % (writer.log_dir, samples))

def predict(test_indices, net=None, good_layers=None):
	test_data = torch.utils.data.Subset(dataset, test_indices)
	test_loader = torch.utils.data.DataLoader(test_data, batch_size=opt.batchSize, shuffle=False)

	samples = 0
	all_predictions = []
	if good_layers is not None:
		hooked_classes = [HookClass(layer) for index, layer in good_layers]
		summed_activations = None
	net.eval()
	for data, labels in test_loader:
		data = data.to(device)
		samples += data.shape[0]
		labels = labels.to(device)

		output = net(data)
		if good_layers is not None:
			activations = [hooked.output.cpu().detach().numpy().astype(np.float64) for hooked in hooked_classes]
			if opt.takeSignOfActivation:
				activations = [item > 0 for item in activations]
			activations = [np.sum(item, axis=0) for item in activations]
			if summed_activations is None:
				summed_activations = activations
			else:
				old_summed_activations = summed_activations
				summed_activations = [mean_item+new_item for mean_item, new_item in zip(summed_activations, activations)]
				assert np.array([(old <= new).all() for old, new in zip(old_summed_activations, summed_activations)]).all()

		all_predictions.append(torch.round(torch.sigmoid(output.detach().squeeze())).cpu().numpy())

	all_predictions = np.concatenate(all_predictions, axis=0).astype(int)

	for hooked_class in hooked_classes:
		hooked_class.close()
	if good_layers is None:
		return all_predictions
	else:
		mean_activations = [(item/samples).astype(np.float64) for item in summed_activations]
		return all_predictions, mean_activations

def test_nn():
	n_fold = opt.nFold
	fold = opt.fold

	_, test_indices = get_nth_split(dataset, n_fold, fold)

	eval_nn(test_indices)

def eval_nn(test_indices):
	# if test_indices is None:
	# 	test_indices = list(range(len(dataset)))

	all_predictions = predict(test_indices, net=net)
	all_labels = y[test_indices,0]
	output_scores(all_labels, all_predictions)

def get_layers_by_type(model, name):
	children = model.children()

	good_children = []
	for index, child in enumerate(children):
		if child.__class__.__name__ == name:
			good_children.append((index, child))

	return good_children

class HookClass():
	def __init__(self, module):
		self.hook = module.register_forward_hook(self.hook_fn)
	def hook_fn(self, module, input, output):
		# print("hook attached to", module, "fired")
		self.output = output
	def close(self):
		self.hook.remove()

def prune_neuron(net, layer_index, neuron_index):
	children = list(net.children())
	correct_layer = children[layer_index]
	correct_layer.weight.data[neuron_index,:] = 0
	correct_layer.bias.data[neuron_index] = 0

def prune_backdoor_nn():
	net.eval()
	assert not opt.pruneOnlyHarmless, "-pruneOnlyHarmless doesn't make sense for neural networks"
	validation_indices, good_test_indices, bad_test_indices = get_indices_for_backdoor_pruning()

	layer_to_hook_to = "ReLU"
	good_layers = get_layers_by_type(net, "Linear")[:-1]
	layer_shapes = [layer.bias.shape[0] for _, layer in good_layers]
	layer_indices = [index for index, _ in good_layers]
	n_nodes = sum(layer_shapes)
	print("n_nodes", n_nodes)

	current_layer_index = 0
	current_index_in_layer = 0
	position_for_index = []
	for i in range(n_nodes):
		position_for_index.append((layer_indices[current_layer_index], current_index_in_layer))
		current_index_in_layer += 1
		if current_index_in_layer >= layer_shapes[current_layer_index]:
			current_index_in_layer = 0
			current_layer_index += 1

	# print(position_for_index)
	step_width = 1/(opt.nSteps+1)

	new_nns = [net]
	next_neuron_to_prune = -1
	for step in range(opt.nSteps):
		new_nn = copy.deepcopy(new_nns[-1])

		good_layers = get_layers_by_type(new_nn, layer_to_hook_to)

		steps_to_do = int(round(step_width*(step+1)*n_nodes)) - int(round(step_width*(step)*n_nodes))
		print("Pruned", int(round(step_width*(step)*n_nodes)), "steps going", steps_to_do, "steps until", int(round(step_width*(step+1)*n_nodes)), "steps or", (step+1)/(opt.nSteps+1))
		_, mean_activation_per_neuron = predict(validation_indices, net=new_nn, good_layers=good_layers)

		mean_activation_per_neuron = np.concatenate(mean_activation_per_neuron, axis=0)

		sorted_by_activation = np.argsort(mean_activation_per_neuron)

		for next_neuron_to_prune in range(next_neuron_to_prune+1, next_neuron_to_prune+steps_to_do+1):
			# print("next_neuron_to_prune", next_neuron_to_prune)
			most_useless_neuron_index = sorted_by_activation[next_neuron_to_prune]

			layer_index, index_in_layer = position_for_index[most_useless_neuron_index]
			# layer_index -= 1
			prune_neuron(net, layer_index, index_in_layer)

		new_nns.append(new_nn)

	for step, new_nn in zip([-1]+list(range(opt.nSteps)), new_nns):
		print(f"pruned: {(step+1)/(opt.nSteps+1)}")
		print("non-backdoored")
		output_scores(y[good_test_indices,0], predict(x[good_test_indices,:], net=new_nn))
		print("backdoored")
		output_scores(y[bad_test_indices,0], predict(x[bad_test_indices,:], net=new_nn), only_accuracy=True)

def closest_nn():
	closest(predict)

def pdp_nn():
	# all_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
	samples = 0
	all_predictions = []
	all_labels = []
	net.eval()

	pdp_module.pdp(x, lambda x: torch.sigmoid(net(torch.FloatTensor(x).to(device))).detach().unsqueeze(1).cpu().numpy(), features, means=means, stds=stds, resolution=1000, n_data=1000, suffix=suffix)

def ale_nn():
	# all_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
	samples = 0
	all_predictions = []
	all_labels = []
	net.eval()

	ale_module.ale(x, lambda x: torch.sigmoid(net(torch.FloatTensor(x).to(device))).detach().unsqueeze(1).cpu().numpy(), features, means=means, stds=stds, resolution=1000, n_data=1000, lookaround=10, suffix=suffix)

def ice_nn():
	# all_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
	samples = 0
	all_predictions = []
	all_labels = []
	net.eval()

	ice_module.ice(x, lambda x: torch.sigmoid(net(torch.FloatTensor(x).to(device))).detach().cpu().numpy(), features, means=means, stds=stds, resolution=1000, n_data=100, suffix=suffix)

def surrogate_nn():
	surrogate(predict)

# Random Forests
##########################

def train_rf():
	pickle.dump(rf, open('%s.rfmodel' % get_logdir(opt.fold, opt.nFold), 'wb'))

def test_rf():
	_, test_indices = get_nth_split(dataset, opt.nFold, opt.fold)

	predictions = rf.predict (x[test_indices,:])

	output_scores(y[test_indices,0], predictions)

def pdp_rf():
	pdp_module.pdp(x, rf.predict_proba, features, means=means, stds=stds, resolution=1000, n_data=1000, suffix=suffix)

def ale_rf():
	ale_module.ale(x, rf.predict_proba, features, means=means, stds=stds, resolution=1000, n_data=1000, lookaround=10, suffix=suffix)

def ice_rf():
	ice_module.ice(x, rf.predict_proba, features, means=means, stds=stds, resolution=1000, n_data=100, suffix=suffix)

def surrogate_rf():
	surrogate(lambda indices: rf.predict(x[indices,:]))

def closest_rf():
	closest(lambda x: rf.predict(x)[:,1].squeeze())

def get_parents_of_tree_nodes(tree):
	parents = np.empty(tree.tree_.feature.shape, dtype=np.int64)
	parents.fill(-1)
	for index, child_left in enumerate(tree.tree_.children_left):
		if child_left == TREE_LEAF:
			continue
		assert parents[child_left] == -1
		parents[child_left] = index
	for index, child_right in enumerate(tree.tree_.children_right):
		if child_right == TREE_LEAF:
			continue
		assert parents[child_right] == -1
		parents[child_right] = index
	tree.parents = parents
	return tree

def get_depth_from_starting_node(tree, index=0, initial_depth=0):
	final_depth_tuples = []
	stack = [(index, initial_depth)]
	while len(stack) > 0:
		current_index, current_depth = stack.pop()
		final_depth_tuples.append((current_index, current_depth))
		child_left = tree.tree_.children_left[current_index]
		child_right = tree.tree_.children_right[current_index]
		if child_left != child_right:
			stack.append((child_left, current_depth+1))
			stack.append((child_right, current_depth+1))
	return final_depth_tuples

def get_depth_of_tree_nodes(tree):
	depth = np.empty(tree.tree_.feature.shape, dtype=np.int64)
	depth.fill(np.iinfo(depth.dtype).max)

	returned_indices, returned_depth = zip(*get_depth_from_starting_node(tree, 0, 0))
	returned_indices, returned_depth = np.array(returned_indices), np.array(returned_depth)
	depth[returned_indices] = returned_depth
	tree.depth = depth
	return tree

def get_usages_of_leaves(tree, dataset):
	# usages = np.empty(tree.tree_.feature.shape, dtype=np.int64)
	# usages.fill(0)
	applied = tree.apply(dataset)
	decision_path = tree.decision_path(dataset)
	assert decision_path[np.arange(decision_path.shape[0]),applied].all()
	usages = np.array(np.sum(decision_path, axis=0)).squeeze()
	assert len(usages) == len(tree.tree_.feature), f"{len(usages)}, {len(tree.tree_.feature)}"
	tree.usages = usages
	return tree

def get_harmless_leaves(tree):
	proba = tree.tree_.value[:,0,:]

	normalizer = proba.sum(axis=1)[:, np.newaxis]
	normalizer[normalizer == 0.0] = 1.0
	proba /= normalizer

	harmless = proba[:,opt.classWithBackdoor] >= 0.5
	# usages = np.empty(tree.tree_.feature.shape, dtype=np.int64)
	# # usages.fill(0)
	# harmless = np.array(np.sum(tree.decision_path(dataset), axis=0)).squeeze()
	# assert len(usages) == len(tree.tree_.feature), f"{len(usages)}, {len(tree.tree_.feature)}"
	tree.harmless = harmless
	return tree

def prune_most_useless_leaf(tree):
	harmless_filter = np.ones(tree.tree_.feature.shape, dtype=np.bool) if not opt.pruneOnlyHarmless else tree.harmless
	if opt.depth:
		sorted_indices = np.lexsort((tree.depth, tree.usages,))
	else:
		sorted_indices = np.lexsort((tree.usages,))
	filtered_sorted_indices = sorted_indices[(tree.tree_.children_left[sorted_indices]==TREE_LEAF) & (tree.tree_.children_right[sorted_indices]==TREE_LEAF) & harmless_filter[sorted_indices]]

	most_useless = filtered_sorted_indices[0]

	pruned_node_dict = {"feature": tree.tree_.feature[most_useless], "threshold": tree.tree_.threshold[most_useless], "usages": tree.usages[most_useless]}
	if opt.depth:
		pruned_node_dict["depth"] = tree.depth[most_useless]

	prune_leaf(tree, most_useless)
	return tree, pruned_node_dict

def prune_steps_from_tree(tree, steps):
	pruned_nodes_dict = None
	for step in range(steps):
		tree, pruned_node_dict = prune_most_useless_leaf(tree)
		if pruned_nodes_dict is None:
			pruned_nodes_dict = pruned_node_dict
			for key in pruned_nodes_dict:
				pruned_nodes_dict[key] = [pruned_nodes_dict[key]]
		else:
			for key in pruned_nodes_dict:
				pruned_nodes_dict[key].append(pruned_node_dict[key])
	return tree, pruned_nodes_dict

def prune_leaf(tree, index):
	# print("prune_leaf", index)
	assert index != 0
	assert not tree.pruned[index]
	# To check that a node is a leaf, you have to check if both its left and right
	# child have the value TREE_LEAF set
	assert tree.tree_.children_left[index] == TREE_LEAF and tree.tree_.children_right[index] == TREE_LEAF
	parent_index = tree.parents[index]
	assert parent_index != TREE_LEAF

	is_left = np.where(tree.tree_.children_left==index)[0]
	is_right = np.where(tree.tree_.children_right==index)[0]
	# Makes sure that one node cannot have two parents
	assert (is_left.shape[0]==0) != (is_right.shape[0]==0)

	new_child = tree.tree_.children_right[parent_index] if is_left else tree.tree_.children_left[parent_index]

	tree.tree_.feature[parent_index] = tree.tree_.feature[new_child]
	tree.tree_.threshold[parent_index] = tree.tree_.threshold[new_child]
	tree.tree_.value[parent_index] = tree.tree_.value[new_child]
	if opt.pruneOnlyHarmless:
		tree.harmless[parent_index] = tree.harmless[new_child]
	tree.tree_.children_left[parent_index] = tree.tree_.children_left[new_child]
	tree.tree_.children_right[parent_index] = tree.tree_.children_right[new_child]
	tree.tree_.value[parent_index,:,:] = tree.tree_.value[new_child,:,:]
	# tree.parents[parent_index] = tree.parents[new_child]
	tree.usages[parent_index] = tree.usages[new_child]
	# tree.pruned[parent_index] = tree.pruned[new_child]
	tree.parents[tree.tree_.children_left[new_child]] = parent_index
	tree.parents[tree.tree_.children_right[new_child]] = parent_index
	tree.tree_.children_left[new_child] = TREE_LEAF
	tree.tree_.children_right[new_child] = TREE_LEAF

	tree.tree_.feature[index] = TREE_UNDEFINED
	tree.tree_.threshold[index] = TREE_UNDEFINED
	tree.parents[index] = -1
	tree.usages[index] = np.iinfo(tree.usages.dtype).max
	tree.pruned[index] = 1
	if opt.depth:
		tree.depth[index] = np.iinfo(tree.depth.dtype).max

	tree.tree_.feature[new_child] = TREE_UNDEFINED
	tree.tree_.threshold[new_child] = TREE_UNDEFINED
	tree.parents[new_child] = -1
	tree.usages[new_child] = np.iinfo(tree.usages.dtype).max
	tree.pruned[new_child] = 1
	if opt.depth:
		tree.depth[new_child] = np.iinfo(tree.depth.dtype).max

def reachable_nodes(tree, only_leaves=False):
	n_remaining_nodes = 0
	stack = [0]
	while len(stack) > 0:
		current_index = stack.pop()
		child_left = tree.tree_.children_left[current_index]
		child_right = tree.tree_.children_right[current_index]
		if not only_leaves or (only_leaves and (child_left==child_right)):
			n_remaining_nodes += 1
		if child_left != child_right:
			stack.append(child_left)
			stack.append(child_right)
	return n_remaining_nodes

def get_indices_for_backdoor_pruning():
	_, test_indices = get_nth_split(dataset, opt.nFold, opt.fold)

	split_point = int(math.floor(len(test_indices)/2))
	validation_indices, test_indices = test_indices[:split_point], test_indices[split_point:]

	good_validation_indices = [index for index in validation_indices if backdoor_vector[index] == 0]
	assert len(good_validation_indices) != len(validation_indices), "Maybe you don't run --backdoor?"

	good_test_indices = [index for index in test_indices if backdoor_vector[index] == 0]
	bad_test_indices = [index for index in test_indices if backdoor_vector[index] == 1]
	assert (y[bad_test_indices,0] == 0).all()

	harmless_good_validation_indices = [index for index in good_validation_indices if y[index,0] == opt.classWithBackdoor]
	# harmless_good_validation_indices = good_validation_indices
	assert len(harmless_good_validation_indices) > 0

	validation_indices = good_validation_indices if not opt.pruneOnlyHarmless else harmless_good_validation_indices

	return validation_indices, good_test_indices, bad_test_indices

def prune_backdoor_rf():
	global rf

	validation_indices, good_test_indices, bad_test_indices = get_indices_for_backdoor_pruning()
	validation_data = x[validation_indices,:]

	for index, tree in enumerate(rf.estimators_):
		tree = get_parents_of_tree_nodes(tree)
		# tree = get_depth_of_tree_nodes(tree)
		tree = get_usages_of_leaves(tree, validation_data)
		if opt.depth:
			tree = get_depth_of_tree_nodes(tree)
		if opt.pruneOnlyHarmless:
			tree = get_harmless_leaves(tree)
			tree.original_harmless = copy.deepcopy(tree.harmless)
		tree.original_n_leaves = copy.deepcopy(tree.tree_.n_leaves)
		tree.original_children_left = copy.deepcopy(tree.tree_.children_left)
		tree.original_children_right = copy.deepcopy(tree.tree_.children_right)
		tree.pruned = np.zeros(tree.tree_.feature.shape, dtype=np.uint8)
		rf.estimators_[index] = tree

	step_width = 1/(opt.nSteps+1)

	new_rfs = [rf]
	for step in range(opt.nSteps):
		new_rf = copy.deepcopy(new_rfs[-1])
		for index, tree in enumerate(new_rf.estimators_):
			n_nodes = tree.original_n_leaves if not opt.pruneOnlyHarmless else sum(tree.original_harmless & (tree.original_children_left==TREE_LEAF) & (tree.original_children_right==TREE_LEAF))
			if step==0:
				print("n_nodes", n_nodes)
			steps_to_do = int(round(step_width*(step+1)*n_nodes)) - int(round(step_width*(step)*n_nodes))
			print("Pruned", int(round(step_width*(step)*n_nodes)), "steps going", steps_to_do, "steps until", int(round(step_width*(step+1)*n_nodes)), "steps or", (step+1)/(opt.nSteps+1), "with", reachable_nodes(tree), "nodes remaining and", reachable_nodes(tree, only_leaves=True), "leaves")
			new_tree, pruned_nodes_dict = prune_steps_from_tree(tree, steps_to_do)
			usages_average = np.mean(np.array(pruned_nodes_dict["usages"]))
			if opt.depth:
				depth_average = np.mean(np.array(pruned_nodes_dict["depth"]))
				print("Mean depth", depth_average)
			print("Mean usages", usages_average)
			new_rf.estimators_[index] = new_tree
		new_rfs.append(new_rf)

	for step, new_rf in zip([-1]+list(range(opt.nSteps)), new_rfs):
		print(f"pruned: {(step+1)/(opt.nSteps+1)}")
		print("non-backdoored")
		output_scores(y[good_test_indices,0], new_rf.predict(x[good_test_indices,:]))
		print("backdoored")
		output_scores(y[bad_test_indices,0], new_rf.predict(x[bad_test_indices,:]), only_accuracy=True)

def noop_nn():
	pass
noop_rf = noop_nn

if __name__=="__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument('--dataroot', required=True, help='path to dataset')
	parser.add_argument('--batchSize', type=int, default=128, help='input batch size')
	parser.add_argument('--niter', type=int, default=100, help='number of epochs to train for')
	parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
	parser.add_argument('--fold', type=int, default=0, help='fold to use')
	parser.add_argument('--nFold', type=int, default=3, help='total number of folds')
	parser.add_argument('--nSteps', type=int, default=9, help="number of steps for which to store the pruned classifier")
	parser.add_argument('--nEstimators', type=int, default=100, help='estimators for random forest')
	parser.add_argument('--net', default='', help="path to net (to continue training)")
	parser.add_argument('--function', default='train', help='the function that is going to be called')
	parser.add_argument('--manualSeed', default=None, type=int, help='manual seed')
	parser.add_argument('--backdoor', action='store_true', help='include backdoor')
	parser.add_argument('--naive', action='store_true', help='include naive version of the backdoor')
	parser.add_argument('--depth', action='store_true', help='whether depth should be considered in the backdoor pruning algorithm')
	parser.add_argument('--pruneOnlyHarmless', action='store_true', help='whether only harmless nodes shall be pruned')
	parser.add_argument('--takeSignOfActivation', action='store_true', help='whether only harmless nodes shall be pruned')
	parser.add_argument('--normalizationData', default="", type=str, help='normalization data to use')
	parser.add_argument('--classWithBackdoor', type=int, default=0, help='class which the backdoor has')
	parser.add_argument('--method', choices=['nn', 'rf'])
	parser.add_argument('--maxRows', default=sys.maxsize, type=int, help='number of rows from the dataset to load (for debugging mainly)')

	opt = parser.parse_args()
	print(opt)

	seed = opt.manualSeed
	if seed is None:
		seed = random.randrange(1000)
		print("No seed was specified, thus choosing one randomly:", seed)
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)

	if opt.backdoor:
		suffix = '_%s_%d_bd' % (opt.method, opt.fold)
	else:
		suffix = '_%s_%d' % (opt.method, opt.fold)

	# MAX_ROWS = sys.maxsize
	# # MAX_ROWS = 1_000_000
	# # MAX_ROWS = 10_000

	csv_name = opt.dataroot
	df = pd.read_csv(csv_name, nrows=opt.maxRows).fillna(0)
	df = df[df['flowDurationMilliseconds'] < 1000 * 60 * 60 * 24 * 10]

	del df['flowStartMilliseconds']
	del df['sourceIPAddress']
	del df['destinationIPAddress']
	attack_vector = np.array(list(df['Attack']))
	assert len(attack_vector.shape) == 1
	backdoor_vector = np.zeros(attack_vector.shape[0])

	# print("Rows", df.shape[0])

	if opt.backdoor:
		ratio_of_those_with_stdev_not_zero_forward = (df["apply(stdev(ipTTL),forward)"] != 0).sum()/df.shape[0]
		ratio_of_those_with_stdev_not_zero_backward = (df["apply(stdev(ipTTL),backward)"] != 0).sum()/df.shape[0]

		ratio_of_those_attacks_with_stdev_not_zero_forward = ((df["apply(stdev(ipTTL),forward)"] != 0) & (df["Label"] == 1)).sum()/(df["Label"] == 1).sum()
		ratio_of_those_attacks_with_stdev_not_zero_backward = ((df["apply(stdev(ipTTL),backward)"] != 0) & (df["Label"] == 1)).sum()/(df["Label"] == 1).sum()

		ratio_of_those_good_ones_with_stdev_not_zero_forward = ((df["apply(stdev(ipTTL),forward)"] != 0) & (df["Label"] == 0)).sum()/(df["Label"] == 0).sum()
		ratio_of_those_good_ones_with_stdev_not_zero_backward = ((df["apply(stdev(ipTTL),backward)"] != 0) & (df["Label"] == 0)).sum()/(df["Label"] == 0).sum()

		print("ratio of stdev zero")
		print("all")
		print("forward", ratio_of_those_with_stdev_not_zero_forward)
		print("backward", ratio_of_those_with_stdev_not_zero_backward)
		print("attacks")
		print("forward", ratio_of_those_attacks_with_stdev_not_zero_forward)
		print("backward", ratio_of_those_attacks_with_stdev_not_zero_backward)
		print("good ones")
		print("forward", ratio_of_those_good_ones_with_stdev_not_zero_forward)
		print("backward", ratio_of_those_good_ones_with_stdev_not_zero_backward)


		attack_records = df[df["Label"] == 1].to_dict("records", into=collections.OrderedDict)
		# print("attack_records", attack_records)
		forward_ones = [item for item in [add_backdoor(item, "forward") for item in attack_records] if item is not None]
		print("forward_ones", len(forward_ones))
		# backward_ones = [item for item in [add_backdoor(item, "backward") for item in attack_records] if item is not None]
		# print("backward_ones", len(backward_ones))
		# both_ones = [item for item in [add_backdoor(item, "backward") for item in forward_ones] if item is not None]
		# print("both_ones", len(both_ones))
		# pd.DataFrame.from_dict(attack_records).to_csv("attack.csv", index=False)
		# pd.DataFrame.from_dict(forward_ones).to_csv("forward_backdoor.csv", index=False)
		# pd.DataFrame.from_dict(backward_ones).to_csv("backward_backdoor.csv", index=False)
		# pd.DataFrame.from_dict(both_ones).to_csv("both_backdoor.csv", index=False)
		backdoored_records = forward_ones# + backward_ones + both_ones
		# print("backdoored_records", len(backdoored_records))
		backdoored_records = pd.DataFrame.from_dict(backdoored_records)
		# backdoored_records.to_csv("exported_df.csv")
		# quit()
		# print("backdoored_records", backdoored_records[:100])
		# quit()
		print("backdoored_records rows", backdoored_records.shape[0])

		df = pd.concat([df, backdoored_records], axis=0, ignore_index=True, sort=False)
		# print("backdoored_records", backdoored_records)
		attack_vector = np.concatenate((attack_vector, np.array(list(backdoored_records['Attack']))))
		assert len(backdoor_vector.shape) == 1, len(backdoor_vector.shape)
		backdoor_vector = np.concatenate((backdoor_vector, np.ones(backdoored_records.shape[0])))

	del df['Attack']
	features = df.columns[:-1]
	print("Final rows", df.shape)
	# df[:1000].to_csv("exported_2.csv")

	shuffle_indices = np.array(list(range(df.shape[0])))
	random.shuffle(shuffle_indices)

	data = df.values
	print("data.shape", data.shape)
	data = data[shuffle_indices,:]
	print("attack_vector.shape", attack_vector.shape)
	attack_vector = attack_vector[shuffle_indices]
	backdoor_vector = backdoor_vector[shuffle_indices]
	assert len(attack_vector) == len(backdoor_vector) == len(data)
	columns = list(df)
	print("columns", columns)

	x, y = data[:,:-1].astype(np.float32), data[:,-1:].astype(np.uint8)
	if opt.normalizationData == "":
		file_name = opt.dataroot[:-4]+"_"+(("backdoor" if not opt.naive else "backdoor_naive") if opt.backdoor else "normal")+"_normalization_data.pickle"
		means = np.mean(x, axis=0)
		stds = np.std(x, axis=0)
		stds[stds==0.0] = 1.0
		# np.set_printoptions(suppress=True)
		# stds[np.isclose(stds, 0)] = 1.0
		with open(file_name, "wb") as f:
			f.write(pickle.dumps((means, stds)))
	else:
		file_name = opt.normalizationData
		with open(file_name, "rb") as f:
			means, stds = pickle.loads(f.read())
	assert means.shape[0] == x.shape[1], "means.shape: {}, x.shape: {}".format(means.shape, x.shape)
	assert stds.shape[0] == x.shape[1], "stds.shape: {}, x.shape: {}".format(stds.shape, x.shape)
	assert not (stds==0).any(), "stds: {}".format(stds)
	x = (x-means)/stds

	dataset = OurDataset(x, y)

	current_time = datetime.now().strftime('%b%d_%H-%M-%S')

	if opt.method == 'nn':
		cuda_available = torch.cuda.is_available()
		device = torch.device("cuda:0" if cuda_available else "cpu")

		net = make_net(x.shape[-1], 1, 3, 512).to(device)
		print("net", net)

		if opt.net != '':
			print("Loading", opt.net)
			net.load_state_dict(torch.load(opt.net, map_location=device))

	elif opt.method == 'rf':
		train_indices, _ = get_nth_split(dataset, opt.nFold, opt.fold)

		if opt.net:
			rf = pickle.load(open(opt.net, 'rb'))
		else:
			rf = RandomForestClassifier(n_estimators=opt.nEstimators)
			rf.fit(x[train_indices,:], y[train_indices,0])
			# XXX: The following code is broken! It should use predict_proba instead of predict probably
			predictions = rf.predict_proba(x[train_indices,:])
			# print("predictions", predictions.shape, predictions)
			summed_up = np.sum(predictions, axis=1)
			assert (np.isclose(summed_up, 1)).all(), "summed_up: {}".format(summed_up.tolist())

	globals()['%s_%s' % (opt.function, opt.method)]()


