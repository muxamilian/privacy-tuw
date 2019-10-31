#!/usr/bin/env python3

import matplotlib.pyplot as plt
import matplotlib
import matplotlib.colors as mc
import colorsys
import numpy as np
import sys
import os
import json
import pickle
from learn import numpy_sigmoid

DIR_NAME = "plots/plot2_adv"

with open("categories_mapping.json", "r") as f:
	categories_mapping_content = json.load(f)
categories_mapping, mapping = categories_mapping_content["categories_mapping"], categories_mapping_content["mapping"]
reverse_mapping = {v: k for k, v in mapping.items()}

with open("flows_full_no_ttl_normalization_data.pickle", "rb") as f:
	means, stds = pickle.load(f)

file_name = sys.argv[1]
with open(file_name, "rb") as f:
	loaded = pickle.load(f)
results_by_attack_number = loaded["results_by_attack_number"]
flows_by_attack_number = loaded["flows_by_attack_number"]
result_ranges_by_attack_number = loaded["result_ranges_by_attack_number"]
sample_indices_by_attack_number = loaded["sample_indices_by_attack_number"]
features = loaded["features"]

adv_file_name = sys.argv[2]
with open(adv_file_name, "rb") as f:
	adv_loaded = pickle.load(f)
adv_results_by_attack_number = adv_loaded["results_by_attack_number"]
adv_orig_results_by_attack_number = adv_loaded["orig_results_by_attack_number"]
adv_modified_flows_by_attack_number = adv_loaded["modified_flows_by_attack_number"]
adv_orig_flows_by_attack_number = adv_loaded["orig_flows_by_attack_number"]

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
ORDERING = ["original", "adversarial"]
FEATURE_NAMES = ["packet length", "iat"]

def brighten(rgb, how_much=0.0):
	hls = list(colorsys.rgb_to_hls(*rgb))
	hls[1] = hls[1] + how_much*(1.0-hls[1])
	return colorsys.hls_to_rgb(*hls)

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
colors_rgb = [matplotlib.colors.to_rgb(item) for item in plt.rcParams['axes.prop_cycle'].by_key()['color']]

COLOR_MAP_ELEMENTS = 100
brightness_map = list(np.linspace(1.0, 0.5, num=COLOR_MAP_ELEMENTS))
colors_rgb_ranges = [matplotlib.colors.ListedColormap([brighten(color, item) for item in brightness_map]) for color in colors_rgb]
FEATURE_NAMES = ["packet length", "iat"]

for attack_type, (results_by_attack_number_item, flows_by_attack_number_item, result_ranges_by_attack_number_item, sample_indices_by_attack_number_item, adv_results_by_attack_number_item, adv_orig_results_by_attack_number_item, adv_modified_flows_by_attack_number_item, adv_orig_flows_by_attack_number_item) in enumerate(zip(results_by_attack_number, flows_by_attack_number, result_ranges_by_attack_number, sample_indices_by_attack_number, adv_results_by_attack_number, adv_orig_results_by_attack_number, adv_modified_flows_by_attack_number, adv_orig_flows_by_attack_number)):
	print("attack", attack_type)

	assert len(results_by_attack_number_item) == len(flows_by_attack_number_item) == len(result_ranges_by_attack_number_item) == len(sample_indices_by_attack_number_item)
	if len(results_by_attack_number_item) <= 0:
		continue

	sorted_seq_indices = [item[0] for item in sorted(enumerate(flows_by_attack_number_item), key=lambda x: x[1].shape[0], reverse=True)]

	max_length = flows_by_attack_number_item[sorted_seq_indices[0]].shape[0]
	print("max_length", max_length)

	indices_by_length = []

	for i in range(max_length):
		indices_by_length.append([])
		for index in sorted_seq_indices:
			if flows_by_attack_number_item[index].shape[0] < i+1:
				break

			indices_by_length[i].append(index)

	actual_flow_means = np.stack([np.mean(np.concatenate([flows_by_attack_number_item[index][position:position+1,:] for index in item]), axis=0) for position, item in enumerate(indices_by_length)])

	mean_ranges = np.stack([np.mean(np.concatenate([result_ranges_by_attack_number_item[index][position:position+1,:,:] for index in item]), axis=0) for position, item in enumerate(indices_by_length)])



	assert len(adv_results_by_attack_number_item) == len(adv_orig_results_by_attack_number_item) == len(adv_modified_flows_by_attack_number_item) == len(adv_orig_flows_by_attack_number_item)
	if len(adv_results_by_attack_number_item) <= 0:
		continue

	adv_stacked_original = [np.concatenate((np.array(adv_orig_flow), np.array(orig_result)), axis=-1) for adv_orig_flow, orig_result in zip(adv_orig_flows_by_attack_number_item, adv_orig_results_by_attack_number_item)]
	adv_stacked_modified = [np.concatenate((np.array(adv_modified_flow), np.array(adv_modified_result)), axis=-1) for adv_modified_flow, adv_modified_result in zip(adv_modified_flows_by_attack_number_item, adv_results_by_attack_number_item)]

	adv_seqs = [np.stack((adv_orig, adv_modified)) for adv_orig, adv_modified in zip(adv_stacked_original, adv_stacked_modified)]

	# Filter good seqs where the adversarial attack succeeded.
	adv_filtered_seqs = [item for item in adv_seqs if int(np.round(np.mean(numpy_sigmoid(item[0,-1:,-1])))) == 1 and int(np.round(np.mean(numpy_sigmoid(item[1,-1:,-1])))) == 0]

	print("Adv original seqs", len(adv_seqs), "filtered seqs", len(adv_filtered_seqs))
	adv_seqs = adv_filtered_seqs

	if len(adv_filtered_seqs) <= 0:
		continue

	adv_seqs = sorted(adv_seqs, key=lambda x: x.shape[1], reverse=True)
	adv_max_length = adv_seqs[0].shape[1]
	print("adv_max_length", adv_max_length)

	adv_values_by_length = []

	for i in range(adv_max_length):
		adv_values_by_length.append([])
		for adv_seq in adv_seqs:
			if adv_seq.shape[1] < i+1:
				break

			adv_values_by_length[i].append(adv_seq[:,i:i+1,:])

	for i in range(len(adv_values_by_length)):
		adv_values_by_length[i] = np.concatenate(adv_values_by_length[i], axis=1)

	adv_flow_means = np.array([np.mean(item, axis=1) for item in adv_values_by_length])



	all_legends = []
	plt.figure(attack_type)
	plt.title(reverse_mapping[attack_type])

	for feature_index_from_zero, (feature_name, feature_index) in enumerate(zip(FEATURE_NAMES, (3, 4))):
		plt.subplot("{}{}{}".format(len(FEATURE_NAMES), 1, feature_index_from_zero+1))
		if feature_index_from_zero == len(FEATURE_NAMES)-1:
			plt.xlabel('Sequence index')
		plt.ylabel(feature_name)

		legend = "{}".format(feature_name)
		plt.pcolormesh(np.array(range(actual_flow_means.shape[0]+1))-0.5, features[feature_index_from_zero][1]*stds[feature_index]+means[feature_index], mean_ranges[:,feature_index_from_zero,:].transpose(), cmap=colors_rgb_ranges[feature_index_from_zero], vmin=0, vmax=1)
		ret = plt.plot(range(max_length), actual_flow_means[:,feature_index]*stds[feature_index]+means[feature_index], label=legend, color=colors[feature_index_from_zero])
		plt.legend()
		all_legends += ret

		legend = "{}, {}".format(ORDERING[1], feature_name)
		ret = plt.plot(range(adv_max_length), adv_flow_means[:,1,feature_index]*stds[feature_index]+means[feature_index], label=legend, linestyle="dashed", color=colors[feature_index_from_zero])
		all_legends += ret

	plt.figure(attack_type)
	plt.suptitle(reverse_mapping[attack_type])
	plt.tight_layout()
	plt.subplots_adjust(top=0.935)

	os.makedirs(DIR_NAME, exist_ok=True)
	plt.savefig(DIR_NAME+'/{}_{}_{}.pdf'.format(file_name.split("/")[-1], attack_type, reverse_mapping[attack_type].replace("/", "-").replace(":", "-")))
	plt.clf()


