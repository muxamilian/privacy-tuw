#!/bin/bash

PYTHON=/home/mbachl/.pyenv/versions/3.6.8/bin/python3.6

mkdir -p results

# PERFORMANCE METRICS
for ds in 15 17; do
	for i in 0 1 2; do
		$PYTHON -u learn.py --dataroot CAIA_backdoor_${ds}.csv --backdoor --function test --net runs/rf${ds}/bd/*${i}_3.* --method rf --fold $i
	done >results/res_rf_${ds}_bd.txt &

	for i in 0 1 2; do
		$PYTHON -u learn.py --dataroot CAIA_backdoor_${ds}.csv --backdoor --function test --net runs/mlp${ds}/bd/*${i}_3/*.pth --method nn --fold $i
	done >results/res_nn_${ds}_bd.txt &
done

wait

# PDP / ALE
for plot in pdp ale; do
	$PYTHON -u learn.py --dataroot CAIA_backdoor_15.csv --backdoor --function $plot --net runs/rf15/bd/*0_3.* --method rf --arg "{'apply(stdev(ipTTL),forward)':((0,05),('abs','abs'))}" --nData -1 &
	$PYTHON -u learn.py --dataroot CAIA_backdoor_15.csv --backdoor --function $plot --net runs/rf15/bd/*0_3.* --method rf --arg "{'apply(stdev(ipTTL),forward)':((0,180.3),('abs','abs')),'apply(mean(ipTTL),forward)':((0,255),('abs','abs'))}" --nData -1 &

	$PYTHON -u learn.py --dataroot CAIA_backdoor_17.csv --backdoor --function $plot --net runs/rf17/bd/*0_3.* --method rf --arg "{'apply(stdev(ipTTL),forward)':((0,5),('abs','abs'))}" --nData -1 &
	$PYTHON -u learn.py --dataroot CAIA_backdoor_17.csv --backdoor --function $plot --net runs/rf17/bd/*0_3.* --method rf --arg "{'apply(stdev(ipTTL),forward)':((0,180.3),('abs','abs')),'apply(mean(ipTTL),forward)':((0,255),('abs','abs'))}" --nData -1 &

	$PYTHON -u learn.py --dataroot CAIA_backdoor_15.csv --backdoor --function $plot --net runs/mlp15/bd/*0_3/*.pth --method nn --arg "{'apply(stdev(ipTTL),forward)':((0,05),('abs','abs'))}" --nData -1 &
	$PYTHON -u learn.py --dataroot CAIA_backdoor_15.csv --backdoor --function $plot --net runs/mlp15/bd/*0_3/*.pth --method nn --arg "{'apply(stdev(ipTTL),forward)':((0,180.3),('abs','abs')),'apply(mean(ipTTL),forward)':((0,255),('abs','abs'))}" --nData -1 &

	$PYTHON -u learn.py --dataroot CAIA_backdoor_17.csv --backdoor --function $plot --net runs/mlp17/bd/*0_3/*.pth --method nn --arg "{'apply(stdev(ipTTL),forward)':((0,5),('abs','abs'))}" --nData -1 &
	$PYTHON -u learn.py --dataroot CAIA_backdoor_17.csv --backdoor --function $plot --net runs/mlp17/bd/*0_3/*.pth --method nn --arg "{'apply(stdev(ipTTL),forward)':((0,180.3),('abs','abs')),'apply(mean(ipTTL),forward)':((0,255),('abs','abs'))}" --nData -1 &
done >results/pdpale_bd.txt  # output is going to be quite messed up

wait 

for plot in pdp ale; do
	$PYTHON -u learn.py --dataroot CAIA_backdoor_15.csv --function $plot --net runs/rf15/non-bd/*0_3.* --method rf --arg "{'apply(stdev(ipTTL),forward)':((0,05),('abs','abs'))}" --nData -1 &
	$PYTHON -u learn.py --dataroot CAIA_backdoor_15.csv --function $plot --net runs/rf15/non-bd/*0_3.* --method rf --arg "{'apply(stdev(ipTTL),forward)':((0,180.3),('abs','abs')),'apply(mean(ipTTL),forward)':((0,255),('abs','abs'))}" --nData -1 &

	$PYTHON -u learn.py --dataroot CAIA_backdoor_17.csv --function $plot --net runs/rf17/non-bd/*0_3.* --method rf --arg "{'apply(stdev(ipTTL),forward)':((0,5),('abs','abs'))}" --nData -1 &
	$PYTHON -u learn.py --dataroot CAIA_backdoor_17.csv --function $plot --net runs/rf17/non-bd/*0_3.* --method rf --arg "{'apply(stdev(ipTTL),forward)':((0,180.3),('abs','abs')),'apply(mean(ipTTL),forward)':((0,255),('abs','abs'))}" --nData -1 &

	$PYTHON -u learn.py --dataroot CAIA_backdoor_15.csv --function $plot --net runs/mlp15/non-bd/*0_3/*.pth --method nn --arg "{'apply(stdev(ipTTL),forward)':((0,05),('abs','abs'))}" --nData -1 &
	$PYTHON -u learn.py --dataroot CAIA_backdoor_15.csv --function $plot --net runs/mlp15/non-bd/*0_3/*.pth --method nn --arg "{'apply(stdev(ipTTL),forward)':((0,180.3),('abs','abs')),'apply(mean(ipTTL),forward)':((0,255),('abs','abs'))}" --nData -1 &

	$PYTHON -u learn.py --dataroot CAIA_backdoor_17.csv --function $plot --net runs/mlp17/non-bd/*0_3/*.pth --method nn --arg "{'apply(stdev(ipTTL),forward)':((0,5),('abs','abs'))}" --nData -1 &
	$PYTHON -u learn.py --dataroot CAIA_backdoor_17.csv --function $plot --net runs/mlp17/non-bd/*0_3/*.pth --method nn --arg "{'apply(stdev(ipTTL),forward)':((0,180.3),('abs','abs')),'apply(mean(ipTTL),forward)':((0,255),('abs','abs'))}" --nData -1 &
done >results/pdpale_non-bd.txt  # output is going to be quite messed up

wait

 # PRUNING
valSizes='0.01 0.02 0.05 0.1 0.2 0.5 1'
for ds in 15 17; do
	for i in 0 1 2; do
		for valSize in $valSizes; do
			$PYTHON -u learn.py --dataroot CAIA_backdoor_${ds}.csv --backdoor --function prune_backdoor --pruneOnlyHarmless --net runs/rf${ds}/bd/*${i}_3.* --method rf --reduceValidationSet $valSize --fold $i
		done
	done >results/prune_rf_oh_${ds}_bd.txt &

	for i in 0 1 2; do
		for valSize in $valSizes; do
			$PYTHON -u learn.py --dataroot CAIA_backdoor_${ds}.csv --backdoor --function prune_backdoor --pruneOnlyHarmless --depth --net runs/rf${ds}/bd/*${i}_3.* --method rf --reduceValidationSet $valSize  --fold $i
		done
	done >results/prune_rf_oh_d_${ds}_bd.txt &

	for i in 0 1 2; do
		for valSize in $valSizes; do
			$PYTHON -u learn.py --dataroot CAIA_backdoor_${ds}.csv --backdoor --function prune_backdoor --correlation --net runs/mlp${ds}/bd/*${i}_3/*.pth --method nn --reduceValidationSet $valSize  --fold $i
		done
	done >results/prune_nn_${ds}_bd.txt &

	for i in 0 1 2; do
		for valSize in $valSizes; do
			$PYTHON -u learn.py --dataroot CAIA_backdoor_${ds}.csv --backdoor --function prune_backdoor --correlation --takeSignOfActivation --net runs/mlp${ds}/bd/*${i}_3/*.pth --method nn --reduceValidationSet $valSize  --fold $i
		done
	done >results/prune_nn_soa_${ds}_bd.txt &
done

wait

# FINE-TUNING
for ds in 15 17; do
	for i in 0 1 2; do
		$PYTHON -u learn.py --dataroot CAIA_backdoor_${ds}.csv --backdoor --function finetune --method nn --net runs/mlp${ds}/bd/*${i}_3/*.pth --fold $i >results/finetune_${ds}_$i.txt &
	done
done
