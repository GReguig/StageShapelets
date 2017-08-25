#!/bin/bash

if [ $# -eq 0 ]
	then
		for file in /home/reguig/datasets/UCR/*
		do
			echo "Dataset $file"
			python /home/reguig/LearningTimeSeriesShapelets/deepCNNExperiments.py $file
		done
	exit 
fi

for file in "$@"
do
	echo "Dataset $file"
	python /home/reguig/LearningTimeSeriesShapelets/deepCNNExperiments.py $file
	
done
