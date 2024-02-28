#!/bin/bash
source activate SamplingPaper || { echo "Failed to activate Conda environment"; exit 1; }

tests="config/nursery.ini config/ASCIncome_USA_2018_binned_imbalanced_16645.ini config/ASCIncome_USA_2018_binned_imbalanced_1664500.ini config/ACSIncome_USA_2018_binned_imbalanced_16645_acc_metric.ini config/cmc.ini"
tests="config/income_binned_USA_1664500.ini"

cd python

for test in  $tests; do
    python arx_consumer.py $test 4
done

cd ..

conda deactivate