#!/bin/bash
source activate SamplingPaper || { echo "Failed to activate Conda environment"; exit 1; }

tests="src/config/nursery.properties src/config/ASCIncome_USA_2018_binned_imbalanced_16645.properties src/config/ASCIncome_USA_2018_binned_imbalanced_1664500.properties src/config/ACSIncome_USA_2018_binned_imbalanced_16645_acc_metric.properties src/config/cmc.properties"
tests="src/config/ASCIncome_USA_2018_binned_imbalanced_1664500.properties"

pythonEnvPath="/home/msec/.conda/envs/SamplingPaper/bin/python"

for test in  $tests; do
    java -jar ./out/artifacts/thesis_project_jar/thesis-project.jar $test 4 $pythonEnvPath
done


conda deactivate