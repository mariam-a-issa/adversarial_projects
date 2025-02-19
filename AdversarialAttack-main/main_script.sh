for MODEL in "neurosymbolic" #"dnn" "hdc"
do
  echo "Running for $MODEL..."
  for DATASET in "fmnist" #TODO:add other datasets
  do
    for ATTACK in "original" "fgsm"  "genetic" # "jsma" "kernel" "deepfool"
    do
      echo "Running for $MODEL/$DATASET/$ATTACK..."
      python model_framework.py $MODEL $DATASET $ATTACK
      echo "Complete"
    done
  done
done

