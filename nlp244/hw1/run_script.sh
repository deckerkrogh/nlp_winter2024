#!/bin/bash

# Step 0. Make sure we're only using 1 gpu, just to save resources
export CUDA_VISIBLE_DEVICES=0

# Step 1. Make a virtual environment
python -m venv .venv
source .venv/bin/activate

# Step 2. Install any libraries specified in the requirements.txt file
pip install -r requirements.txt

# Step 3. Run main.py to train your best model on the file specified by --data
python main.py --train --data "../hw1_train.csv" --save_model "./joint_trained_model.pt"

# Step 4. Run main.py to test the newly trained model on the test data
python main.py --test --data "../hw1_test.csv" --model_path "./joint_trained_model.pt" --output "./preds.csv"

# Step 5. Compute metrics (don't worry about this step - this will succeed as long as preds.csv is in the right format)
python ../metrics.py --gold "../gold.csv" --predictions "./preds.csv"
