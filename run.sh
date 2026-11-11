#!/usr/bin/env bash

eval "$(conda shell.bash hook)"
conda activate autismCNN_env

export PROJECT_DIR=$(pwd)
echo "Project directory set to $PROJECT_DIR"
export ORIENTATION="sagittal"  # default orientation
export SEED=42  # default seed

while getopts "eo:mts:h" flag; do
 case $flag in
   e) # Handle the -e flag
   # Extract the zip files and process the data 
   echo "Extracting zip files..."
   python3 src/features/transformers.py
   ;;
   o) # Handle the -o flag
   if [[ "$OPTARG" != "axial" && "$OPTARG" != "sagittal" && "$OPTARG" != "coronal" ]]; then
     echo "Invalid orientation: $OPTARG. Please choose from 'axial', 'sagittal', or 'coronal'."
     exit 1
   fi
   export ORIENTATION=$OPTARG
   ;;
   m) # Handle the -m flag
   echo "Making manifest..."
   python3 src/features/preprocessing.py -o $ORIENTATION
   ;;
   t) # Handle the -t flag
   export USE_SMOKE=1
   ;;
   s) # Handle the -s flag
   export SEED=$OPTARG
   ;;
   h) # Handle the -h flag
   echo "Usage: $0 [-e] [-o orientation] [-m] [-t] [-s seed] [-h]"
   echo "  -e: Extract zip files and process the data"
   echo "  -o: Set scan orientation (axial, sagittal, coronal)"
   echo "  -m: Make manifest"
   echo "  -t: Use smoke test mode (tiny run)"
   echo "  -s: Set random seed (integer value)"
   echo "  -h: Display this help message"
   exit 0
   ;;
   *) # Handle unknown flags
   echo "Unknown option: -$OPTARG" >&2
   echo "Usage: $0 [-e] [-o orientation] [-m] [-t] [-s seed] [-h]"
   exit 1
   ;;
 esac
done

echo "Orientation set to '$ORIENTATION'"
echo "Seed set to '$SEED'"

if [ -z "$USE_SMOKE" ]; then
  echo "Doing a full run..."
  python3 src/train_baseline.py; else
  echo "Doing a test run..."
  python3 src/train_baseline.py --smoke;
fi