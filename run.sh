#!/usr/bin/env bash
set -e  # stop at the first failing step

eval "$(conda shell.bash hook)"
conda activate autismCNN_env

# --------------------------------------------------------------------------- #
# Settings
# --------------------------------------------------------------------------- #
export PROJECT_DIR=$(cd "$(dirname "$0")" && pwd)
echo "Project directory set to $PROJECT_DIR"
export ORIENTATION="sagittal"  # default orientation
export SEED=42  # default seed
export TARGET_SHAPE="96,96,96"  # default image size (3 integers)

# --------------------------------------------------------------------------- #
# Flags
# --------------------------------------------------------------------------- #
while getopts "o:s:p:emth" flag; do
 case $flag in
   o) # Handle the -o flag
   if [[ "$OPTARG" != "axial" && "$OPTARG" != "sagittal" && "$OPTARG" != "coronal" ]]; then
     echo "Invalid orientation: $OPTARG. Please choose from 'axial', 'sagittal', or 'coronal'."
     exit 1
   fi
   export ORIENTATION=$OPTARG
   ;;
   s) # Handle the -s flag
   export SEED=$OPTARG
   ;;
   p) # Handle the -z flag
   export TARGET_SHAPE=$OPTARG
   ;;
   e) # Handle the -e flag
   DO_EXTRACT=1
   ;;
   m) # Handle the -m flag
   DO_MANIFEST=1
   ;;
   t) # Handle the -t flag
   export USE_SMOKE=1
   ;;
   h) # Handle the -h flag
   echo "Usage: $0 [-o orientation] [-s seed] [-p shape] [-e] [-m] [-t] [-h]"
   echo "  -o: Set scan orientation (axial, sagittal, coronal)"
   echo "  -s: Set random seed (integer value)"
   echo "  -p: Set image size as 3 integers (e.g. '96,96,96')"
   echo "  -e: Extract zip files and process the data"
   echo "  -m: Make manifest"
   echo "  -t: Use smoke test mode (tiny run)"
   echo "  -h: Display this help message"
   exit 0
   ;;
   *) # Handle unknown flags
   echo "Unknown option: -$OPTARG" >&2
   echo "Usage: $0 [-e] [-o orientation] [-m] [-t] [-s seed] [-z shape] [-h]"
   exit 1
   ;;
 esac
done

# --------------------------------------------------------------------------- #
# Running Scripts
# --------------------------------------------------------------------------- #
echo "Orientation set to '$ORIENTATION'"
echo "Seed set to '$SEED'"
echo "Image size set to '$TARGET_SHAPE'"

# run the steps after all flags are read, so the order of the flags doesn't matter
if [ -n "$DO_EXTRACT" ]; then
  # Extract the zip files and process the data
  echo "Extracting zip files..."
  python3 src/features/transformers.py --o $ORIENTATION --shape $TARGET_SHAPE
fi

if [ -n "$DO_MANIFEST" ]; then
  echo "Making manifest..."
  python3 src/features/preprocessing.py --o $ORIENTATION --shape $TARGET_SHAPE
fi

if [ -z "$USE_SMOKE" ]; then
  echo "Doing a full run..."
  python3 src/train_baseline.py --o $ORIENTATION --shape $TARGET_SHAPE --seed $SEED; else
  echo "Doing a test run..."
  python3 src/train_baseline.py --smoke --o $ORIENTATION --shape $TARGET_SHAPE --seed $SEED;
fi