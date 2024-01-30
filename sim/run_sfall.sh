#!/usr/bin/env bash

# Source the CCP4 setup script (modify the path to match your CCP4 installation)
source /path/to/ccp4/bin/ccp4.setup-sh

# Check if PDB file and space group are provided
if [ -z "$1" ] || [ -z "$2" ]; then
    echo "PDB file and space group are required."
    exit 1
fi

# Assign parameters to variables
PDB_FILE="$1"
SPACEGROUP="$2"
RESOLUTION=${3:-2.0} # Default resolution is 2.0 if not provided
POINTGROUP=${4:-""} # Optional point group

# Base output directory
BASE_OUTDIR="sfall_output"

# Define output directories within the base directory
TXT_OUTDIR="$BASE_OUTDIR/out_$SPACEGROUP"
MTZ_OUTDIR="$BASE_OUTDIR/mtz_$SPACEGROUP"
HKL_OUTDIR="$BASE_OUTDIR/data_$SPACEGROUP"

# Create base and output directories if they do not exist
mkdir -p "$BASE_OUTDIR"
mkdir -p "$TXT_OUTDIR"
mkdir -p "$MTZ_OUTDIR"
mkdir -p "$HKL_OUTDIR"

# Extract base name from PDB file
BASENAME=$(basename "$PDB_FILE" .pdb)
MTZ_FILE="$MTZ_OUTDIR/${BASENAME}.mtz"
TXT_FILE="$TXT_OUTDIR/${BASENAME}.txt"
HKL_FILE="$HKL_OUTDIR/${BASENAME}.hkl"

echo "-----------------------------------"
echo "Processing $PDB_FILE"

# Run sfall
echo "Running sfall for $PDB_FILE..."
sfall XYZIN "$PDB_FILE" HKLOUT "$HKL_FILE" <<EOF
MODE SFCALC XYZIN
RESOLUTION $RESOLUTION
FORM NGAUSS 5
SYMM $SPACEGROUP
END
EOF
echo "Finished sfall for $PDB_FILE."

# Further processing steps can be added here, such as running mtz2various

echo "Processed $PDB_FILE"
echo "-----------------------------------"
