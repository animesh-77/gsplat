#!/bin/bash
# Activate the desired conda environment
conda activate gaussian_splatting
# Run gsplat on a list of input files
# first define some arguments
directory="$1" 
port="8080"
count=0
while [[ $count -lt 10 ]]; do
    if ! nc -z localhost $port; then
        break
    fi
    ((count++))
    ((port++))
done
echo "Using port: $port"
# first argument passed

python examples/simple_viewer.py --cwd $directory --port $port
