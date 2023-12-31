#!/bin/bash

# Exist on error
set -e

# Get the current directory
current_dir=$(pwd)

# Loop through each child directory
for dir in "$current_dir"/*/; do
    # Check if the child directory has a pyproject.toml file
    if [ -f "$dir/pyproject.toml" ]; then
        # Change into the child directory
        cd "$dir"

        # # Conda
        # # ! conda activate doesn't work somehow
        last_dir=$(basename "$dir")
        conda_env_name=".venv_$last_dir"
        # conda create -n $conda_env_name python=3.11 -y
        # conda activate $conda_env_name

        # venv
        python -m venv ../$conda_env_name
        source ../$conda_env_name/bin/activate

        # Run 'install' command
        python -m pip install -e "."

        # Delete the `target_repo` directory
        rm -rf target_repo

        # Symbolic link `target_repo` to one directory up
        ln -s ../target_repo target_repo

        # Change back to the original directory
        cd "$current_dir"
    fi
done
