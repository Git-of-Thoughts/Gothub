set -e

python -m pip install \
    -e "./gots[dev,test]" \
    -e "./gothub[dev,test]" \
    -e "./gothub_server[dev,test]"

# Set up pre-commit hooks
pre-commit install
