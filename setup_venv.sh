#!/usr/bin/env bash
# Sets up a local Python virtual environment for development/testing
# without Docker. Use this if you want to run/debug the pipeline directly.
#
# Usage:
#   chmod +x setup_venv.sh
#   ./setup_venv.sh
#   source venv/bin/activate

set -e

VENV_DIR="venv"
PYTHON_BIN="${PYTHON_BIN:-python3}"

echo "Creating virtual environment in ./${VENV_DIR}..."
$PYTHON_BIN -m venv "$VENV_DIR"

echo "Activating virtual environment..."
source "$VENV_DIR/bin/activate"

echo "Upgrading pip..."
pip install --upgrade pip

echo "Installing requirements..."
pip install -r requirements.txt

echo ""
echo "Done. Virtual environment ready."
echo "Activate it with:  source $VENV_DIR/bin/activate"
echo ""
echo "Before running the pipeline locally, make sure Ollama is running:"
echo "  ollama serve &"
echo "  ollama pull mistral"
echo ""
echo "Then run:"
echo "  export OLLAMA_HOST=http://localhost:11434"
echo "  export CSV_PATH=./data/modanisa_products.csv"
echo "  export QDRANT_PATH=./data/qdrant_storage"
echo "  python app/outfit_pipeline.py --index"
echo "  python app/outfit_pipeline.py --image data/sample.jpg --text 'I want a blouse'"
