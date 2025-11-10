# Makefile for MyST site

# Environment name
ENV_NAME := myst_env

# Environment file
ENV_FILE := environment.yml

# Directories to clean
CLEAN_DIRS := _build figures audio

.PHONY: env html clean

# ----------------------------
# Create or update the environment
env:
	@echo "Creating/updating conda environment '$(ENV_NAME)'..."
	conda env create -f $(ENV_FILE) -n $(ENV_NAME) || conda env update -f $(ENV_FILE) -n $(ENV_NAME)
	@echo "Environment setup complete. To activate it: conda activate $(ENV_NAME)"

# ----------------------------
# Build HTML version of MyST site
html:
	@echo "Building HTML site..."
	myst build --html
	@echo "HTML build complete. You can view it locally in _build/html"

# ----------------------------
# Clean generated files
clean:
	@echo "Cleaning _build, figures, and audio folders..."
	rm -rf $(CLEAN_DIRS)
	@echo "Cleanup complete."
