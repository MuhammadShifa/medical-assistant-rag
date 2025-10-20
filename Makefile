# medical-assistant-rag - rag_app
# Author: Muhammad Shifa
# Description: Essential automation for development workflow

.PHONY: help setup check-prerequisites setup-env conda-create conda-install conda-remove conda-info
# Default target
.DEFAULT_GOAL := help

# Python and environment settings
PYTHON := python3
CONDA := conda
CONDA_ENV := rag_env
PROJECT_NAME := medical-assistant-rag

# Directories
SRC_DIR := src

# Colors for output
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[1;33m
BLUE := \033[0;34m
NC := \033[0m

help: ## Show this help message
	@echo "Medical Assitant Rag - Available Commands:"
	@echo "============================================="
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

# Environment Setup
setup: ## Complete environment setup (prerequisites + env + conda)
	@echo -e "$(GREEN)[$(shell date +'%Y-%m-%d %H:%M:%S')]$(NC) Starting complete environment setup..."
	$(MAKE) check-prerequisites

	$(MAKE) conda-create
	@echo ""
	@echo "========================================"
	@echo "  Environment Setup Complete!"
	@echo "========================================"
	@echo "Activate the environment:"
	@echo "   conda activate $(CONDA_ENV)"

# Prerequisites check
check-prerequisites: ## Check if all required tools are installed
	@echo -e "$(GREEN)[$(shell date +'%Y-%m-%d %H:%M:%S')]$(NC) Checking prerequisites..."
	@command -v python >/dev/null 2>&1 || { echo -e "$(RED)[$(shell date +'%Y-%m-%d %H:%M:%S')] ERROR:$(NC) Python is not installed"; exit 1; }
	@command -v conda >/dev/null 2>&1 || { echo -e "$(RED)[$(shell date +'%Y-%m-%d %H:%M:%S')] ERROR:$(NC) Conda is not installed"; exit 1; }
	@command -v git >/dev/null 2>&1 || { echo -e "$(RED)[$(shell date +'%Y-%m-%d %H:%M:%S')] ERROR:$(NC) Git is not installed"; exit 1; }
	@echo -e "$(GREEN)[$(shell date +'%Y-%m-%d %H:%M:%S')]$(NC) Prerequisites check completed"

# Conda Environment Management
# Create conda environment
conda-create:
	@echo "Creating conda environment: $(CONDA_ENV)..."
	$(CONDA) create -n $(CONDA_ENV) python=3.10 -y
	@echo "Conda environment created! Activate with: conda activate $(CONDA_ENV)"

# Create conda env and install production requirements
conda-install: conda-create
	@echo "Installing production requirements in conda environment..."
	$(CONDA) run -n $(CONDA_ENV) pip install --upgrade pip
	$(CONDA) run -n $(CONDA_ENV) pip install -r requirements.txt
	@echo "Conda environment ready with production requirements!"
	
# Remove conda environment
conda-remove:
	@echo "Removing conda environment: $(CONDA_ENV)..."
	$(CONDA) env remove -n $(CONDA_ENV) -y
	@echo "Conda environment removed!"


# Code Quality
lint: ## Run all linting checks
	@echo "Running pylint..."
	pylint $(SRC_DIR)

format: ## Format code with black and isort
	@echo "Formatting code..."
	black $(SRC_DIR) $(TEST_DIR)
	isort $(SRC_DIR) $(TEST_DIR)
	@echo "Code formatting completed!"

format-check: ## Check if code needs formatting
	@echo "Checking code formatting..."
	black --check $(SRC_DIR) $(TEST_DIR)
	isort --check-only $(SRC_DIR) $(TEST_DIR)

# Project Information
info: ## Show project information
	@echo "Safety Risk Assessment"
	@echo "==============================="
	@echo "Python Version: $$($(PYTHON) --version)"
	@echo "Conda Version: $$($(CONDA) --version)"
	@echo "Current Environment: $$(conda info --envs | grep '*' | awk '{print $$1}' || echo 'Not in conda env')"
	@echo "Project Environment: $(CONDA_ENV)"
	@echo "Project Structure:"
	@tree -L 2 -I '__pycache__|*.pyc|.git' || ls -la
