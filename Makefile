.DEFAULT_GOAL := help

.PHONY: help download audit split train train-all evaluate gradcam infer report demo benchmark all

help: ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

download: ## Download BTXRD dataset from figshare
	python scripts/download.py --config configs/default.yaml

audit: ## Profile dataset quality and generate audit report
	python scripts/audit.py --config configs/default.yaml

split: ## Generate train/val/test split manifests
	python scripts/split.py --config configs/default.yaml

train: ## Train EfficientNet-B0 classifier on BTXRD dataset
	python scripts/train.py --config configs/default.yaml

train-all: ## Train on both stratified and center-holdout splits
	python scripts/train.py --config configs/default.yaml --override training.split_strategy=stratified
	python scripts/train.py --config configs/default.yaml --override training.split_strategy=center

evaluate: ## Evaluate trained model on both split strategies
	python scripts/eval.py --config configs/default.yaml

gradcam: ## Generate Grad-CAM heatmaps for selected examples
	python scripts/gradcam.py --config configs/default.yaml

infer: ## Run single-image or batch inference with Grad-CAM overlay
	python scripts/infer.py --config configs/default.yaml

report: ## Verify documentation is complete (model card + PoC report)
	@test -f docs/model_card.md || (echo "ERROR: docs/model_card.md not found" && exit 1)
	@test -f docs/poc_report.md || (echo "ERROR: docs/poc_report.md not found" && exit 1)
	@echo "Documentation verified:"
	@echo "  - docs/model_card.md ($$(wc -l < docs/model_card.md) lines)"
	@echo "  - docs/poc_report.md ($$(wc -l < docs/poc_report.md) lines)"
	@echo "  - docs/data_audit_report.md"
	@echo "  - docs/dataset_spec.md"

demo: ## Launch Streamlit demo app
	streamlit run app/app.py

benchmark: ## Run model improvement benchmark (all experiments)
	python scripts/benchmark.py --config configs/default.yaml

all: download audit split train-all evaluate gradcam report ## Run full pipeline end-to-end
