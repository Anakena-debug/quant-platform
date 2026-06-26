.PHONY: help setup test sp500 intraday fundamentals fund-test catalog validate query clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*##' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

setup: ## Install dependencies
	uv pip install yfinance polars duckdb pyarrow

test: ## Ingest 10 test tickers (daily + fundamentals) — smoke test
	python3 scripts/ingest_yfinance.py --mode daily --scope test
	python3 scripts/ingest_fundamentals.py --scope test
	python3 scripts/build_catalog.py
	python3 scripts/validate_data.py

sp500: ## Ingest full S&P 500 daily bars
	python3 scripts/ingest_yfinance.py --mode daily --scope sp500
	python3 scripts/build_catalog.py
	python3 scripts/validate_data.py

intraday: ## Ingest 1m bars for test universe (last 7 days)
	python3 scripts/ingest_yfinance.py --mode intraday --interval 1m --scope test
	python3 scripts/build_catalog.py

fundamentals: ## Ingest fundamentals for full universe
	python3 scripts/ingest_fundamentals.py --scope file --file us_large_cap_tickers.txt
	python3 scripts/build_catalog.py

fund-test: ## Ingest fundamentals for 10 test tickers only
	python3 scripts/ingest_fundamentals.py --scope test
	python3 scripts/build_catalog.py

catalog: ## Rebuild DuckDB catalog from existing parquet files
	python3 scripts/build_catalog.py

validate: ## Run data quality checks
	python3 scripts/validate_data.py

query: ## Open interactive DuckDB shell
	duckdb catalog/quantdata.duckdb

clean: ## Remove all ingested data (careful!)
	@echo "This will delete ALL parquet files and the catalog."
	@read -p "Are you sure? [y/N] " confirm && [ "$$confirm" = "y" ] && \
		rm -rf raw/yfinance/daily/*.parquet raw/yfinance/intraday/ raw/yfinance/fundamentals/ catalog/quantdata.duckdb && \
		echo "Cleaned." || echo "Aborted."
