##@ General

.PHONY: help
help: ## Display this help.
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m<target>\033[0m\n"} /^[a-zA-Z_0-9-]+:.*?##/ { printf "  \033[36m%-17s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)

##@ Development

.env:
	@test -f $@ || (printf "Error: File $@ not found!\nRun:\n    cp env_example .env\n\nThen edit $@ file.\n\n"; exit 1)

.PHONY: run
run: .env ## Run the application in Docker.
	docker compose up --detach --build

.PHONY: stop
stop: ## Stop the application in Docker.
	docker compose down

.PHONY: logs
logs: ## Show logs
	docker compose logs --follow
