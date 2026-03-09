APP_NAME := transcripter

build: ## Build for current platform (macOS/Linux)
	go build -o $(APP_NAME) .

build-arm7: ## Build for old ARM Synology (DS214, DS216, DS218 etc.)
	GOOS=linux GOARCH=arm GOARM=7 go build -o $(APP_NAME)-linux-armv7 .

build-arm64: ## Build for newer ARM64 Synology (DS220+, DS720+ etc.)
	GOOS=linux GOARCH=arm64 go build -o $(APP_NAME)-linux-arm64 .

build-amd64: ## Build for Intel/AMD Synology (DS918+, DS1621+ etc.)
	GOOS=linux GOARCH=amd64 go build -o $(APP_NAME)-linux-amd64 .

build-all: build-arm7 build-arm64 build-amd64 ## Build for all Synology architectures

clean: ## Remove all built binaries
	rm -f $(APP_NAME) $(APP_NAME)-linux-*

help: ## Show this help
	@awk 'BEGIN {FS = ":.*## "} /^[a-zA-Z0-9_-]+:.*## / {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST) | sort

.PHONY: build build-arm7 build-arm64 build-amd64 build-all clean help
