.PHONY: install dev build

install:
	npm install

# Unified runtime - AGENT_MODE in .env decides whether you connect to local or remote agents
dev: install
	@echo "🚀 Starting SimuLab UI (mode determined by AGENT_MODE in .env)..."
	@if [ ! -f .env ]; then \
		echo "⚠️  .env not found, creating from template..."; \
		cp env.example .env; \
		echo "⚠️  Please review .env and set AGENT_MODE plus credentials"; \
	fi
	npm run dev

build: install
	npm run build

