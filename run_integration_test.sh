#!/bin/bash

echo "Starting FastAPI server..."
./bin/dev --port 8000 &
SERVER_PID=$!
sleep 5
echo "Running integration tests..."
poetry run pytest integration_test/test_app.py