#!/bin/bash
# Script to set up and run Arc Core gRPC server for testing

echo "=== Arc Core gRPC Server Setup ==="
echo

# Check if server directory exists
if [ -d "arc-core-test" ]; then
    echo "Arc Core test directory already exists. Using existing installation."
    cd arc-core-test/Arc-Failure-Taxonomy/arc-core
else
    echo "Setting up Arc Core gRPC server..."
    mkdir -p arc-core-test
    cd arc-core-test
    
    # Clone the repository
    echo "Cloning Arc Core repository..."
    git clone https://github.com/Arc-Computer/Arc-Failure-Taxonomy.git
    cd Arc-Failure-Taxonomy
    git checkout gRPC
    cd arc-core
    
    # Install dependencies
    echo "Installing dependencies..."
    pip install grpcio protobuf
    
    # Generate gRPC code
    echo "Generating gRPC code..."
    chmod +x generate_grpc_code.sh
    ./generate_grpc_code.sh
fi

echo
echo "Starting Arc Core gRPC server on port 50051..."
echo "Press Ctrl+C to stop the server."
echo
python -m arc.grpc.server