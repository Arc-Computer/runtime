#!/usr/bin/env python3
"""
Generate Python code from proto files
"""

import os
import sys
import subprocess


def generate_proto():
    """Generate Python code from proto files"""
    proto_dir = "runtime/proto"
    proto_file = "telemetry.proto"
    
    # Ensure we're in the project root
    if not os.path.exists(proto_dir):
        print(f"Error: {proto_dir} not found. Run this script from the project root.")
        return 1
    
    # Run protoc to generate Python code
    cmd = [
        sys.executable, "-m", "grpc_tools.protoc",
        f"-I{proto_dir}",
        f"--python_out={proto_dir}",
        f"--grpc_python_out={proto_dir}",
        f"{proto_dir}/{proto_file}"
    ]
    
    print(f"Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Error generating proto: {result.stderr}")
            return 1
        
        print("Proto generation successful!")
        
        # Fix imports in generated files to use relative imports
        fix_imports()
        
        return 0
    except Exception as e:
        print(f"Error: {e}")
        return 1


def fix_imports():
    """Fix imports in generated files to use relative imports"""
    proto_dir = "runtime/proto"
    
    # Fix imports in telemetry_pb2_grpc.py
    grpc_file = os.path.join(proto_dir, "telemetry_pb2_grpc.py")
    if os.path.exists(grpc_file):
        with open(grpc_file, 'r') as f:
            content = f.read()
        
        # Replace absolute import with relative import
        content = content.replace(
            "import telemetry_pb2 as telemetry__pb2",
            "from . import telemetry_pb2 as telemetry__pb2"
        )
        
        with open(grpc_file, 'w') as f:
            f.write(content)
        
        print(f"Fixed imports in {grpc_file}")


if __name__ == "__main__":
    sys.exit(generate_proto())