# Development Scripts

This directory contains scripts used for Arc Runtime development.

## generate_proto.py

Regenerates Python gRPC code from the protobuf definitions.

Usage:
```bash
# From the project root
python scripts/generate_proto.py
```

This is only needed when:
- The `runtime/proto/telemetry.proto` file is modified
- You want to update to a newer version of protobuf/grpcio-tools

The generated files are already included in the package, so end users don't need to run this.