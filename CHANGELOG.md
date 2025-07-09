# Changelog

All notable changes to Arc Runtime will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.5] - 2025-01-09

### Added
- Enhanced Kong Gateway integration with comprehensive TLS support
- Retry logic with exponential backoff for improved reliability
- Connectivity health checks for Kong endpoint validation
- New TelemetryConfig dataclass for advanced configuration options
- Support for Kong Konnect with authentication and routing

### Improved
- Restructured README using Diátaxis framework for better team onboarding
- Enhanced error handling with Kong-specific error messages
- Better telemetry client configuration with backward compatibility
- Updated all code examples to use gpt-4.1 model

## [0.1.4] - 2025-01-08

### Fixed
- Fixed protobuf compatibility issue with `runtime_version` import error
- Removed runtime_version dependency to support protobuf versions 3.20.0+
- Ensures compatibility with older protobuf installations

## [0.1.3] - 2025-01-08

### Fixed
- Added missing `httpx` dependency for MCP interceptor support
- Fixed import error when using the package from PyPI

## [0.1.2] - 2025-01-08

### Added
- Real gRPC integration with Arc Core server
- Protobuf definitions for telemetry events (`runtime/proto/telemetry.proto`)
- Streaming RPC support for batch telemetry transmission
- API key authentication for gRPC connections
- Proto generation script (`generate_proto.py`)

### Changed
- Updated telemetry client to use real gRPC streaming instead of mock implementation
- Moved `grpcio` and `protobuf` from optional to required dependencies
- Enhanced telemetry event format to match Arc Core protobuf schema
- Improved error handling for gRPC connection failures

### Fixed
- Telemetry event structure now properly includes all required fields
- Import paths in generated protobuf files use relative imports

## [0.1.1] - 2025-01-07

### Added
- Multi-agent support with comprehensive context tracking
- MCP (Model Context Protocol) interception
- LangGraph integration with custom StateGraph
- Enhanced telemetry with agent-specific traces
- Pipeline execution tracking
- Failure recording for training data

### Changed
- Improved OpenTelemetry integration
- Enhanced metrics collection

### Fixed
- Build warnings and packaging configuration
- PEP 561 compliance with py.typed marker

## [0.1.0] - 2024-12-24

### Added
- Initial release
- OpenAI SDK interception
- Pattern matching and fix application
- Basic telemetry and metrics
- Singleton Arc instance
- Zero-config initialization