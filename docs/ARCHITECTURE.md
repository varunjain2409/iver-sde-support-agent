# ARCHITECTURE - iver-sde-support-agent

## Overview

This document is part of the documentation set for `iver-sde-support-agent`.

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Development Workflow](#development-workflow)

## Architecture

| Layer            | Responsibility                                  |
|------------------|-------------------------------------------------|
| Presentation    | UI layer (templates / views)                    |
| Application     | Request handling, validation, business rules    |
| Domain / Logic   | Core algorithms and business behaviour          |
| Data             | Persistence layer (database / file storage)     |

### Key Design Decisions

1. **Separation of concerns** - each layer only depends on the one below it.
2. **Configuration via environment** - secrets and endpoints are never hard-coded.
3. **Testability** - business logic is kept free of framework dependencies.

## Development Workflow

1. Fork the repository and create a feature branch.
2. Make your changes, keeping the existing code style.
3. Run the test suite locally before opening a pull request.
4. Open a PR describing what changed and why.

## Contributing

Contributions are welcome. For major changes, please open an issue first to
discuss the approach.

## License

Released under the [MIT License](../LICENSE).