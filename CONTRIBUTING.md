## Contributing to AutoDist

**Thanks for taking the time to contribute!**

Refer to the following guidelines to contribute new functionality or bug fixes:

1. [Install](docs/usage/tutorials/installation.md) from source under development mode.
2. Use Prospector to lint the Python code: `prospector autodist`.
3. Add unit and/or integration tests for any new code you write.
4. Run unit and or integration tests in both CPU and GPU environments: `cd tests && python3 -m pytest -s --run-integration .`
where `--run-integration` is optional when only running unit tests.
