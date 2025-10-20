from setuptools import find_packages, setup

setup(
    name="hyperthymesia-cli",  # This is just the distribution name
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "click>=8.1.0",
        "colorama>=0.4.6",
    ],
    entry_points={
        "console_scripts": [
            # Fixed: use hyperthymesia_cli (the actual directory name)
            "hyperthymesia=hyperthymesia_cli.cli.main:cli",
        ],
    },
)
