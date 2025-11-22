from setuptools import setup, find_packages

setup(
    name="neuroseal",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch",
        "transformers",
        "peft",
    ],
    entry_points={
        "console_scripts": [
            "neuroseal = neuroseal.cli:main",
        ],
    },
)

