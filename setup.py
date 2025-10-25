from setuptools import find_packages, setup

setup(
    name='hyperthymesia',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        "click>=8.1.0",
        "colorama>=0.4.6",
        "chromadb>=0.4.0",
        "sentence-transformers>=2.2.0",
        "pypdf2>=3.0.0",
        "python-docx>=0.8.11",
        "platformdirs>=3.0.0",
        "tqdm>=4.65.0",
        "PyPDF2>=3.0.0",
        "google-generativeai>=0.1.0",
        "openai>=1.0.0",
        "anthropic>=0.1.0",
        "requests>=2.31.0",
        "llama-cpp-python>=0.2.0",
        "mlx-lm>=0.1.0",
    ],
    entry_points={
        'console_scripts': [
            'hyperthymesia=cli.main:cli',
            'mesia=cli.repl:start_repl',  # Add this!
        ],
    },
)