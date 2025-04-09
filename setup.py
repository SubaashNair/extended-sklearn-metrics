from setuptools import setup, find_packages

# Read the contents of README.md
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="sklearn_metrics",
    version="0.1.3",
    author="Subashanan Nair",
    author_email="subashnair12@gmail.com",
    description="A Python package for enhanced model evaluation metrics",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/subashanannair/sklearn_metrics",
    packages=find_packages(),
    include_package_data=True,
    package_data={
        "sklearn_metrics": ["*.py"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0"
    ],
) 