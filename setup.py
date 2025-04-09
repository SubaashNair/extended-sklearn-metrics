from setuptools import setup, find_packages

setup(
    name="sklearnMetrics",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pandas>=2.0.0",
        "scikit-learn>=1.0.0",
        "numpy>=1.20.0",
    ],
    author="sklearnMetrics Contributors",
    description="A library for evaluating scikit-learn regression models with comprehensive metrics",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/sklearnMetrics",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
) 