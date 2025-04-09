from setuptools import setup, find_packages

# Read the contents of README.md
with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

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
    author_email="subaashnair@gmail.com",  # Replace with your email
    description="A library for evaluating scikit-learn regression models with comprehensive metrics",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/SubaashNair/sklearnMetrics",
    project_urls={
        "Bug Tracker": "https://github.com/SubaashNair/sklearnMetrics/issues",
        "Documentation": "https://github.com/SubaashNair/sklearnMetrics#readme",
        "Source Code": "https://github.com/SubaashNair/sklearnMetrics",
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.7",
    keywords="scikit-learn, machine learning, metrics, evaluation, regression",
) 