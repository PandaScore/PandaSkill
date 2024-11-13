from setuptools import setup, find_packages

setup(
    name="pandaskill",
    version="0.1.0",
    packages=find_packages(exclude=["tests*"]),
    
    install_requires=[],
    
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    
    python_requires=">=3.12.7",
)