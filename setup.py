from setuptools import setup, find_packages

setup(
    name="conjunctionreservoir",
    version="0.1.0",
    author="Antti Luode",
    description="Sentence-windowed conjunction retrieval grounded in auditory neuroscience",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=["numpy>=1.21"],
    extras_require={
        "dev": ["pytest"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
