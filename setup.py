import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="OptiClustTest",
    version="0.0.6",
    author="Shreyas Kera",
    author_email="shreykera7@gmail.com",
    description="Python implementation of various algorithms to find the optimal number of clusters",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/shreyas-bk/OptiClustTest",
    packages=['OptiClustTest'],
    license="MIT",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
