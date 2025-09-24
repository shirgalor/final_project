from setuptools import setup, Extension

module = Extension("mykmeanspp", sources=["kmeansmodule.c"])

setup(
    name="kmeans",
    version="1.0",
    description="Python interface for k-means clustering",
    ext_modules=[module],
)