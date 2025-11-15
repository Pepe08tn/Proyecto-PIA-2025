from setuptools import setup, find_packages

setup(
    name="mia_predictor",
    version="0.0.1",
    packages=find_packages(),
    install_requires=[
        "torch",
        "torchvision",
        "Pillow"
    ],
)
