import setuptools

setuptools.setup(
    name="attrbench",
    version="0.0.1",
    author="Arne Gevaert, Axel-Jan Rousseau",
    author_email="arne.gevaert@ugent.be, axeljan.rousseau@uhasselt.be",
    description="An benchmark for feature attribution techniques and metrics",
    url="https://github.com/zoeparman/benchmark",
    packages=setuptools.find_packages(),
    install_requires=[
        "captum>=0.2.0",
        "torch>=1.5.0",
        "torchvision>=0.6.0",
        "tqdm>=4.46.0",
        "imgaug>=0.4.0",
        "sklearn"
    ]
)
