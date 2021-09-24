import setuptools

setuptools.setup(
    name="attrbench",
    version="0.0.1",
    author="Arne Gevaert, Axel-Jan Rousseau",
    author_email="arne.gevaert@ugent.be, axeljan.rousseau@uhasselt.be",
    description="A benchmark for feature attribution techniques",
    url="https://github.com/zoeparman/benchmark",
    packages=setuptools.find_packages(),
    install_requires=[
        "torch>=1.5.0",
        "h5py==2.10.0",
        "tqdm==4.46.1",
        "tensorboard==2.4.1"
    ]
)
