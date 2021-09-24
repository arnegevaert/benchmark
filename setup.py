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
        "tensorboard==2.4.1",
        "krippendorff==0.4.0",
        "opencv-python==4.2.0.34",
        "dash==1.18.1",
        "dash-bootstrap-components==0.10.7",
        "dash-core-components==1.14.1",
        "dash-html-components==1.1.1",
        "dash-renderer==1.8.3",
        "dash-table==4.11.1",
        "scikit-learn==0.24.1"
    ]
)
