from setuptools import setup, find_packages

setup(
    name="sac_traffic_control",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "gym==0.26.2",
        "torch",
        "numpy==1.26.2",
        "matplotlib==3.8.2",
        "pandas==2.1.4",
        "scipy==1.11.4",
        "tensorboard==2.14.0",
    ],
    extras_require={
        "rl": [
            "stable-baselines3==2.2.1",
            "gymnasium==0.29.1",
        ]
    },
    include_package_data=True,
)
