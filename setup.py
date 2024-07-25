from setuptools import find_packages
from setuptools import setup


setup(
    name="causal_actors",
    author="Siddhesh Acharekar",
    install_requires=[
        'numpy>=1.17',
        'pandas>=0.25.3',
        'pyro-ppl==0.4.1',
        'torch==2.2.0',
        'torchvision==0.4.2',
    ],
    python_requires='>=3.6',
)
