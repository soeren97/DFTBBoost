from setuptools import setup, find_packages

# Read the contents of the requirements file
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='DFTBBoost',
    version='0.1.0',
    author='Søren Langkidle',
    author_email='soeren@langkilde.com',
    packages=find_packages(),
    install_requires=requirements,
    dependency_links=["./source"],
    extras_require={
        'GPU': [
            'torch @ https://download.pytorch.org/whl/cu118'
        ],
        'CPU': [
            'torch'
        ]
    },
    editables=['./source']
)
