from distutils.core import setup


def read_requirements():
    with open('requirements.txt', 'r') as file:
        return [line.strip() for line in file.readlines() if not line.startswith('#')]


setup(
    name = 'ChemSpaceAL',
    packages = ['ChemSpaceAL'],
    version = '1.0.2',
    description = 'ChemSpaceAL Python package: an efficient active learning methodology applied to protein-specific molecular generation',
    install_requires=read_requirements(),
    author = 'Gregory W. Kyro, Anton Morgunov & Rafael I. Brent',
    author_email = 'gregory.kyro@yale.edu',
    url = 'https://github.com/gregory-kyro/ChemSpaceAL/tree/main',
    download_url = 'https://github.com/gregory-kyro/ChemSpaceAL/archive/refs/tags/v1.0.2.tar.gz',
    keywords = ['active learning', 'artificial intelligence', 'deep learning', 'machine learning', 'molecular generation', 'drug discovery'],
    classifiers = []
)
