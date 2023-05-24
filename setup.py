from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    README = fh.read()

setup(
    name='energypredictor',
    version='0.1.0',
    long_description=README,
    long_description_content_type="text/markdown",
    description='Weather based energy predictor',
    author='resonanz.io',
    author_email='info@resonanz.io',
    url=
    'https://gitlab.com/resonanz/private/projects/ncg/balancing-energy-demand-predictor',
    packages=find_packages(include=[
        'energypredictor', 'energypredictor.*', 'energypredictor.main.*',
        'energypredictor.main.*.*'
    ],
                           exclude=['config', 'config.*', 'data', 'data.*']),
    install_requires=[
        'arch==4.15', 
        'BeautifulSoup4==4.9.1', 
        'Cython==0.29.14',
        'dask[dataframe]==2.24.0', 
        'dateparser==0.7.6', 
        'importlib_resources==3.0.0',
        'jsonformatter==0.2.3',
        'numba==0.51.0', 
        'numpy==1.19.1', 
        'pandas==0.25.3', 
        'plotly==4.9.0', 
        'pmdarima==1.7.0',
        'python-dateutil==2.8.1', 
        'python-dotenv==0.14.0', 
        'PyYAML==5.3.1', 
        'scipy==1.5.2', 
        'scikit-learn==0.23.2',
        'statsmodels==0.11.1', 
        'streamlit==0.65.2', 
        'tqdm==4.48.2', 
        'xmltodict==0.12.0'
    ],
    entry_points={
        'console_scripts': ['energy-predictor=energypredictor.__main__:main']
    },
    package_data={'energypredictor.resource': ['*.yml']},
    python_requires='>=3.6, <3.7',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
