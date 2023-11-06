import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name='Stabl',
    version='0.0.1',
    author='Grégoire Bellan',
    author_email='gbellan@surge.care',
    description='Stabl package',
    packages=['stabl'],
    install_requires=[
        'joblib',
        'tqdm',
        'matplotlib',
        "knockpy",
        "scikit-learn",
        "seaborn",
        "groupyr",
        "pandas",
        "statsmodels",
        "openpyxl",
        "adjustText",
        "scipy",
        "osqp",
    ]
)
