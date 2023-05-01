from setuptools import setup, find_packages

from cmpe255gp1 import __version__

setup(
    name='cmpe255gp1',
    version=__version__,

    url='https://github.com/d1ndra/cmpe255_group1',
    author='Indranil Dutta',
    author_email='duttaindranil497@gmail.com',

    py_modules=find_packages(),
    install_requires=[
        'pandas',
        'numpy',
        'matplotlib',
	'seaborn'
    ],
)
