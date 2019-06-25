# encoding: utf-8
from setuptools import setup

setup(name='rascal',
      version='0.5',
      description='Visually grounded speech with inductive biases',
      url='https://github.com/gchrupala/vgs',
      author='Grzegorz Chrupała',
      author_email='g.chrupala@uvt.nl',
      license='MIT',
      packages=['onion','vg', 'vg.defn'],
      zip_safe=False,
      install_requires=[
          'torch == 1.0.0',
          'torchvision == 0.2.1',
          'scikit-learn == 0.20.1',
          'scipy == 1.2.0rc2',
          'python-Levenshtein == 0.12.0'
          
      ])
