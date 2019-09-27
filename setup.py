from setuptools import setup, find_packages


setup(version='0.1.0',
      name='pillownet',
      description='U-Nets for genomics',
      long_description=open('README.md').read(),
      url='https://github.com/daquang/PillowNet',
      license='MIT',
      author='Daniel Quang',
      author_email='daquang@umich.edu',
      packages=find_packages(),
      install_requires=['numpy', 'genomeloader', 'keras', 'tensorflow']
      )
