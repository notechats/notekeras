from setuptools import setup, find_packages

install_requires = ['numpy', 'scipy', 'opencv-python', 'pillow', 'tqdm']

setup(name='notekeras',
      version='0.0.8',
      description='notekeras',
      author='niuliangtao',
      author_email='1007530194@qq.com',
      url='https://github.com/1007530194',

      packages=find_packages(),
      install_requires=install_requires
      )