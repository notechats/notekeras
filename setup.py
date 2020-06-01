from setuptools import setup, find_packages

install_requires = ['tensorflow', 'numpy', 'scipy', 'opencv-python', 'pillow', 'tqdm']

setup(name='notekeras',
      version='0.1.3',
      description='notekeras',
      author='niuliangtao',
      author_email='1007530194@qq.com',
      url='https://github.com/1007530194',

      packages=find_packages(),
      install_requires=install_requires
      )
