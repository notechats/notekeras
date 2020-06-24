import sys
from os import path

from setuptools import setup, find_packages


def get_version(argv):
    here = path.abspath(path.dirname(__file__))
    version_path = path.join(here, '__version__.md')
    with open(version_path, 'r') as f:
        version1 = [int(i) for i in f.read().split('.')]

    if len(argv) >= 2 and argv[1] == 'build':
        version2 = version1[0] * 100 + version1[1] * 10 + version1[2] + 1

        version1[2] = version2 % 10
        version1[1] = int(version2 / 10) % 10
        version1[0] = int(version2 / 100)

    version3 = '{}.{}.{}'.format(*version1)
    with open(version_path, 'w') as f:
        f.write(version3)
    return version3


version = get_version(sys.argv)

install_requires = ['tensorflow', 'numpy', 'scipy', 'opencv-python', 'pillow', 'tqdm']

setup(name='notekeras',
      version=version,
      description='notekeras',
      author='niuliangtao',
      author_email='1007530194@qq.com',
      url='https://github.com/1007530194',

      packages=find_packages(),
      install_requires=install_requires
      )
