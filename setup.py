from pathlib import Path
from setuptools import find_packages, setup


def write_version_to_file(version, target_file):
    with open(target_file, 'w') as f:
        print('__version__ = "%s"' % version, file=f)

if __name__ == '__main__':
    version = '0.3.1'
    write_version_to_file(version, 'struct_eqtable/version.py')
    with Path(Path(__file__).parent,
              'README.md').open(encoding='utf-8') as file:
        long_description = file.read()
    setup(
        name='struct_eqtable',
        version=version,
        description='A High-efficiency Open-source Toolkit for Table-to-Latex Transformation',
        long_description=long_description,
        long_description_content_type="text/markdown",
        install_requires=[
            'torch',
            'transformers',
        ],
        python_requires=">=3.9",
        author='Hongbin Zhou, Xiangchao Yan, Bo Zhang',
        author_email='zhangbo@pjlab.org.cn',
        url="https://github.com/UniModal4Reasoning/StructEqTable-Deploy",
        license='Apache License 2.0',
        packages=find_packages(exclude=['demo']),
    )
