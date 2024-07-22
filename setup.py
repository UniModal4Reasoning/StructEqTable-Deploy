from setuptools import find_packages, setup


def write_version_to_file(version, target_file):
    with open(target_file, 'w') as f:
        print('__version__ = "%s"' % version, file=f)

if __name__ == '__main__':
    version = '0.1.0'
    write_version_to_file(version, 'struct_eqtable/version.py')

    setup(
        name='struct_eqtable',
        version=version,
        description='A High-efficiency Open-source Toolkit for Table-to-Latex Transformation',
        install_requires=[
            'torch',
            'transformers',
        ],

        author='Hongbin Zhou, Xiangchao Yan, Bo Zhang',
        author_email='zhangbo@pjlab.org.cn',
        license='Apache License 2.0',
        packages=find_packages(exclude=['demo']),
    )
