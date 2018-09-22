from setuptools import setup

setup(
    name='vuln_toolkit',
    version='0.1',
    description='Transfer Learning Toolkit',
    url='https://para.cs.umd.edu/purtilo/vulnerability-detection-tool-set/tree/master',
    author='Ashton Webster',
    author_email='ashton.webster@gmail.com',
    license='MIT',
    packages=['vuln_toolkit', 'vuln_toolkit.common'],
    zip_safe=False
)
