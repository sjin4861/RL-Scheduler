from setuptools import setup, find_packages

setup(
    name='RJSPEnv',
    version='0.1',
    packages=['RJSPEnv'], 
    include_package_data=True,  # Includes additional files if declared in MANIFEST.in
    install_requires=[
        'seaborn',
    ],  # Add your dependencies here
)
