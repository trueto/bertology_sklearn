from setuptools import find_packages, setup

setup(
    name = 'bertology_sklearn',
    version = "1.0.0",
    author = 'trueto',
    author_email='ab1509359472@163.com',
    description="A sklearn wrapper for Transformers",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    keywords='scikit_sklearn Transformers NLP Deep Learning',
    license='Apache',
    url='https://github.com/trueto/bertology_sklearn',
    packages=find_packages(exclude="tutorial"),
    install_requires=['torch>=1.0.0',
                      'transformers>=2.2.0',
                      'scikit-learn',
                      'numpy',
                      'pandas',
                      'boto3',
                      'requests',
                      'tqdm',
                      'ignite'
                      ],
    python_requires='>=3.5.0',
    classifiers=[
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: Apache Software License',
          'Programming Language :: Python :: 3',
          'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
)