import shutil
from pathlib import Path

import setuptools

stale_egg_info = Path(__file__).parent / "bertology_sklearn.egg-info"
if stale_egg_info.exists():
    shutil.rmtree(stale_egg_info)

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="bertology_sklearn",
    version="0.0.1",
    author="trueto",
    author_email="ab1509359472@163.com",
    description="a package for users working with bertology models in scikit-learn style",
    long_description= long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/trueto/bertology_sklearn",
    license="Apache",
    package_dir={"": "src"},
    packages=setuptools.find_packages("src"),
    classifiers=[
        "Deep Learning::BERT"
    ],
    python_requires='>=3.6'
)