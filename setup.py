import os
from setuptools import setup, find_packages

package_dir = os.path.dirname(os.path.abspath(__file__))
requirements_file = os.path.join(package_dir, "requirements.txt")
with open(requirements_file, "r") as rf:
    install_requires = [
        req.strip()
        for req in rf.readlines()
        if req.strip() and not req.startswith("#")
    ]

setup(
    name="TemplateFitter",
    version="0.0.1",
    author="Maximilian Welsch, Felix Metzner",
    url="https://github.com/FelixMetzner/TemplateFitter",
    packages=find_packages(),
    description="Perform extended binned log-likelihood fits using histogram templates as PDFs.",
    install_requires=install_requires
)
