import os
from setuptools import setup, find_packages

package_dir = os.path.abspath(__file__)
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
    author="Maximilian Welsch",
    url="https://github.com/welschma/TemplateFitter",
    packages=find_packages(),
    description="Perform extended binnend log-likelhood fits using histogram templates as pdfs.",
    install_requires=install_requires
)
