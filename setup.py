import os
import re
from typing import Dict
from setuptools import find_packages, setup

ROOT_PATH = os.path.dirname(__file__)
PKG_NAME = "src"
PKG_PATH = os.path.join(ROOT_PATH, PKG_NAME)


def _load_version() -> str:
    version = ""
    version_path = os.path.join(PKG_PATH, "version.py")
    with open(version_path) as fp:
        version_module: Dict[str, str] = {}
        exec(fp.read(), version_module)
        version = version_module["__version__"]
    return version


def _load_description() -> str:
    readme_path = os.path.join(ROOT_PATH, "README.md")
    with open(readme_path) as fp:
        return fp.read()


def parse_file(requirement_file):
    try:
        requirement_list = []
        with open(requirement_file, "r") as file:
            for line in file:
                if not line.startswith("#") and not line.startswith("--"):
                    line = re.sub(r"\+cu.+", "", line).strip()
                    line = line.replace(r"${PWD}", os.getcwd())
                    requirement_list.append(line)
        return requirement_list
    except FileNotFoundError:
        return []


def find_requirements():
    requirements_path = os.path.join(ROOT_PATH, "requirements.txt")
    requirements = parse_file(requirements_path)
    return requirements


setup(
    name=PKG_NAME,
    version=_load_version(),
    description="Crypo tool",
    packages=find_packages(exclude=["tests", "tests.*"]),
    install_requires=find_requirements(),
    long_description=_load_description(),
    long_description_content_type="text/markdown"
)
