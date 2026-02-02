from setuptools import setup, find_packages
from pathlib import Path

# Read requirements from requirements.txt
requirements_path = Path(__file__).parent / "requirements.txt"
install_requires = requirements_path.read_text().splitlines()

setup(
    name='vesselfm',
    version='1.0',
    description='vesselfm',
    packages=find_packages(),
    install_requires=install_requires,
    entry_points={
        'console_scripts': [
            'vesselfm-infer=vesselfm.cli:main',
        ],
    },
    python_requires=">=3.9"
)