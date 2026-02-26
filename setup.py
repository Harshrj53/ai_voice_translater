"""
setup.py — Hindi Video Dubbing Pipeline
Install with: pip install -e .
"""
from setuptools import setup, find_packages

with open("requirement.txt") as f:
    requirements = [
        line.strip()
        for line in f
        if line.strip() and not line.startswith("#")
    ]

with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="hindi-dubbing-pipeline",
    version="1.0.0",
    author="Harsh Raj",
    description="Modular Hindi video dubbing pipeline with voice cloning and lip-sync",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Harshrj53/Supernan_video_dubbing_project",
    packages=find_packages(exclude=["ml_env", "*.egg-info"]),
    python_requires=">=3.9",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "dub-video=dub_video:main",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
