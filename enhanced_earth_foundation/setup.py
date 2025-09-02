"""Setup script for Enhanced Earth Foundation Model"""

from setuptools import setup, find_packages
from pathlib import Path

# 读取README
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

# 读取requirements
requirements_path = Path(__file__).parent / "requirements.txt"
if requirements_path.exists():
    with open(requirements_path, 'r') as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]
else:
    requirements = []

setup(
    name="enhanced-earth-foundation",
    version="0.1.0",
    description="Enhanced Earth Foundation Model - A global-scale multimodal earth observation foundation model",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Enhanced Earth Team",
    author_email="contact@enhanced-earth.ai",
    url="https://github.com/enhanced-earth/foundation-model",
    
    packages=find_packages("src"),
    package_dir={"": "src"},
    
    python_requires=">=3.8",
    install_requires=requirements,
    
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "black>=23.7.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
            "mypy>=1.5.0",
            "pre-commit>=3.3.0"
        ],
        "flash": [
            "flash-attn>=2.3.0"
        ],
        "distributed": [
            "deepspeed>=0.10.0",
            "fairscale>=0.4.13"
        ]
    },
    
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: GIS",
    ],
    
    keywords="earth observation, foundation model, multimodal, remote sensing, transformer",
    
    entry_points={
        "console_scripts": [
            "enhanced-earth-train=enhanced_earth.scripts.train:main",
            "enhanced-earth-demo=enhanced_earth.scripts.demo:main_demo",
        ],
    },
    
    include_package_data=True,
    zip_safe=False,
)