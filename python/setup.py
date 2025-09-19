"""
Setup script for TensorCore Python bindings
"""

from setuptools import setup, Extension, find_packages
from pybind11.setup_helpers import Pybind11Extension, build_ext
from pybind11 import get_cmake_dir
import pybind11
import os
import sys
import subprocess
import platform

# Get the directory containing this setup.py file
here = os.path.abspath(os.path.dirname(__file__))

# Read the README file
with open(os.path.join(here, '..', 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# Read requirements
with open(os.path.join(here, 'requirements.txt'), 'r') as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

# Define the extension module
ext_modules = [
    Pybind11Extension(
        "tensorcore_core",
        [
            "tensorcore_core.cpp",
        ],
        include_dirs=[
            # Path to pybind11 headers
            pybind11.get_include(),
            # Path to TensorCore headers
            os.path.join(here, '..', 'include'),
        ],
        libraries=['tensorcore'],
        library_dirs=[
            os.path.join(here, '..', 'build'),
        ],
        language='c++',
        cxx_std=17,
    ),
]

# Custom build command to handle CMake
class CustomBuildExt(build_ext):
    def build_extensions(self):
        # Build the C++ library first using CMake
        self.build_cmake()
        super().build_extensions()
    
    def build_cmake(self):
        """Build the C++ library using CMake"""
        build_dir = os.path.join(here, '..', 'build')
        os.makedirs(build_dir, exist_ok=True)
        
        # Configure CMake
        cmake_args = [
            '-DCMAKE_BUILD_TYPE=Release',
            '-DCMAKE_CXX_STANDARD=17',
            '-DBUILD_PYTHON_BINDINGS=ON',
        ]
        
        # Add platform-specific arguments
        if platform.system() == "Windows":
            cmake_args.extend(['-G', 'Visual Studio 16 2019'])
        elif platform.system() == "Darwin":
            cmake_args.extend(['-DCMAKE_OSX_DEPLOYMENT_TARGET=10.14'])
        
        # Run CMake configuration
        subprocess.check_call(['cmake', '..'] + cmake_args, cwd=build_dir)
        
        # Build the library
        subprocess.check_call(['cmake', '--build', '.', '--config', 'Release'], cwd=build_dir)

# Setup configuration
setup(
    name="tensorcore",
    version="1.0.0",
    author="TensorCore Contributors",
    author_email="tensorcore@example.com",
    description="Educational Machine Learning Library",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/tensorcore",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/tensorcore/issues",
        "Source": "https://github.com/yourusername/tensorcore",
        "Documentation": "https://tensorcore.readthedocs.io/",
    },
    packages=find_packages(),
    ext_modules=ext_modules,
    cmdclass={"build_ext": CustomBuildExt},
    zip_safe=False,
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.800",
            "sphinx>=4.0",
            "sphinx-rtd-theme>=0.5",
        ],
        "test": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "pytest-benchmark>=3.4",
        ],
        "docs": [
            "sphinx>=4.0",
            "sphinx-rtd-theme>=0.5",
            "myst-parser>=0.15",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: C++",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Education",
    ],
    keywords="machine learning, deep learning, tensor, numpy, pytorch, tensorflow, education, c++",
    include_package_data=True,
    package_data={
        "tensorcore": ["*.hpp", "*.cpp"],
    },
    entry_points={
        "console_scripts": [
            "tensorcore-benchmark=tensorcore.benchmarks:main",
            "tensorcore-test=tensorcore.tests:main",
        ],
    },
    platforms=["any"],
    license="MIT",
    zip_safe=False,
)
