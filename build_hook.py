"""
Custom build hook for hatchling to handle Cython extensions.
This maintains backward compatibility while using modern build tools.
"""

import os
import numpy
from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class CustomBuildHook(BuildHookInterface):
    """Custom build hook to handle Cython extensions and data files."""
    
    def initialize(self, version, build_data):
        """Initialize the build process."""
        # Add Cython extension
        if 'ext_modules' not in build_data:
            build_data['ext_modules'] = []
        
        # Import Cython here to avoid import issues during build
        try:
            from Cython.Build import cythonize
            
            # Try to import Extension from setuptools, fallback to distutils
            try:
                from setuptools import Extension
            except ImportError:
                try:
                    from distutils.core import Extension
                except ImportError:
                    print("Warning: Neither setuptools nor distutils available, skipping extension compilation")
                    build_data['ext_modules'] = []
                    return
            
            extensions = [
                Extension(
                    "radvel._kepler", 
                    ["src/_kepler.pyx"],
                    include_dirs=[numpy.get_include()]
                )
            ]
            
            # Cythonize the extensions
            build_data['ext_modules'] = cythonize(extensions)
            print("Successfully compiled Cython extensions")
            
        except ImportError as e:
            # Fallback if Cython is not available
            print(f"Warning: Cython not available ({e}), skipping extension compilation")
            build_data['ext_modules'] = []
        except Exception as e:
            print(f"Error during Cython compilation: {e}")
            build_data['ext_modules'] = []
        
        # Add data files
        if 'include_files' not in build_data:
            build_data['include_files'] = []
        
        data_files = [
            ('radvel_example_data', [
                'example_data/164922_fixed.txt',
                'example_data/epic203771098.csv',
                'example_data/k2-131.txt',
                'example_data/rvs_toi141.dat',
            ])
        ]
        
        # Filter out non-existent files
        existing_files = []
        for target_dir, files in data_files:
            existing_files_in_dir = []
            for file_path in files:
                if os.path.exists(file_path):
                    existing_files_in_dir.append(file_path)
            if existing_files_in_dir:
                existing_files.append((target_dir, existing_files_in_dir))
        
        build_data['include_files'] = existing_files
