import os
import shutil
from setuptools.command.build_ext import build_ext
from setuptools import  setup, find_packages, Extension
import setuptools
import warnings
import sys
import subprocess
class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=""):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


def is_ninja_available():
    r'''
    Returns ``True`` if the `ninja <https://ninja-build.org/>`_ build system is
    available on the system, ``False`` otherwise.
    '''
    with open(os.devnull, 'wb') as devnull:
        try:
            subprocess.check_call('ninja --version'.split(), stdout=devnull)
        except OSError:
            return False
        else:
            return True


class CMakeBuild(build_ext):
 
    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))

        # required for auto-detection & inclusion of auxiliary "native" libs
        if not extdir.endswith(os.path.sep):
            extdir += os.path.sep

        debug = int(os.environ.get("DEBUG", 0)) if self.debug is None else self.debug
        cfg = "Debug" if debug else "Release"

        # CMake lets you override the generator - we need to check this.
        # Can be set with Conda-Build, for example.
        cmake_generator = os.environ.get("CMAKE_GENERATOR", "")

        # Set Python_EXECUTABLE instead if you use PYBIND11_FINDPYTHON
        # EXAMPLE_VERSION_INFO shows you how to pass a value into the C++ code
        # from Python.
        cmake_args = [
	        f"-DCMAKE_CXX_STANDARD=14",
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            f"-DPYTHON_VERSION={sys.version_info.major}.{sys.version_info.minor}",
            f"-DCMAKE_BUILD_TYPE={cfg}",  # not used on MSVC, but no harm
        ]

        build_args = []
        if "CMAKE_ARGS" in os.environ:
            cmake_args += [item for item in os.environ["CMAKE_ARGS"].split(" ") if item]

        cmake_args += [f"-DEXAMPLE_VERSION_INFO={self.distribution.get_version()}"]


        if not cmake_generator or cmake_generator == "Ninja":
            try:
                import ninja  # noqa: F401

                ninja_executable_path = os.path.join(ninja.BIN_DIR, "ninja")
                cmake_args += [
                    "-GNinja",
                    f"-DCMAKE_MAKE_PROGRAM:FILEPATH={ninja_executable_path}",
                ]
            except ImportError:
                pass

        if "CMAKE_BUILD_PARALLEL_LEVEL" not in os.environ:
            # self.parallel is a Python 3 only way to set parallel jobs by hand
            # using -j in the build_ext call, not supported by pip or PyPA-build.
            if hasattr(self, "parallel") and self.parallel:
                # CMake 3.12+ only.
                build_args += [f"-j{self.parallel}"]

        build_temp = os.path.join(self.build_temp, ext.name)
        if os.path.exists(build_temp):
            shutil.rmtree(build_temp)
        os.makedirs(build_temp)

        cmake_args += ["-DPython_ROOT_DIR=" + os.path.dirname(os.path.dirname(sys.executable))]
        subprocess.check_call(["cmake", ext.sourcedir] + cmake_args, cwd=build_temp)
        subprocess.check_call(["cmake", "--build", "."] + build_args, cwd=build_temp)
        
ext_modules = [
    CMakeExtension("bmtrain.C"),
]
setup(
    name='bmtrain',
    version='1.0.1',
    author="Guoyang Zeng",
    author_email="qbjooo@qq.com",
    description="A toolkit for training big models",
    packages=find_packages(),
    install_requires=[
        "numpy",
		"nvidia-nccl-cu11>=2.14.3"
    ],
    setup_requires=[
        "pybind11",
        "nvidia-nccl-cu11>=2.14.3"
    ],
    ext_modules=ext_modules,
    cmdclass={
        'build_ext': CMakeBuild
    })

