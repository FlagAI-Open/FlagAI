cmake_minimum_required(VERSION 3.18)
project(bmtrain)
enable_language(C)
enable_language(CXX)
set(CMAKE_CUDA_ARCHITECTURES "61;62;70;72;75;80")
enable_language(CUDA)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED True)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode=arch=compute_61,code=sm_61 -gencode=arch=compute_62,code=sm_62 -gencode=arch=compute_70,code=sm_70 -gencode=arch=compute_72,code=sm_72 -gencode=arch=compute_75,code=sm_75 -gencode=arch=compute_80,code=sm_80")

if(NOT DEFINED ENV{BUILD_DOCKER_ENV} OR "$ENV{BUILD_DOCKER_ENV}" STREQUAL "0")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode=arch=compute_86,code=sm_86")
    set(AVX_FLAGS "${AVX_FLAGS} -march=native")
else()
    message("Building in docker environment, skipping compute_86 and enable all avx flag")
    set(AVX_FLAGS "${AVX_FLAGS} -mavx -mfma -mf16c -mavx512f")
endif()

set(CMAKE_BUILD_RPATH $ORIGIN)
set(CMAKE_INSTALL_RPATH $ORIGIN)
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/)

find_package(NCCL REQUIRED)
find_package(Python ${PYTHON_VERSION} EXACT COMPONENTS Interpreter Development.Module REQUIRED)
message (STATUS "Python_EXECUTABLE: ${Python_EXECUTABLE}")
execute_process(COMMAND ${Python_EXECUTABLE} "-c"
    "import pybind11; print(pybind11.get_cmake_dir())"
    OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE)
message (STATUS "PYBIND11_CMAKE_DIR: ${PYB
IND11_CMAKE_DIR}")
list(APPEND CMAKE_PREFIX_PATH ${PYBIND11_CMAKE_DIR})
find_package(pybind11 REQUIRED)

message (STATUS "CMAKE_INSTALL_RPATH: ${CMAKE_INSTALL_RPATH}")

file(GLOB_RECURSE SOURCES "csrc/*.cpp")
file(GLOB_RECURSE CUDA_SOURCES "csrc/cuda/*.cu")


pybind11_add_module(C ${SOURCES} ${CUDA_SOURCES})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${AVX_FLAGS}")

target_link_libraries(C PRIVATE
    "-Wl,-Bsymbolic"
    "-Wl,-Bsymbolic-functions"
    ${NCCL_LIBRARIES}
)
target_include_directories(C PRIVATE ${NCCL_INCLUDE_DIRS})
target_compile_definitions(C
    PRIVATE VERSION_INFO=${EXAMPLE_VERSION_INFO})

set_target_properties(C PROPERTIES CUDA_ARCHITECTURES "61;62;70;72;75;80")

target_include_directories(C
    PRIVATE "csrc/include"
    PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)



