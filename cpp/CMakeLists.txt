cmake_minimum_required(VERSION 3.0)
project(torch_eval)

set(CMAKE_CXX_STANDARD 17)

# Set the paths to the required include directories
set(TORCH_INCLUDE_DIRS "/home/ubuntu/chenxinwei/libtorch/include")
set(TORCH_API_INCLUDE_DIRS "/home/ubuntu/chenxinwei/libtorch/include/torch/csrc/api/include")

# Set the path to the libtorch library
set(TORCH_LIBRARIES "/home/ubuntu/chenxinwei/libtorch/lib")

# Add the include directories
include_directories(${TORCH_INCLUDE_DIRS})
include_directories(${TORCH_API_INCLUDE_DIRS})

# Add the executable target
add_executable(torch_eval torch_eval.cpp)

# Link the required libraries
target_link_libraries(torch_eval "${TORCH_LIBRARIES}/libtorch_cpu.so")
target_link_libraries(torch_eval "${TORCH_LIBRARIES}/libc10.so")
target_link_libraries(torch_eval pthread)
target_link_libraries(torch_eval stdc++)

# Set the output path for the executable
set(EXECUTABLE_OUTPUT_PATH "/home/ubuntu/chenxinwei/grl/cpp/")

# The following code block is suggested to be used on Windows.
# According to https://github.com/pytorch/pytorch/issues/25457,
# the DLLs need to be copied to avoid memory errors.
if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET torch_eval
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:torch_eval>)
endif (MSVC)
