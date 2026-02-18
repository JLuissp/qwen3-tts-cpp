cmake_minimum_required(VERSION 3.18)
project(qwen3_tts_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ONNX Runtime
set(ONNX_DIR "" CACHE PATH "Path to ONNX Runtime root directory (optional)")
set(ONNX_INCLUDE_DIR "" CACHE PATH "Path to ONNX Runtime headers directory")
set(ONNX_RUNTIME_LIBRARY "" CACHE FILEPATH "Path to libonnxruntime shared library")

if(ONNX_DIR)
  if(NOT ONNX_INCLUDE_DIR)
    set(ONNX_INCLUDE_DIR "${ONNX_DIR}/include")
  endif()
  if(NOT ONNX_RUNTIME_LIBRARY)
    set(ONNX_RUNTIME_LIBRARY "${ONNX_DIR}/lib/libonnxruntime.so")
  endif()
endif()

if(NOT ONNX_RUNTIME_LIBRARY)
  file(GLOB _venv_ort_libs "${CMAKE_SOURCE_DIR}/venv/lib/python*/site-packages/onnxruntime/capi/libonnxruntime.so*")
  list(SORT _venv_ort_libs)
  if(_venv_ort_libs)
    list(GET _venv_ort_libs -1 ONNX_RUNTIME_LIBRARY)
  endif()
endif()

if(NOT EXISTS "${ONNX_RUNTIME_LIBRARY}")
  message(FATAL_ERROR
    "ONNX Runtime library not found. Set -DONNX_RUNTIME_LIBRARY=/path/to/libonnxruntime.so "
    "or -DONNX_DIR=/path/to/onnxruntime")
endif()

if(NOT ONNX_INCLUDE_DIR OR NOT EXISTS "${ONNX_INCLUDE_DIR}")
  message(FATAL_ERROR
    "ONNX Runtime headers not found. Set -DONNX_INCLUDE_DIR=/path/to/onnxruntime/include. "
    "Note: python wheel in venv usually provides runtime .so but not C++ headers.")
endif()

get_filename_component(ONNX_RUNTIME_DIR "${ONNX_RUNTIME_LIBRARY}" DIRECTORY)
get_filename_component(ONNX_RUNTIME_NAME "${ONNX_RUNTIME_LIBRARY}" NAME)

add_library(onnxruntime SHARED IMPORTED)
set_target_properties(onnxruntime PROPERTIES
  IMPORTED_LOCATION "${ONNX_RUNTIME_LIBRARY}"
  INTERFACE_INCLUDE_DIRECTORIES "${ONNX_INCLUDE_DIR}"
)

add_library(qwen3_tts_cpp
  src/voice.h
  src/voice.cpp
  src/tokenizer.h
  src/tokenizer.cpp
  src/utils.h
  src/utils.cpp
)

target_include_directories(qwen3_tts_cpp PUBLIC ${ONNX_INCLUDE_DIR} ${CMAKE_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(qwen3_tts_cpp PUBLIC onnxruntime)

add_executable(qwen3_tts_cpp_cli_example
  examples/voice_design_cli_example.cpp
)
add_executable(qwen3_tts_cpp_timing_example
  examples/voice_design_timing_example.cpp
)
add_executable(qwen3_tts_cpp_full_profile_example
  examples/voice_design_full_profile_example.cpp
)

target_include_directories(qwen3_tts_cpp_cli_example PRIVATE ${ONNX_INCLUDE_DIR})
target_link_libraries(qwen3_tts_cpp_cli_example PRIVATE qwen3_tts_cpp)
target_include_directories(qwen3_tts_cpp_timing_example PRIVATE ${ONNX_INCLUDE_DIR})
target_link_libraries(qwen3_tts_cpp_timing_example PRIVATE qwen3_tts_cpp)
target_include_directories(qwen3_tts_cpp_full_profile_example PRIVATE ${ONNX_INCLUDE_DIR})
target_link_libraries(qwen3_tts_cpp_full_profile_example PRIVATE qwen3_tts_cpp)
target_include_directories(qwen3_tts_cpp_cli_example PRIVATE ${CMAKE_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/src)
target_include_directories(qwen3_tts_cpp_timing_example PRIVATE ${CMAKE_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/src)
target_include_directories(qwen3_tts_cpp_full_profile_example PRIVATE ${CMAKE_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/src)

find_package(Threads REQUIRED)
target_link_libraries(qwen3_tts_cpp PUBLIC Threads::Threads)
target_link_libraries(qwen3_tts_cpp_timing_example PRIVATE Threads::Threads)
target_link_libraries(qwen3_tts_cpp_full_profile_example PRIVATE Threads::Threads)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(qwen3_tts_cpp PRIVATE -Wall -Wextra -Wno-unused-parameter)
  target_compile_options(qwen3_tts_cpp_cli_example PRIVATE -Wall -Wextra -Wno-unused-parameter)
  target_compile_options(qwen3_tts_cpp_timing_example PRIVATE -Wall -Wextra -Wno-unused-parameter)
  target_compile_options(qwen3_tts_cpp_full_profile_example PRIVATE -Wall -Wextra -Wno-unused-parameter)
endif()

set_target_properties(qwen3_tts_cpp_cli_example PROPERTIES BUILD_RPATH "${CMAKE_BINARY_DIR};${ONNX_RUNTIME_DIR}" INSTALL_RPATH "${ONNX_RUNTIME_DIR}")
set_target_properties(qwen3_tts_cpp_timing_example PROPERTIES BUILD_RPATH "${CMAKE_BINARY_DIR};${ONNX_RUNTIME_DIR}" INSTALL_RPATH "${ONNX_RUNTIME_DIR}")
set_target_properties(qwen3_tts_cpp_full_profile_example PROPERTIES BUILD_RPATH "${CMAKE_BINARY_DIR};${ONNX_RUNTIME_DIR}" INSTALL_RPATH "${ONNX_RUNTIME_DIR}")

if(ONNX_RUNTIME_NAME MATCHES "^libonnxruntime\\.so\\.[0-9].*")
  add_custom_target(onnxruntime_symlink ALL
    COMMAND ${CMAKE_COMMAND} -E rm -f "${CMAKE_BINARY_DIR}/libonnxruntime.so.1"
    COMMAND ${CMAKE_COMMAND} -E create_symlink "${ONNX_RUNTIME_LIBRARY}" "${CMAKE_BINARY_DIR}/libonnxruntime.so.1")
  add_dependencies(qwen3_tts_cpp_cli_example onnxruntime_symlink)
  add_dependencies(qwen3_tts_cpp_timing_example onnxruntime_symlink)
  add_dependencies(qwen3_tts_cpp_full_profile_example onnxruntime_symlink)
endif()
