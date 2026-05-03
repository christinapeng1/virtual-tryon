cmake_minimum_required(VERSION 3.10)
project(CVAR)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(OpenGL REQUIRED)
find_package(glfw3 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(glm REQUIRED)

# Explicitly link assimp
find_library(ASSIMP_LIBRARY assimp PATHS /opt/homebrew/opt/assimp/lib)
find_path(ASSIMP_INCLUDE assimp/Importer.hpp PATHS /opt/homebrew/opt/assimp/include)

if(NOT ASSIMP_LIBRARY)
    message(FATAL_ERROR "assimp library not found")
endif()
if(NOT ASSIMP_INCLUDE)
    message(FATAL_ERROR "assimp headers not found")
endif()

add_executable(main src/main.cpp src/pose_tracker.cpp src/utils.cpp src/mesh.cpp)

target_include_directories(main PRIVATE ${OpenCV_INCLUDE_DIRS} ${ASSIMP_INCLUDE})

target_link_libraries(main
    glfw
    OpenGL::GL
    ${OpenCV_LIBS}
    ${ASSIMP_LIBRARY}
    glm::glm
)

add_executable(calibration calibration.cpp src/mesh.cpp)

target_include_directories(calibration PRIVATE src ${OpenCV_INCLUDE_DIRS} ${ASSIMP_INCLUDE})

target_link_libraries(calibration
    glfw
    OpenGL::GL
    ${OpenCV_LIBS}
    ${ASSIMP_LIBRARY}
    glm::glm
)
