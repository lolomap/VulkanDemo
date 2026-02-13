cmake_minimum_required(VERSION 4.2)
project(Vulkan)

set(CMAKE_CXX_STANDARD 23)

find_package(Vulkan REQUIRED)
find_package(Boost REQUIRED COMPONENTS lockfree)
find_package(glfw3 CONFIG REQUIRED)
find_package(glm CONFIG REQUIRED)

add_executable(${PROJECT_NAME} main.cpp
        Render.cpp)
target_sources(${PROJECT_NAME} PUBLIC FILE_SET modules TYPE CXX_MODULES BASE_DIRS ${PROJECT_SOURCE_DIR} FILES Render.cppm)

target_link_libraries(${PROJECT_NAME} PRIVATE Boost::lockfree)
target_link_libraries(${PROJECT_NAME} PRIVATE glfw)
target_link_libraries(${PROJECT_NAME} PRIVATE glm::glm)
target_link_libraries(${PROJECT_NAME} PRIVATE Vulkan::Vulkan)