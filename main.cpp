import Render;
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <stdexcept>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>

std::vector<char> loadShader(const std::string &filename)
{
    // Open at the end of file to get size
    std::ifstream fin(filename, std::ios::ate | std::ios::binary);
    if (!fin.is_open())
        throw std::runtime_error("Failed to open shader file");

    std::streamsize fileSize = (std::streamsize) fin.tellg();
    std::vector<char> buffer(fileSize);

    // Move to the start of file and read all data with known size
    fin.seekg(0);
    fin.read(buffer.data(), fileSize);

    fin.close();
    return buffer;
}

vulkan_render::Renderer* renderer;

int main()
{
    std::vector<char> vertexShader = loadShader("E:/study/CG/Vulkan/vert.spv");
    std::vector<char> fragmentShader = loadShader("E:/study/CG/Vulkan/frag.spv");

    glfwInit();
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

    GLFWwindow *window = glfwCreateWindow(800, 600, "Computer Graphics DEMO", nullptr, nullptr);
    renderer = new vulkan_render::Renderer(window, 1000000, 1000000, vertexShader, fragmentShader);

    glfwSetFramebufferSizeCallback(window, [](GLFWwindow *window, int width, int height) {
        renderer->ResizeWindow(width, height);
    });


    vulkan_data::RenderObject rect {
        .transform = {.pos = {0.0f, 0.0f, 0.0f}},
        .mesh = {
            .vertices = {
                {.pos = {-0.5f, -0.5f, 0.0f}, .color = {1.0f, 0.0f, 0.0f}},
                {.pos = {0.5f, -0.5f, 0.0f}, .color = {0.0f, 1.0f, 0.0f}},
                {.pos = {0.5f, 0.5f, 0.0f}, .color = {0.0f, 0.0f, 1.0f}},
                {.pos = {-0.5f, 0.5f, 0.0f}, .color = {0.0f, 0.0f, 0.0f}}
            },
            .indices = { 0, 1, 2, 2, 3, 0}
        }
    };

    vulkan_data::RenderObject cube {
        .transform = {.pos = {0.0f, 0.0f, 0.0f}},
        .mesh = {
            .vertices = {
                // Front (red)
                {.pos = {-0.5f, -0.5f,  0.5f}, .color = {1.0f, 0.0f, 0.0f}},
                {.pos = { 0.5f, -0.5f,  0.5f}, .color = {1.0f, 0.0f, 0.0f}},
                {.pos = { 0.5f,  0.5f,  0.5f}, .color = {1.0f, 0.0f, 0.0f}},
                {.pos = {-0.5f,  0.5f,  0.5f}, .color = {1.0f, 0.0f, 0.0f}},

                // Back (green)
                {.pos = {-0.5f, -0.5f, -0.5f}, .color = {0.0f, 1.0f, 0.0f}},
                {.pos = { 0.5f, -0.5f, -0.5f}, .color = {0.0f, 1.0f, 0.0f}},
                {.pos = { 0.5f,  0.5f, -0.5f}, .color = {0.0f, 1.0f, 0.0f}},
                {.pos = {-0.5f,  0.5f, -0.5f}, .color = {0.0f, 1.0f, 0.0f}},

                // Left (blue)
                {.pos = {-0.5f, -0.5f, -0.5f}, .color = {0.0f, 0.0f, 1.0f}},
                {.pos = {-0.5f, -0.5f,  0.5f}, .color = {0.0f, 0.0f, 1.0f}},
                {.pos = {-0.5f,  0.5f,  0.5f}, .color = {0.0f, 0.0f, 1.0f}},
                {.pos = {-0.5f,  0.5f, -0.5f}, .color = {0.0f, 0.0f, 1.0f}},

                // Right (yellow)
                {.pos = {0.5f, -0.5f, -0.5f}, .color = {1.0f, 1.0f, 0.0f}},
                {.pos = {0.5f, -0.5f,  0.5f}, .color = {1.0f, 1.0f, 0.0f}},
                {.pos = {0.5f,  0.5f,  0.5f}, .color = {1.0f, 1.0f, 0.0f}},
                {.pos = {0.5f,  0.5f, -0.5f}, .color = {1.0f, 1.0f, 0.0f}},

                // Top (pink)
                {.pos = {-0.5f, 0.5f, -0.5f}, .color = {1.0f, 0.0f, 1.0f}},
                {.pos = { 0.5f, 0.5f, -0.5f}, .color = {1.0f, 0.0f, 1.0f}},
                {.pos = { 0.5f, 0.5f,  0.5f}, .color = {1.0f, 0.0f, 1.0f}},
                {.pos = {-0.5f, 0.5f,  0.5f}, .color = {1.0f, 0.0f, 1.0f}},

                // Bottom (cyan)
                {.pos = {-0.5f, -0.5f, -0.5f}, .color = {0.0f, 1.0f, 1.0f}},
                {.pos = { 0.5f, -0.5f, -0.5f}, .color = {0.0f, 1.0f, 1.0f}},
                {.pos = { 0.5f, -0.5f,  0.5f}, .color = {0.0f, 1.0f, 1.0f}},
                {.pos = {-0.5f, -0.5f,  0.5f}, .color = {0.0f, 1.0f, 1.0f}},
            },

            .indices = {
                0,1,2, 2,3,0,        // Front
                4,6,5, 6,4,7,        // Back
                8,9,10, 10,11,8,     // Left
                12,14,13, 14,12,15,  // Right
                16,17,18, 18,19,16,  // Top
                20,22,21, 22,20,23   // Bottom
            }
        }
    };


    vulkan_data::RenderObject objects[] = {cube};
    renderer->RegisterObjects(objects);


    while(!glfwWindowShouldClose(window))
    {
        glfwPollEvents();
        renderer->Draw();
    }

    delete renderer;
    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}
