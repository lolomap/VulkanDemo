module;

#include <vector>
#include <array>
#include <span>
#include <set>
#include <cstdint>
#include <cstddef>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#define QUEUE_FAMILY_NO_VALUE UINT32_MAX
#define ATTRIBUTE_FORMAT_VEC3 VK_FORMAT_R32G32B32_SFLOAT

export module Render;

export namespace vulkan_data
{
    struct GPU_Transform
    {
        glm::vec3 pos = glm::vec3(0.0f, 0.0f, 0.0f);
        glm::vec3 scale = glm::vec3(1.0f, 1.0f, 1.0f);
    };

    struct GPU_Vertex
    {
        glm::vec3 pos;
        glm::vec3 color;

        static std::vector<VkVertexInputAttributeDescription> GetAttributes()
        {
            VkVertexInputAttributeDescription posAttribute = {
                    .location = 0,
                    .binding = 0,
                    .format = ATTRIBUTE_FORMAT_VEC3,
                    .offset = offsetof(GPU_Vertex, pos)
            };
            VkVertexInputAttributeDescription colorAttribute = {
                    .location = 1,
                    .binding = 0,
                    .format = ATTRIBUTE_FORMAT_VEC3,
                    .offset = offsetof(GPU_Vertex, color)
            };

            return {posAttribute, colorAttribute};
        }
    };

    struct GPU_GlobalUBO
    {
        glm::mat4 view;
        glm::mat4 proj;
    };

    struct GPU_TranformUBO
    {
        glm::mat4 model;

        void Update(const GPU_Transform &transform)
        {
            model = glm::mat4(1.0f);
            model = glm::translate(model, transform.pos);
            model = glm::scale(model, transform.scale);
        }
    };

    struct Mesh
    {
        size_t verticesBufferPosition = 0;
        size_t indicesBufferPosition = 0;
        std::vector<GPU_Vertex> vertices;
        std::vector<uint16_t> indices;
    };

    struct RenderObject
    {
        uint32_t engineId = UINT32_MAX;
        GPU_Transform transform;
        Mesh mesh;
        //TODO: texture

        // This is needed to store RenderObject in std::set
        bool operator < (const vulkan_data::RenderObject& rhs) const
        {
            return engineId < rhs.engineId;
        }
    };
}

namespace vulkan_configs
{
    struct QueueFamiliesBundle
    {
        uint32_t transfer = QUEUE_FAMILY_NO_VALUE;
        uint32_t graphics = QUEUE_FAMILY_NO_VALUE;
        uint32_t present = QUEUE_FAMILY_NO_VALUE;
    };

    struct QueuesBundle
    {
        VkQueue transfer;
        VkQueue graphics;
        VkQueue present;
    };

    struct DrawConfig
    {
        VkCommandBufferBeginInfo commandBeginConfig {
                .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                .pNext = nullptr,
                .flags = 0,
                .pInheritanceInfo = nullptr
        };

        VkClearValue clearColor {{{1.0f, 1.0f, 1.0f, 1.0f}}};

        VkRenderPassBeginInfo renderPassBeginConfig {
                .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
                .pNext = nullptr,
                .clearValueCount = 1,
                .pClearValues = &clearColor
        };

        VkPipelineStageFlags waitImageStage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;

        VkSubmitInfo submitConfig {
                .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                .pNext = nullptr,
                .waitSemaphoreCount = 1,
                .pWaitDstStageMask = &waitImageStage,
                .commandBufferCount = 1,
                .signalSemaphoreCount = 1
        };

        VkPresentInfoKHR presentConfig {
                .sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
                .pNext = nullptr,
                .waitSemaphoreCount = 1,
                .swapchainCount = 1,
                .pResults = nullptr
        };

        VkVertexInputBindingDescription vertexBindingConfig {
                .binding = 0,
                .stride = sizeof(vulkan_data::GPU_Vertex),
                .inputRate = VK_VERTEX_INPUT_RATE_VERTEX
        };
    };

    struct SurfaceInfo
    {
        VkSurfaceCapabilitiesKHR capabilities;
        std::vector<VkSurfaceFormatKHR> availableFormats;
        std::vector<VkPresentModeKHR> availablePresentModes;

        VkSurfaceFormatKHR format;
        VkPresentModeKHR presentMode;
        VkExtent2D extent;
    };

    struct PipelineBinding
    {
        const VkRenderPass *baseRenderPass;
        const VkDevice *device;
        const VkExtent2D *surfaceExtent;

        const VkVertexInputBindingDescription *vertexBindingConfig;
        uint32_t vertexAttributesCount;
        const VkVertexInputAttributeDescription *vertexAttributesConfig;
    };

    struct PipelineConfig
    {
        VkPipelineVertexInputStateCreateInfo vertexInputConfig;
        VkPipelineInputAssemblyStateCreateInfo inputAssemblyConfig;
        VkPipelineViewportStateCreateInfo viewportStateConfig;
        VkPipelineRasterizationStateCreateInfo rasterizerConfig;
        VkPipelineMultisampleStateCreateInfo multisamplingConfig;
        VkPipelineDepthStencilStateCreateInfo depthStencilConfig;

        // SRC - new color, DST - old color
        std::vector<VkPipelineColorBlendAttachmentState> colorBlendAttachmentConfigs;
        VkPipelineColorBlendStateCreateInfo colorBlendConfig;
        VkPipelineDynamicStateCreateInfo dynamicStateConfig;
        VkPipelineLayoutCreateInfo pipelineLayoutConfig;

        VkViewport defaultViewport;
        VkRect2D defaultScissors;
        std::vector<VkDynamicState> dynamicStates;

        std::vector<VkDescriptorSetLayoutBinding> uniforms;
        int vertexBindingsCount, fragmentBindingsCount;
        VkDescriptorSetLayoutCreateInfo uniformsConfig;

        VkGraphicsPipelineCreateInfo pipelineConfig;
    };

    struct Pipeline
    {
        VkPipeline instance;
        VkPipelineLayout layout;
        VkDescriptorSetLayout uniformsLayout;
        VkViewport viewport;
        VkRect2D scissors;
    };
}

export namespace vulkan_render
{
    class Renderer
    {
    public:
        Renderer(GLFWwindow *window, const size_t &vertexBufferSize, const size_t &indexBufferSize,
                 std::span<char> vertexShader, std::span<char> fragmentShader);
        ~Renderer();

        // Isn't safe for multiple calls, just simple overriding buffers for demo
        void RegisterObjects(std::span<vulkan_data::RenderObject> objects);
        void ResizeWindow(int width, int height);

        void Draw();

    private:
        int screenWidth, screenHeight;
        bool isDrawing = true;

        GLFWwindow *window;

        VkInstance vulkan;
        VkSurfaceKHR surface;
        VkPhysicalDevice gpu;
        VkDevice device;
        VkSwapchainKHR swapchain;
        VkRenderPass baseRenderPass;
        VkCommandPool graphicsCommandPool = nullptr;
        VkCommandPool transferCommandPool = nullptr;
        VkDeviceMemory verticesMemory;
        VkDeviceMemory indicesMemory;
        VkDeviceMemory stagingVerticesMemory;
        VkDeviceMemory stagingIndicesMemory;
        VkBuffer vertices;
        VkBuffer indices;
        VkBuffer stagingVertices;
        VkBuffer stagingIndices;
        std::vector<VkDeviceMemory> modelUniformsMemory;
        std::vector<VkBuffer> modelUniforms;
        std::vector<VkDeviceMemory> vpUniformsMemory;
        std::vector<VkBuffer> vpUniforms;
        std::vector<VkCommandBuffer> graphicsCommandBuffers;
        std::vector<VkCommandBuffer> transferCommandBuffers;
        std::vector<VkImage> frames;
        std::vector<VkImageView> frameViews;
        std::vector<VkFramebuffer> baseFrameBuffers;
        std::vector<VkSemaphore> imageAvailable;
        std::vector<VkSemaphore> renderFinished;
        std::vector<VkFence> inFlightFences;
        std::vector<VkFence> imagesInFlight;
        std::vector<VkVertexInputAttributeDescription> vertexAttributes;

        vulkan_configs::SurfaceInfo surfaceInfo;
        vulkan_configs::QueueFamiliesBundle queueFamilies;
        vulkan_configs::QueuesBundle queues;
        vulkan_configs::DrawConfig drawConfig;
        vulkan_configs::PipelineConfig basePipelineConfig;
        vulkan_configs::PipelineBinding pipelineBinding;
        vulkan_configs::Pipeline pipeline;
        std::vector<VkPipelineShaderStageCreateInfo> stagesConfigs;
        size_t currentFrame = 0;

        std::span<vulkan_data::GPU_Vertex> verticesBinded;
        std::span<uint16_t> indicesBinded;

        const float DEFAULT_QUEUE_PRIORITY = 1.0f;
        const size_t MAX_FRAMES_IN_FLIGHT = 2;
        const size_t VERTEX_BUFFER_SIZE;
        const size_t INDEX_BUFFER_SIZE;

        std::set<vulkan_data::RenderObject> objects;
        std::span<char> vertexShader;
        std::span<char> fragmentShader;

        void InitVulkan();
        void SetupDevice();
        void SetupSwapchain();
        void RecreateSwapchain();
        void ReleaseSwapchain();
        void SetupRenderPasses();
        void SetupPipelineConfig();
        void SetupPipeline();
        void SetupObjectsBuffers();
        void SetupRenderBuffers();
        void SetupSync();

        void UpdateUniformBuffer(uint32_t currentImage);
        void CopyBuffer(VkBuffer src, size_t offset, VkBuffer dst, VkDeviceSize size);
    };
}