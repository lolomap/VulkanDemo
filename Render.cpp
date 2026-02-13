module;

#include <vector>
#include <array>
#include <span>
#include <set>
#include <cstdint>
#include <cstddef>
#include <algorithm>
#include <stdexcept>
#include <chrono>

#define GLM_FORCE_RADIANS
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#define VALIDATION_LAYER "VK_LAYER_KHRONOS_validation"

#define GPU_DISCRETE_PRIORITY 1000
#define GPU_GRAPHICS_WITH_PRESENT_PRIORITY 500
#define GPU_B8G8R8_SRGB_PRIORITY 10

#define QUEUE_FAMILY_NO_VALUE UINT32_MAX
#define ATTRIBUTE_FORMAT_VEC3 VK_FORMAT_R32G32B32_SFLOAT

// I don't want to bring P7Log library from my main project to this demo
#include <cstdarg>
#include <cstdio>
#define LOG(LEVEL, MESSAGE, ...) println(MESSAGE, __VA_ARGS__)

module Render;

void println(const char* msg, ...)
{
    va_list args;
    va_start(args, msg);
    vprintf(msg, args);
    va_end(args);
    printf("\n");
}

vulkan_render::Renderer::Renderer(GLFWwindow *window, const size_t &vertexBufferSize, const size_t &indexBufferSize,
                                  std::span<char> vertexShader, std::span<char> fragmentShader) :
        VERTEX_BUFFER_SIZE(vertexBufferSize),
        INDEX_BUFFER_SIZE(indexBufferSize)
{
    this->window = window;
    this->vertexShader = vertexShader;
    this->fragmentShader = fragmentShader;
    glfwGetFramebufferSize(window, &screenWidth, &screenHeight);

    InitVulkan();
    SetupDevice();
    SetupSwapchain();
    SetupRenderPasses();

    SetupPipelineConfig();
    SetupPipeline();
    SetupObjectsBuffers();
    SetupRenderBuffers();
    SetupDescriptorPool();
    SetupDescriptorSets();
    CopyBuffer(stagingVertices, 0, vertices, verticesBinded.size() * sizeof(vulkan_data::GPU_Vertex));
    CopyBuffer(stagingIndices, 0, indices, indicesBinded.size() * sizeof(uint16_t));
    SetupSync();
}

void vulkan_render::Renderer::InitVulkan()
{
    std::vector<const char*> vulkanExtensions {
        //
    };
    std::vector<const char*> vulkanLayers {
        VALIDATION_LAYER
    };

    // Get Vulkan extensions list
    uint32_t extensionsGLFWCount = 0;
    const char **glfwExtensions = glfwGetRequiredInstanceExtensions(&extensionsGLFWCount);
    for (size_t i = 0; i < extensionsGLFWCount; i++)
        vulkanExtensions.push_back(glfwExtensions[i]);

    for (const char *extension : vulkanExtensions)
    {
        LOG(INFO, "%hs Vulkan extension registered", extension);
    }

    for (const char *layer : vulkanLayers)
    {
        LOG(INFO, "%hs Vulkan layer registered", layer);
    }

    // Create VkInstance
    VkApplicationInfo vkAppConfig {
            .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
            .pNext = nullptr,
            .pApplicationName = "Etredia Empires",
            .applicationVersion = VK_MAKE_VERSION(1, 0, 0),
            .pEngineName = "No Engine",
            .engineVersion = VK_MAKE_VERSION(1, 0, 0),
            .apiVersion = VK_API_VERSION_1_0
    };

    VkInstanceCreateInfo vkCreateConfig {
            .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
            .pNext = nullptr,
            .pApplicationInfo = &vkAppConfig,
            .enabledLayerCount = static_cast<uint32_t>(vulkanLayers.size()),
            .ppEnabledLayerNames = vulkanLayers.data(),
            .enabledExtensionCount = static_cast<uint32_t>(vulkanExtensions.size()),
            .ppEnabledExtensionNames = vulkanExtensions.data(),
    };

    VkResult vkResult = vkCreateInstance(&vkCreateConfig, nullptr, &vulkan);
    if (vkResult != VK_SUCCESS)
        LOG(CRITICAL, "Vulkan is not instantiated! Code: %d", vkResult);
    else LOG(INFO, "Vulkan is instantiated");

    // Create surface

    vkResult = glfwCreateWindowSurface(vulkan, window, nullptr, &surface);
    if (vkResult != VK_SUCCESS)
        LOG(CRITICAL, "Failed to create GLFW Vulkan surface. Code: %d", vkResult);
    else LOG(INFO, "GLFW Vulkan surface created");
}

void vulkan_render::Renderer::SetupDevice()
{
    // Find physical devices
    VkResult vkResult;
    uint32_t devicesCount = 0;
    vkResult = vkEnumeratePhysicalDevices(vulkan, &devicesCount, nullptr);
    if (vkResult != VK_SUCCESS)
    {
        LOG(CRITICAL, "Failed to enumerate physical devices. Code: %d", vkResult);
        return;
    }

    if (devicesCount == 0)
    {
        LOG(CRITICAL, "No devices with Vulkan support found!");
        return;
    }

    std::vector<VkPhysicalDevice> devices(devicesCount);
    vkResult = vkEnumeratePhysicalDevices(vulkan, &devicesCount, devices.data());
    if (vkResult != VK_SUCCESS)
    {
        LOG(CRITICAL, "Failed to enumerate %d physical devices. Code: %d", devicesCount, vkResult);
        return;
    }
    else
    {
        LOG(INFO, "Found %zu Vulkan compatible physical devices.", devices.size());
    }

    // Filter and select device
    gpu = VK_NULL_HANDLE;
    VkPhysicalDeviceProperties deviceProperties {};
    VkPhysicalDeviceFeatures deviceFeatures {};
    VkPhysicalDeviceFeatures enabledFeatures {};
    uint32_t best_priority = 0;
    for (const VkPhysicalDevice &physicalDevice : devices)
    {
        uint32_t priority = 1;

        // Check device features and properties
        vkGetPhysicalDeviceFeatures(physicalDevice, &deviceFeatures);

        if (deviceFeatures.wideLines) enabledFeatures.wideLines = VK_TRUE; else continue;
        if (deviceFeatures.independentBlend) enabledFeatures.independentBlend = VK_TRUE; else continue;

        vkGetPhysicalDeviceProperties(physicalDevice, &deviceProperties);

        // Check queue families
        uint32_t queueFamiliesCount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(
                physicalDevice,
                &queueFamiliesCount,
                nullptr
        );
        std::vector<VkQueueFamilyProperties> queueFamiliesProperties(queueFamiliesCount);
        vkGetPhysicalDeviceQueueFamilyProperties(
                physicalDevice,
                &queueFamiliesCount,
                queueFamiliesProperties.data()
        );

        uint32_t i = 0;
        vulkan_configs::QueueFamiliesBundle queueFamiliesBuffer {};
        for (const VkQueueFamilyProperties &familyProps : queueFamiliesProperties)
        {
            if (familyProps.queueFlags & VK_QUEUE_GRAPHICS_BIT)
                queueFamiliesBuffer.graphics = i;

            if (
                    queueFamiliesBuffer.graphics != queueFamiliesBuffer.present ||
                    queueFamiliesBuffer.graphics == QUEUE_FAMILY_NO_VALUE
                    )
            {
                VkBool32 presentSupport = VK_FALSE;
                vkResult = vkGetPhysicalDeviceSurfaceSupportKHR(physicalDevice, i, surface, &presentSupport);
                if (vkResult != VK_SUCCESS)
                    LOG(WARNING, "Failed to check surface present support. Code: %d. Device: %hs", vkResult, deviceProperties.deviceName);
                if (presentSupport)
                    queueFamiliesBuffer.present = i;
            }

            i++;
        }

        if (
                queueFamiliesBuffer.graphics == QUEUE_FAMILY_NO_VALUE ||
                queueFamiliesBuffer.present == QUEUE_FAMILY_NO_VALUE
                )
            continue;
        queueFamiliesBuffer.transfer = queueFamiliesBuffer.graphics;

        // Check surface info
        vulkan_configs::SurfaceInfo surfaceInfoBuffer {};

        uint32_t surfaceFormatsCount = 0;
        vkResult = vkGetPhysicalDeviceSurfaceFormatsKHR(
                physicalDevice,
                surface,
                &surfaceFormatsCount,
                nullptr
        );
        if (vkResult != VK_SUCCESS)
            LOG(WARNING, "Failed to get surface formats number. Code: %d. Device: %hs", vkResult, deviceProperties.deviceName);
        surfaceInfoBuffer.availableFormats.resize(surfaceFormatsCount);
        vkResult = vkGetPhysicalDeviceSurfaceFormatsKHR(
                physicalDevice,
                surface,
                &surfaceFormatsCount,
                surfaceInfoBuffer.availableFormats.data()
        );
        if (vkResult != VK_SUCCESS)
            LOG(WARNING, "Failed to get surface formats. Code: %d. Device: %hs", vkResult, deviceProperties.deviceName);

        uint32_t presentModesCount = 0;
        vkResult = vkGetPhysicalDeviceSurfacePresentModesKHR(
                physicalDevice,
                surface,
                &presentModesCount,
                nullptr
        );
        if (vkResult != VK_SUCCESS)
            LOG(WARNING, "Failed to get surface present modes number. Code: %d. Device: %hs", vkResult, deviceProperties.deviceName);
        surfaceInfoBuffer.availablePresentModes.resize(presentModesCount);
        vkResult = vkGetPhysicalDeviceSurfacePresentModesKHR(
                physicalDevice,
                surface,
                &surfaceFormatsCount,
                surfaceInfoBuffer.availablePresentModes.data()
        );
        if (vkResult != VK_SUCCESS)
            LOG(WARNING, "Failed to get surface present modes. Code: %d. Device: %hs", vkResult, deviceProperties.deviceName);

        if (
                surfaceInfoBuffer.availableFormats.empty() ||
                surfaceInfoBuffer.availablePresentModes.empty()
                )
            continue;

        // Rate and select physical device
        if (queueFamiliesBuffer.graphics == queueFamiliesBuffer.present)
            priority += GPU_GRAPHICS_WITH_PRESENT_PRIORITY;

        if (deviceProperties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU)
            priority += GPU_DISCRETE_PRIORITY;

        surfaceInfoBuffer.format.format = VK_FORMAT_B8G8R8A8_UNORM;
        for (const VkSurfaceFormatKHR &format : surfaceInfoBuffer.availableFormats)
        {
            surfaceInfoBuffer.format = format;
            if (format.format == VK_FORMAT_B8G8R8A8_SRGB && format.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR)
            {
                priority += GPU_B8G8R8_SRGB_PRIORITY;
                break;
            }
        }

        surfaceInfoBuffer.presentMode = VK_PRESENT_MODE_FIFO_KHR;
        for (const VkPresentModeKHR &presentMode : surfaceInfoBuffer.availablePresentModes)
        {
            if (presentMode == VK_PRESENT_MODE_FIFO_RELAXED_KHR)
            {
                surfaceInfoBuffer.presentMode = presentMode;
                break;
            }
        }

        if (priority > best_priority)
        {
            best_priority = priority;
            gpu = physicalDevice;
            queueFamilies = queueFamiliesBuffer;
            surfaceInfo = surfaceInfoBuffer;
        }
    }

    if (gpu == VK_NULL_HANDLE)
    {
        LOG(CRITICAL, "Failed to select suitable physical device");
        return;
    }

    // Define queues
    std::vector<VkDeviceQueueCreateInfo> queuesCreateConfigs;
    std::set<uint32_t> uniqueQueueFamilies {
            queueFamilies.transfer,
            queueFamilies.graphics,
            queueFamilies.present
    };

    for (uint32_t queueFamily : uniqueQueueFamilies)
    {
        VkDeviceQueueCreateInfo queueCreateConfig {
                .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
                .pNext = nullptr,
                .queueFamilyIndex = queueFamily,
                .queueCount = 1,
                .pQueuePriorities = &DEFAULT_QUEUE_PRIORITY
        };

        queuesCreateConfigs.push_back(queueCreateConfig);
    }

    // Create logical device
    std::vector<const char*> deviceExtensions {
            VK_KHR_SWAPCHAIN_EXTENSION_NAME
    };

    VkDeviceCreateInfo deviceInfo {
            .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
            .pNext = nullptr,
            .queueCreateInfoCount = static_cast<uint32_t>(queuesCreateConfigs.size()),
            .pQueueCreateInfos = queuesCreateConfigs.data(),
            .enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size()),
            .ppEnabledExtensionNames = deviceExtensions.data(),
            .pEnabledFeatures = &enabledFeatures,
    };

    vkResult = vkCreateDevice(gpu, &deviceInfo, nullptr, &device);
    if (vkResult != VK_SUCCESS)
    {
        LOG(CRITICAL, "Failed to create logical device on %hs. Code: %d", deviceProperties.deviceName, vkResult);
        return;
    }
    else LOG(INFO, "%hs is selected for render with %d priority score", deviceProperties.deviceName, best_priority);

    // Save queues descriptor
    vkGetDeviceQueue(device, queueFamilies.transfer, 0, &queues.transfer);
    vkGetDeviceQueue(device, queueFamilies.graphics, 0, &queues.graphics);
    vkGetDeviceQueue(device, queueFamilies.present, 0, &queues.present);
}

void vulkan_render::Renderer::SetupSwapchain()
{
    VkResult vkResult;
    vkResult = vkGetPhysicalDeviceSurfaceCapabilitiesKHR(gpu, surface, &surfaceInfo.capabilities);
    if (vkResult != VK_SUCCESS)
        LOG(CRITICAL, "Failed to get surface capabilities. Code: %d", vkResult);

    int width = 0, height = 0;
    glfwGetFramebufferSize(window, &width, &height);

    // Clamp extent
    surfaceInfo.extent.width = (std::max)(
            surfaceInfo.capabilities.minImageExtent.width,
            (std::min)(surfaceInfo.capabilities.maxImageExtent.width, static_cast<uint32_t>(width))
    );
    surfaceInfo.extent.height = (std::max)(
            surfaceInfo.capabilities.minImageExtent.height,
            (std::min)(surfaceInfo.capabilities.maxImageExtent.height, static_cast<uint32_t>(height))
    );

    uint32_t imagesCount = surfaceInfo.capabilities.minImageCount + 1;
    // Clamp min images count
    if (surfaceInfo.capabilities.maxImageCount > 0 && imagesCount > surfaceInfo.capabilities.maxImageCount)
        imagesCount = surfaceInfo.capabilities.maxImageCount;

    VkSwapchainCreateInfoKHR swapchainCreateConfig {
            .sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
            .pNext = nullptr,
            .surface = surface,
            .minImageCount = imagesCount,
            .imageFormat = surfaceInfo.format.format,
            .imageColorSpace = surfaceInfo.format.colorSpace,
            .imageExtent = surfaceInfo.extent,
            .imageArrayLayers = 1,
            .imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
            .preTransform = surfaceInfo.capabilities.currentTransform,
            .compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
            .presentMode = surfaceInfo.presentMode,
            .clipped = VK_TRUE,
            .oldSwapchain = VK_NULL_HANDLE
    };

    if (queueFamilies.graphics != queueFamilies.present)
    {
        uint32_t queueFamiliesShare[] {queueFamilies.graphics, queueFamilies.present};
        swapchainCreateConfig.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
        swapchainCreateConfig.queueFamilyIndexCount = 2;
        swapchainCreateConfig.pQueueFamilyIndices = queueFamiliesShare;
    }
    else
        swapchainCreateConfig.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;

    vkResult = vkCreateSwapchainKHR(device, &swapchainCreateConfig, nullptr, &swapchain);
    if (vkResult != VK_SUCCESS)
        LOG(CRITICAL, "Failed to create swapchain. Code: %d", vkResult);
    else LOG(INFO, "Swapchain is created");

    vkResult = vkGetSwapchainImagesKHR(device, swapchain, &imagesCount, nullptr);
    if (vkResult != VK_SUCCESS)
        LOG(CRITICAL, "Failed to get frames number. Code: %d", vkResult);
    frames.resize(imagesCount);
    vkResult = vkGetSwapchainImagesKHR(device, swapchain, &imagesCount, frames.data());
    if (vkResult != VK_SUCCESS)
        LOG(CRITICAL, "Failed to get frames descriptors. Code: %d", vkResult);
    else LOG(INFO, "Frames are ready");

    frameViews.resize(imagesCount);
    for (size_t i = 0; i < frames.size(); i++)
    {
        VkImageViewCreateInfo frameViewConfig {
                .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
                .pNext = nullptr,
                .image = frames[i],
                .viewType = VK_IMAGE_VIEW_TYPE_2D,
                .format = surfaceInfo.format.format,
                .components {
                        .r = VK_COMPONENT_SWIZZLE_IDENTITY,
                        .g = VK_COMPONENT_SWIZZLE_IDENTITY,
                        .b = VK_COMPONENT_SWIZZLE_IDENTITY,
                        .a = VK_COMPONENT_SWIZZLE_IDENTITY
                },
                .subresourceRange {
                        .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                        .baseMipLevel = 0,
                        .levelCount = 1,
                        .baseArrayLayer = 0,
                        .layerCount = 1
                }
        };

        vkResult = vkCreateImageView(device, &frameViewConfig, nullptr, &frameViews[i]);
        if (vkResult != VK_SUCCESS)
            LOG(CRITICAL, "Failed to create image view for frame. Code: %d", vkResult);
    }
    LOG(INFO, "Frame views are ready.");
}

void vulkan_render::Renderer::RecreateSwapchain()
{
    // We should not render with minimized window
    if (screenWidth == 0 || screenHeight == 0)
    {
        isDrawing = false;
        return;
    }
    else isDrawing = true;

    vkDeviceWaitIdle(device);
    ReleaseSwapchain();

    SetupSwapchain();
    SetupRenderPasses();
    SetupPipeline();
    SetupRenderBuffers();
    SetupDescriptorPool();
    SetupDescriptorSets();
}

void vulkan_render::Renderer::ReleaseSwapchain()
{
    for (VkFramebuffer &frameBuffer : baseFrameBuffers)
    {
        vkDestroyFramebuffer(device, frameBuffer, nullptr);
    }
    vkFreeCommandBuffers(
            device,
            graphicsCommandPool,
            static_cast<uint32_t>(graphicsCommandBuffers.size()),
            graphicsCommandBuffers.data()
    );
    vkFreeCommandBuffers(
            device,
            transferCommandPool,
            static_cast<uint32_t>(transferCommandBuffers.size()),
            transferCommandBuffers.data()
    );

    vkDestroyRenderPass(device, baseRenderPass, nullptr);
    vkDestroyPipeline(device, pipeline.instance, nullptr);
    vkDestroyPipelineLayout(device, pipeline.layout, nullptr);
    vkDestroyDescriptorSetLayout(device, pipeline.uniformsLayout, nullptr);

    for (VkImageView &frameView : frameViews)
    {
        vkDestroyImageView(device, frameView, nullptr);
    }

    vkDestroySwapchainKHR(device, swapchain, nullptr);

    for (size_t i = 0; i < frames.size(); i++)
    {
        vkDestroyBuffer(device, vpUniforms[i], nullptr);
        vkFreeMemory(device, vpUniformsMemory[i], nullptr);
        vkDestroyBuffer(device, modelUniforms[i], nullptr);
        vkFreeMemory(device, modelUniformsMemory[i], nullptr);
    }

    vkDestroyDescriptorPool(device, descriptors.pool, nullptr);
}

void vulkan_render::Renderer::SetupPipelineConfig()
{
    vertexAttributes = vulkan_data::GPU_Vertex::GetAttributes();

    pipelineBinding = {
        .baseRenderPass = &baseRenderPass,
        .device = &device,
        .surfaceExtent = &surfaceInfo.extent,

        .vertexBindingConfig = &drawConfig.vertexBindingConfig,
        .vertexAttributesCount = static_cast<uint32_t>(vertexAttributes.size()),
        .vertexAttributesConfig = vertexAttributes.data()
    };

    basePipelineConfig.dynamicStates.push_back(VK_DYNAMIC_STATE_VIEWPORT);
    basePipelineConfig.dynamicStates.push_back(VK_DYNAMIC_STATE_SCISSOR);

    basePipelineConfig.colorBlendAttachmentConfigs.push_back({
        .blendEnable = VK_TRUE,
        .srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA,
        .dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,
        .colorBlendOp = VK_BLEND_OP_ADD,
        .srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE,
        .dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO,
        .alphaBlendOp = VK_BLEND_OP_ADD,
        .colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT
    });

    basePipelineConfig.uniforms.push_back({
        .binding = 0,
        .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
        .descriptorCount = 1,
        .stageFlags = VK_SHADER_STAGE_VERTEX_BIT,
        .pImmutableSamplers = nullptr
    }); // GPU_GlobalUBO

    basePipelineConfig.uniforms.push_back({
        .binding = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
        .descriptorCount = 1,
        .stageFlags = VK_SHADER_STAGE_VERTEX_BIT,
        .pImmutableSamplers = nullptr
    }); // GPU_TransformUBO

    basePipelineConfig.vertexInputConfig = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
        .pNext = nullptr,
        .vertexBindingDescriptionCount = 1,
        .pVertexBindingDescriptions = pipelineBinding.vertexBindingConfig,
        .vertexAttributeDescriptionCount = pipelineBinding.vertexAttributesCount,
        .pVertexAttributeDescriptions = pipelineBinding.vertexAttributesConfig
    };

    basePipelineConfig.inputAssemblyConfig = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
        .pNext = nullptr,
        .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
        .primitiveRestartEnable = VK_FALSE
    };

    basePipelineConfig.rasterizerConfig = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
        .pNext = nullptr,
        .depthClampEnable = VK_FALSE,
        .rasterizerDiscardEnable = VK_FALSE,
        .polygonMode = VK_POLYGON_MODE_FILL,
        .cullMode = VK_CULL_MODE_BACK_BIT,
        .frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE,
        .depthBiasEnable = VK_FALSE,
        .lineWidth = 1.0f,
    };

    basePipelineConfig.multisamplingConfig = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
        .pNext = nullptr,
        .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
        .sampleShadingEnable = VK_FALSE,
    };

    basePipelineConfig.depthStencilConfig = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
        .pNext = nullptr,
    };

    basePipelineConfig.dynamicStateConfig = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
        .pNext = nullptr,
        .dynamicStateCount = static_cast<uint32_t>(basePipelineConfig.dynamicStates.size()),
        .pDynamicStates = basePipelineConfig.dynamicStates.data()
    };

    basePipelineConfig.uniformsConfig = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = static_cast<uint32_t>(basePipelineConfig.uniforms.size()),
        .pBindings = basePipelineConfig.uniforms.data()
    };

    basePipelineConfig.pipelineLayoutConfig = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .pNext = nullptr,
        .setLayoutCount = 1,
        .pushConstantRangeCount = 0,
        .pPushConstantRanges = nullptr
    };

    basePipelineConfig.defaultViewport = {
        .x = 0.0f,
        .y = 0.0f,
        .width = static_cast<float>(pipelineBinding.surfaceExtent->width),
        .height = static_cast<float>(pipelineBinding.surfaceExtent->height),
        .minDepth = 0.0f,
        .maxDepth = 1.0f
    };

    basePipelineConfig.defaultScissors = {
        .offset = {.x = 0, .y = 0},
        .extent = *pipelineBinding.surfaceExtent
    };

    basePipelineConfig.viewportStateConfig = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
        .pNext = nullptr,
        .viewportCount = 1,
        .pViewports = &basePipelineConfig.defaultViewport,
        .scissorCount = 1,
        .pScissors = &basePipelineConfig.defaultScissors
    };

    basePipelineConfig.colorBlendConfig = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
        .pNext = nullptr,
        .logicOpEnable = VK_FALSE,
        .attachmentCount = static_cast<uint32_t>(basePipelineConfig.colorBlendAttachmentConfigs.size()),
        .pAttachments = basePipelineConfig.colorBlendAttachmentConfigs.data(),
    };

    basePipelineConfig.pipelineConfig = {
        .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
        .pNext = nullptr,
        .pVertexInputState = &basePipelineConfig.vertexInputConfig,
        .pInputAssemblyState = &basePipelineConfig.inputAssemblyConfig,
        .pViewportState = &basePipelineConfig.viewportStateConfig,
        .pRasterizationState = &basePipelineConfig.rasterizerConfig,
        .pMultisampleState = &basePipelineConfig.multisamplingConfig,
        .pDepthStencilState = nullptr,
        .pColorBlendState = &basePipelineConfig.colorBlendConfig,
        .pDynamicState = &basePipelineConfig.dynamicStateConfig,
        .basePipelineHandle = VK_NULL_HANDLE,
        .basePipelineIndex = -1
    };

    LOG(INFO, "Pipeline config constructed");
}

void vulkan_render::Renderer::SetupPipeline()
{
    VkResult vkResult;

    // Configure shaders
    VkShaderModuleCreateInfo shaderModuleCreateConfig;
    stagesConfigs.reserve(3);

    // Vertex
    VkShaderModule vertexModule;
    shaderModuleCreateConfig = {
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .pNext = nullptr,
        .codeSize = vertexShader.size(),
        .pCode = reinterpret_cast<const uint32_t*>(vertexShader.data())
    };
    vkResult = vkCreateShaderModule(device, &shaderModuleCreateConfig, nullptr, &vertexModule);
    if (vkResult != VK_SUCCESS)
        LOG(ERROR, "Failed to create vertex shader module. Code: %d", vkResult);
    stagesConfigs.push_back({
        .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .pNext = nullptr,
        .stage = VK_SHADER_STAGE_VERTEX_BIT,
        .module = vertexModule,
        .pName = "main",
        .pSpecializationInfo = nullptr
    });

    // Fragment
    VkShaderModule fragmentModule;
    shaderModuleCreateConfig = {
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .pNext = nullptr,
        .codeSize = fragmentShader.size(),
        .pCode = reinterpret_cast<const uint32_t*>(fragmentShader.data())
    };
    vkResult = vkCreateShaderModule(device, &shaderModuleCreateConfig, nullptr, &fragmentModule);
    if (vkResult != VK_SUCCESS)
        LOG(ERROR, "Failed to create fragment shader module. Code: %d", vkResult);
    stagesConfigs.push_back({
        .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .pNext = nullptr,
        .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
        .module = fragmentModule,
        .pName = "main",
        .pSpecializationInfo = nullptr
    });

    //

    vkResult = vkCreateDescriptorSetLayout(device, &basePipelineConfig.uniformsConfig,
                                           nullptr, &pipeline.uniformsLayout);
    if (vkResult != VK_SUCCESS)
        LOG(CRITICAL, "Failed to create uniforms layout");
    basePipelineConfig.pipelineLayoutConfig.pSetLayouts = &pipeline.uniformsLayout;

    vkResult = vkCreatePipelineLayout(
        device,
        &basePipelineConfig.pipelineLayoutConfig,
        nullptr,
        &pipeline.layout
    );
    if (vkResult != VK_SUCCESS)
        LOG(CRITICAL, "Failed to create pipeline layout");
    basePipelineConfig.pipelineConfig.layout = pipeline.layout;

    basePipelineConfig.pipelineConfig.stageCount = static_cast<uint32_t>(stagesConfigs.size());
    basePipelineConfig.pipelineConfig.pStages = stagesConfigs.data();
    basePipelineConfig.pipelineConfig.renderPass = baseRenderPass;
    basePipelineConfig.pipelineConfig.subpass = 0;

    vkResult = vkCreateGraphicsPipelines(
            device,
            VK_NULL_HANDLE,
            1,
            &basePipelineConfig.pipelineConfig,
            nullptr,
            &pipeline.instance
    );
    if (vkResult != VK_SUCCESS)
        LOG(CRITICAL, "Failed to create pipeline");

    pipeline.viewport = basePipelineConfig.defaultViewport;
    pipeline.scissors = basePipelineConfig.defaultScissors;

    vkDestroyShaderModule(device, vertexModule, nullptr);
    vkDestroyShaderModule(device, fragmentModule, nullptr);

    LOG(INFO, "Pipeline created");
}

void vulkan_render::Renderer::SetupRenderPasses()
{
    VkAttachmentDescription colorAttachmentConfig {
            .format = surfaceInfo.format.format,
            .samples = VK_SAMPLE_COUNT_1_BIT, // No multisampling
            .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
            .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
            .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
            .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
            .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
            .finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR
    };

    VkAttachmentReference colorAttachmentRef {
            .attachment = 0,
            .layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL
    };

    VkSubpassDescription subpass {
            .pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
            .colorAttachmentCount = 1,
            .pColorAttachments = &colorAttachmentRef
    };

    VkSubpassDependency dependency {
            .srcSubpass = VK_SUBPASS_EXTERNAL,
            .dstSubpass = 0,
            .srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
            .dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
            .srcAccessMask = 0,
            .dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
    };

    VkRenderPassCreateInfo baseRenderPassConfig {
            .sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
            .pNext = nullptr,
            .attachmentCount = 1,
            .pAttachments = &colorAttachmentConfig,
            .subpassCount = 1,
            .pSubpasses = &subpass,
            .dependencyCount = 1,
            .pDependencies = &dependency
    };


    VkResult vkResult = vkCreateRenderPass(device, &baseRenderPassConfig, nullptr, &baseRenderPass);
    if (vkResult != VK_SUCCESS)
        LOG(CRITICAL, "Failed to create base render pass. Code: %d", vkResult);
}

void vulkan_render::Renderer::SetupObjectsBuffers()
{
    VkResult vkResult;

    VkBufferCreateInfo vertexBufferConfig {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .pNext = nullptr,
        .size = sizeof(vulkan_data::GPU_Vertex) * VERTEX_BUFFER_SIZE,
        .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE
    };
    vkResult = vkCreateBuffer(device, &vertexBufferConfig, nullptr, &vertices);
    if (vkResult != VK_SUCCESS)
        LOG(CRITICAL, "Failed to create buffer for vertices. Code: %d", vkResult);

    VkBufferCreateInfo indexBufferConfig {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .pNext = nullptr,
        .size = sizeof(uint16_t ) * INDEX_BUFFER_SIZE,
        .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE
    };
    vkResult = vkCreateBuffer(device, &indexBufferConfig, nullptr, &indices);
    if (vkResult != VK_SUCCESS)
        LOG(CRITICAL, "Failed to create buffer for indices. Code: %d", vkResult);

    VkBufferCreateInfo stagingVertexBufferConfig {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .pNext = nullptr,
        .size = sizeof(vulkan_data::GPU_Vertex) * VERTEX_BUFFER_SIZE,
        .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE
    };
    vkResult = vkCreateBuffer(device, &stagingVertexBufferConfig, nullptr, &stagingVertices);
    if (vkResult != VK_SUCCESS)
        LOG(CRITICAL, "Failed to create buffer for staging vertices. Code: %d", vkResult);

    VkBufferCreateInfo stagingIndexBufferConfig {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .pNext = nullptr,
        .size = sizeof(uint16_t ) * INDEX_BUFFER_SIZE,
        .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE
    };
    vkResult = vkCreateBuffer(device, &stagingIndexBufferConfig, nullptr, &stagingIndices);
    if (vkResult != VK_SUCCESS)
        LOG(CRITICAL, "Failed to create buffer for staging indices. Code: %d", vkResult);

    VkMemoryRequirements memoryRequirements;
    vkGetBufferMemoryRequirements(device, vertices, &memoryRequirements);
    VkMemoryRequirements stagingMemoryRequirements;
    vkGetBufferMemoryRequirements(device, stagingVertices, &stagingMemoryRequirements);

    VkMemoryRequirements idMemoryRequirements;
    vkGetBufferMemoryRequirements(device, indices, &idMemoryRequirements);
    VkMemoryRequirements stagingIdMemoryRequirements;
    vkGetBufferMemoryRequirements(device, stagingIndices, &stagingIdMemoryRequirements);

    VkPhysicalDeviceMemoryProperties memoryProps;
    vkGetPhysicalDeviceMemoryProperties(gpu, &memoryProps);
    uint32_t optimizedMemTypeIndex = UINT32_MAX, cpuAccessMemTypeIndex = UINT32_MAX;
    VkMemoryPropertyFlags optimizedMemoryProps = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    VkMemoryPropertyFlags cpuAccessMemoryProps = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    for (uint32_t i = 0; i < memoryProps.memoryTypeCount; i++)
    {
        if (
            optimizedMemTypeIndex == UINT32_MAX &&
            memoryRequirements.memoryTypeBits & (1 << i) &&
            (memoryProps.memoryTypes[i].propertyFlags & optimizedMemoryProps) == optimizedMemoryProps
        )
            optimizedMemTypeIndex = i;
        if (
            cpuAccessMemTypeIndex == UINT32_MAX &&
            stagingMemoryRequirements.memoryTypeBits & (1 << i) &&
            (memoryProps.memoryTypes[i].propertyFlags & cpuAccessMemoryProps) == cpuAccessMemoryProps
        )
            cpuAccessMemTypeIndex = i;
    }
    if (optimizedMemTypeIndex == UINT32_MAX)
        LOG(CRITICAL, "Failed to find GPU memory available for local data. Code: %d", vkResult);
    if (cpuAccessMemTypeIndex == UINT32_MAX)
        LOG(CRITICAL, "Failed to find GPU memory available for transfering data. Code: %d", vkResult);

    VkMemoryAllocateInfo allocateVertices {
            .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            .pNext = nullptr,
            .allocationSize = memoryRequirements.size,
            .memoryTypeIndex = optimizedMemTypeIndex
    };
    VkMemoryAllocateInfo allocateIndices {
            .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            .pNext = nullptr,
            .allocationSize = idMemoryRequirements.size,
            .memoryTypeIndex = optimizedMemTypeIndex
    };
    VkMemoryAllocateInfo allocateStagingVertices {
            .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            .pNext = nullptr,
            .allocationSize = stagingMemoryRequirements.size,
            .memoryTypeIndex = cpuAccessMemTypeIndex
    };
    VkMemoryAllocateInfo allocateStagingIndices {
            .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            .pNext = nullptr,
            .allocationSize = stagingIdMemoryRequirements.size,
            .memoryTypeIndex = cpuAccessMemTypeIndex
    };

    vkResult = vkAllocateMemory(device, &allocateVertices, nullptr, &verticesMemory);
    if (vkResult != VK_SUCCESS)
        LOG(CRITICAL, "Failed to allocate GPU memory for vertices. Code: %d", vkResult);
    vkResult = vkAllocateMemory(
            device, &allocateStagingVertices, nullptr, &stagingVerticesMemory);
    if (vkResult != VK_SUCCESS)
        LOG(CRITICAL, "Failed to allocate GPU memory for staging vertices. Code: %d", vkResult);

    vkResult = vkAllocateMemory(device, &allocateIndices, nullptr, &indicesMemory);
    if (vkResult != VK_SUCCESS)
        LOG(CRITICAL, "Failed to allocate GPU memory for indices. Code: %d", vkResult);
    vkResult = vkAllocateMemory(
            device, &allocateStagingIndices, nullptr, &stagingIndicesMemory);
    if (vkResult != VK_SUCCESS)
        LOG(CRITICAL, "Failed to allocate GPU memory for staging indices. Code: %d", vkResult);

    vkResult = vkBindBufferMemory(device, vertices, verticesMemory, 0);
    if (vkResult != VK_SUCCESS)
        LOG(CRITICAL, "Failed to bind GPU memory for vertices. Code: %d", vkResult);
    vkResult = vkBindBufferMemory(device, stagingVertices, stagingVerticesMemory, 0);
    if (vkResult != VK_SUCCESS)
        LOG(CRITICAL, "Failed to bind GPU memory for staging vertices. Code: %d", vkResult);

    vkResult = vkBindBufferMemory(device, indices, indicesMemory, 0);
    if (vkResult != VK_SUCCESS)
        LOG(CRITICAL, "Failed to bind GPU memory for indices. Code: %d", vkResult);
    vkResult = vkBindBufferMemory(device, stagingIndices, stagingIndicesMemory, 0);
    if (vkResult != VK_SUCCESS)
        LOG(CRITICAL, "Failed to bind GPU memory for staging indices. Code: %d", vkResult);

    vulkan_data::GPU_Vertex *ptrVerticesBinded = nullptr;
    vkResult = vkMapMemory(
            device,
            stagingVerticesMemory,
            0,
            stagingVertexBufferConfig.size,
            0,
            reinterpret_cast<void**>(&ptrVerticesBinded)
    );
    if (vkResult != VK_SUCCESS)
        LOG(CRITICAL, "Failed to map staging vertices buffer to CPU");
    verticesBinded = {ptrVerticesBinded, VERTEX_BUFFER_SIZE};

    uint16_t *ptrIndicesBinded = nullptr;
    vkResult = vkMapMemory(
            device,
            stagingIndicesMemory,
            0,
            stagingIndexBufferConfig.size,
            0,
            reinterpret_cast<void**>(&ptrIndicesBinded)
    );
    if (vkResult != VK_SUCCESS)
        LOG(CRITICAL, "Failed to map staging indices buffer to CPU");
    indicesBinded = {ptrIndicesBinded, INDEX_BUFFER_SIZE};

    memset(verticesBinded.data(), 0, stagingVertexBufferConfig.size);
    memset(indicesBinded.data(), 0, stagingIndexBufferConfig.size);

    vpUniforms.resize(frames.size());
    vpUniformsMemory.resize(frames.size());
    for (size_t i = 0; i < frames.size(); i++)
    {
        VkBufferCreateInfo uniformBufferConfig {
                .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                .pNext = nullptr,
                .size = sizeof(vulkan_data::GPU_GlobalUBO),
                .usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                .sharingMode = VK_SHARING_MODE_EXCLUSIVE
        };
        vkResult = vkCreateBuffer(device, &uniformBufferConfig, nullptr, &vpUniforms[i]);
        if (vkResult != VK_SUCCESS)
            LOG(CRITICAL, "Failed to create buffer for uniform. Code: %d", vkResult);

        VkMemoryRequirements ubMemoryRequirements;
        vkGetBufferMemoryRequirements(device, vpUniforms[i], &ubMemoryRequirements);
        VkMemoryAllocateInfo allocateUniform {
                .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                .pNext = nullptr,
                .allocationSize = ubMemoryRequirements.size,
                .memoryTypeIndex = cpuAccessMemTypeIndex
        };

        vkResult = vkAllocateMemory(device, &allocateUniform, nullptr, &vpUniformsMemory[i]);
        if (vkResult != VK_SUCCESS)
            LOG(CRITICAL, "Failed to allocate GPU memory for uniform. Code: %d", vkResult);

        vkResult = vkBindBufferMemory(device, vpUniforms[i], vpUniformsMemory[i], 0);
        if (vkResult != VK_SUCCESS)
            LOG(CRITICAL, "Failed to bind GPU memory for uniform. Code: %d", vkResult);
    }

    modelUniforms.resize(frames.size());
    modelUniformsMemory.resize(frames.size());
    for (size_t i = 0; i < frames.size(); i++)
    {
        VkBufferCreateInfo uniformBufferConfig {
                .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                .pNext = nullptr,
                .size = sizeof(vulkan_data::GPU_TranformUBO),
                .usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                .sharingMode = VK_SHARING_MODE_EXCLUSIVE
        };
        vkResult = vkCreateBuffer(device, &uniformBufferConfig, nullptr, &modelUniforms[i]);
        if (vkResult != VK_SUCCESS)
            LOG(CRITICAL, "Failed to create buffer for uniform. Code: %d", vkResult);

        VkMemoryRequirements ubMemoryRequirements;
        vkGetBufferMemoryRequirements(device, modelUniforms[i], &ubMemoryRequirements);
        VkMemoryAllocateInfo allocateUniform {
                .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                .pNext = nullptr,
                .allocationSize = ubMemoryRequirements.size,
                .memoryTypeIndex = cpuAccessMemTypeIndex
        };

        vkResult = vkAllocateMemory(device, &allocateUniform, nullptr, &modelUniformsMemory[i]);
        if (vkResult != VK_SUCCESS)
            LOG(CRITICAL, "Failed to allocate GPU memory for uniform. Code: %d", vkResult);

        vkResult = vkBindBufferMemory(device, modelUniforms[i], modelUniformsMemory[i], 0);
        if (vkResult != VK_SUCCESS)
            LOG(CRITICAL, "Failed to bind GPU memory for uniform. Code: %d", vkResult);
    }

}

void vulkan_render::Renderer::SetupRenderBuffers()
{
    VkResult vkResult;

    baseFrameBuffers.resize(frameViews.size());
    size_t i = 0;
    for (const VkImageView &frameView : frameViews)
    {
        VkImageView attachments[] = {frameView};

        VkFramebufferCreateInfo frameBufferConfig {
            .sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
            .pNext = nullptr,
            .renderPass = baseRenderPass,
            .attachmentCount = 1,
            .pAttachments = attachments,
            .width = surfaceInfo.extent.width,
            .height = surfaceInfo.extent.height,
            .layers = 1
        };

        vkResult = vkCreateFramebuffer(device, &frameBufferConfig, nullptr, &baseFrameBuffers[i]);
        if (vkResult != VK_SUCCESS)
            LOG(CRITICAL, "Failed to create framebuffer for base render pass. Code: %d", vkResult);

        i++;
    }

    if (graphicsCommandPool == nullptr)
    {
        VkCommandPoolCreateInfo commandPoolConfig {
            .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
            .pNext = nullptr,
            .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
            .queueFamilyIndex = queueFamilies.graphics,
        };

        vkResult = vkCreateCommandPool(device, &commandPoolConfig, nullptr, &graphicsCommandPool);
        if (vkResult != VK_SUCCESS)
            LOG(CRITICAL, "Failed to create graphics command pool. Code: %d", vkResult);
    }

    graphicsCommandBuffers.resize(MAX_FRAMES_IN_FLIGHT);
    VkCommandBufferAllocateInfo allocateConfig {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .pNext = nullptr,
        .commandPool = graphicsCommandPool,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, //TODO: reuse some commands using SECONDARY
        .commandBufferCount = static_cast<uint32_t>(graphicsCommandBuffers.size())
    };
    vkResult = vkAllocateCommandBuffers(device, &allocateConfig, graphicsCommandBuffers.data());
    if (vkResult != VK_SUCCESS)
        LOG(CRITICAL, "Failed to allocate graphics command buffers. Code: %d", vkResult);

    if (transferCommandPool == nullptr)
    {
        VkCommandPoolCreateInfo commandPoolConfig {
            .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
            .pNext = nullptr,
            .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT | VK_COMMAND_POOL_CREATE_TRANSIENT_BIT,
            .queueFamilyIndex = queueFamilies.transfer,
        };

        vkResult = vkCreateCommandPool(device, &commandPoolConfig, nullptr, &transferCommandPool);
        if (vkResult != VK_SUCCESS)
            LOG(CRITICAL, "Failed to create transfer command pool. Code: %d", vkResult);
    }

    transferCommandBuffers.resize(MAX_FRAMES_IN_FLIGHT);
    allocateConfig = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .pNext = nullptr,
        .commandPool = transferCommandPool,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, //TODO: reuse some commands using SECONDARY
        .commandBufferCount = static_cast<uint32_t>(transferCommandBuffers.size())
    };
    vkResult = vkAllocateCommandBuffers(device, &allocateConfig, transferCommandBuffers.data());
    if (vkResult != VK_SUCCESS)
        LOG(CRITICAL, "Failed to allocate transfer command buffers. Code: %d", vkResult);
}

void vulkan_render::Renderer::SetupDescriptorPool()
{
    // Hardcoded! Create pool for 2 uniforms: viewProject and model

    VkResult vkResult;

    // We need one descriptor for each uniform and each frame in total
    VkDescriptorPoolSize poolSize {
        .type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
        .descriptorCount = static_cast<uint32_t>(frames.size() * 2)
    };

    // We need pool with 1 set for each frame, each set has all uniform descriptors for frame
    VkDescriptorPoolCreateInfo poolConfig {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .pNext = nullptr,
        .maxSets = static_cast<uint32_t>(frames.size()),
        .poolSizeCount = 1,
        .pPoolSizes = &poolSize,
    };

    vkResult = vkCreateDescriptorPool(device, &poolConfig, nullptr, &descriptors.pool);
    if (vkResult != VK_SUCCESS)
        LOG(CRITICAL, "Failed to create descriptor pool. Code: %d", vkResult);
}

void vulkan_render::Renderer::SetupDescriptorSets()
{
    VkResult vkResult;

    std::vector<VkDescriptorSetLayout> layouts(frames.size(), pipeline.uniformsLayout);
    VkDescriptorSetAllocateInfo setAllocateConfig {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .pNext = nullptr,
        .descriptorPool = descriptors.pool,
        .descriptorSetCount = static_cast<uint32_t>(frames.size()),
        .pSetLayouts = layouts.data()
    };

    descriptors.descriptorSets.resize(frames.size());
    vkResult = vkAllocateDescriptorSets(device, &setAllocateConfig, descriptors.descriptorSets.data());
    if (vkResult != VK_SUCCESS)
        LOG(CRITICAL, "Failed to allocate descriptor sets. Code: %d", vkResult);

    for (size_t i = 0; i < frames.size(); i++)
    {
        VkDescriptorBufferInfo vpBufferConfig {
            .buffer = vpUniforms[i],
            .offset = 0,
            .range = sizeof(vulkan_data::GPU_GlobalUBO)
        };

        VkDescriptorBufferInfo modelBufferConfig {
            .buffer = modelUniforms[i],
            .offset = 0,
            .range = sizeof(vulkan_data::GPU_TranformUBO)
        };

        VkWriteDescriptorSet vpWrite {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .pNext = nullptr,
            .dstSet = descriptors.descriptorSets[i],
            .dstBinding = 0,
            .dstArrayElement = 0,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            .pBufferInfo = &vpBufferConfig
        };

        VkWriteDescriptorSet modelWrite {
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .pNext = nullptr,
                .dstSet = descriptors.descriptorSets[i],
                .dstBinding = 1,
                .dstArrayElement = 0,
                .descriptorCount = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                .pBufferInfo = &modelBufferConfig
        };

        std::array<VkWriteDescriptorSet, 2> writes = {vpWrite, modelWrite};
        vkUpdateDescriptorSets(device, 2, writes.data(), 0, nullptr);
    }
}

void vulkan_render::Renderer::SetupSync()
{
    VkResult vkResult;

    VkSemaphoreCreateInfo semaphoreConfig {
        .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
        .pNext = nullptr
    };
    VkFenceCreateInfo fenceConfig {
        .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
        .pNext = nullptr,
        .flags = VK_FENCE_CREATE_SIGNALED_BIT
    };

    imageAvailable.resize(MAX_FRAMES_IN_FLIGHT);
    renderFinished.resize(MAX_FRAMES_IN_FLIGHT);
    inFlightFences.resize(MAX_FRAMES_IN_FLIGHT);
    imagesInFlight.resize(frameViews.size(), VK_NULL_HANDLE);

    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
    {
        vkResult = vkCreateSemaphore(device, &semaphoreConfig, nullptr, &imageAvailable[i]);
        if (vkResult != VK_SUCCESS)
            LOG(CRITICAL, "Failed to create semaphore. Code: %d", vkResult);
        vkResult = vkCreateSemaphore(device, &semaphoreConfig, nullptr, &renderFinished[i]);
        if (vkResult != VK_SUCCESS)
            LOG(CRITICAL, "Failed to create semaphore. Code: %d", vkResult);
        vkResult = vkCreateFence(device, &fenceConfig, nullptr, &inFlightFences[i]);
        if (vkResult != VK_SUCCESS)
            LOG(CRITICAL, "Failed to create fence. Code: %d", vkResult);
    }
}

void vulkan_render::Renderer::CopyBuffer(VkBuffer src, size_t offset, VkBuffer dst, VkDeviceSize size)
{
    VkResult vkResult;

    VkCommandBufferAllocateInfo bufferAllocateConfig {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = transferCommandPool,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1,
    };

    VkCommandBuffer commandBuffer;
    vkResult = vkAllocateCommandBuffers(device, &bufferAllocateConfig, &commandBuffer);
    if (vkResult != VK_SUCCESS)
        LOG(ERROR, "Failed to allocate temporary command buffer for copy operation. Code: %d", vkResult);

    VkCommandBufferBeginInfo commandBeginConfig {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .pNext = nullptr,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT
    };
    vkResult = vkBeginCommandBuffer(commandBuffer, &commandBeginConfig);
    if (vkResult != VK_SUCCESS)
        LOG(ERROR, "Failed to begin command for copy operation. Code: %d", vkResult);

    VkBufferCopy copyRegion {
        .srcOffset = offset,
        .dstOffset = offset,
        .size = size
    };
    vkCmdCopyBuffer(commandBuffer, src, dst, 1, &copyRegion);

    vkResult = vkEndCommandBuffer(commandBuffer);
    if (vkResult != VK_SUCCESS)
        LOG(ERROR, "Failed to end command for copy operation. Code: %d", vkResult);

    VkSubmitInfo submitConfig {
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .pNext = nullptr,
        .commandBufferCount = 1,
        .pCommandBuffers = &commandBuffer
    };
    vkQueueSubmit(queues.transfer, 1, &submitConfig, VK_NULL_HANDLE); //TODO: use fence for optimizatoin
    vkQueueWaitIdle(queues.transfer);

    vkFreeCommandBuffers(device, transferCommandPool, 1, &commandBuffer);
}

void vulkan_render::Renderer::Draw()
{
    if (!isDrawing) return;

    VkResult vkResult;
    uint32_t imageIndex = 0;

    vkResult = vkWaitForFences(device, 1, &inFlightFences[currentFrame], VK_TRUE, UINT64_MAX);
    if (vkResult != VK_SUCCESS)
        LOG(ERROR, "Failed to wait for fence. Code: %d", vkResult);

    vkResult = vkAcquireNextImageKHR(
        device, swapchain,
        UINT64_MAX,
        imageAvailable[currentFrame],
        VK_NULL_HANDLE,
        &imageIndex
    );
    switch (vkResult)
    {
        case VK_SUCCESS:
            break;

        case VK_ERROR_OUT_OF_DATE_KHR:
        case VK_SUBOPTIMAL_KHR:
            LOG(WARNING, "Swapchain needs to be recreated");
            RecreateSwapchain();
            break;

        default:
            LOG(ERROR, "Failed to get next image index. Code: %d", vkResult);
            break;
    }

    UpdateUniformBuffer(imageIndex);

    if (imagesInFlight[imageIndex] != VK_NULL_HANDLE)
    {
        vkWaitForFences(device, 1, &imagesInFlight[imageIndex], VK_TRUE, UINT64_MAX);
    }
    imagesInFlight[imageIndex] = inFlightFences[currentFrame];

    vkResult = vkResetCommandBuffer(graphicsCommandBuffers[currentFrame], 0);
    if (vkResult != VK_SUCCESS)
        LOG(ERROR, "Failed to reset command buffer. Code: %d", vkResult);
    vkResult = vkBeginCommandBuffer(graphicsCommandBuffers[currentFrame], &drawConfig.commandBeginConfig);
    if (vkResult != VK_SUCCESS)
        LOG(ERROR, "Failed to begin command buffer. Code: %d", vkResult);

    drawConfig.renderPassBeginConfig.renderPass = baseRenderPass;
    drawConfig.renderPassBeginConfig.framebuffer = baseFrameBuffers[imageIndex];
    drawConfig.renderPassBeginConfig.renderArea = {
        .offset = {.x = 0, .y = 0},
        .extent = surfaceInfo.extent
    };

    vkCmdBeginRenderPass(
        graphicsCommandBuffers[currentFrame],
        &drawConfig.renderPassBeginConfig,
        VK_SUBPASS_CONTENTS_INLINE
    );

    vkCmdBindPipeline(
        graphicsCommandBuffers[currentFrame],
        VK_PIPELINE_BIND_POINT_GRAPHICS,
        pipeline.instance
    );
    vkCmdSetViewport(graphicsCommandBuffers[currentFrame], 0, 1, &pipeline.viewport);
    vkCmdSetScissor(graphicsCommandBuffers[currentFrame], 0, 1, &pipeline.scissors);

    for (const vulkan_data::RenderObject &object : objects)
    {
        std::vector<VkBuffer> vertexBuffers;
        VkBuffer indexBuffer;
        vertexBuffers.reserve(1);
        vertexBuffers.push_back(vertices);
        indexBuffer = indices;

        VkDeviceSize vertexOffsets[] = {object.mesh.verticesBufferPosition};
        VkDeviceSize indexOffset = object.mesh.indicesBufferPosition;
        vkCmdBindVertexBuffers(
            graphicsCommandBuffers[currentFrame],
            0,
            1,
            vertexBuffers.data(),
            vertexOffsets
        );
        vkCmdBindIndexBuffer(
            graphicsCommandBuffers[currentFrame],
            indexBuffer,
            indexOffset,
            VK_INDEX_TYPE_UINT16
        );

        vkCmdBindDescriptorSets(
            graphicsCommandBuffers[currentFrame],
            VK_PIPELINE_BIND_POINT_GRAPHICS,
            pipeline.layout,
            0,
            1,
            &descriptors.descriptorSets[currentFrame],
            0,
            nullptr
        );

        vkCmdDrawIndexed(
            graphicsCommandBuffers[currentFrame],
            static_cast<uint32_t>(object.mesh.indices.size()),
            1,
            0,
            0,
            0
        );
    }

    vkCmdEndRenderPass(graphicsCommandBuffers[currentFrame]);
    vkResult = vkEndCommandBuffer(graphicsCommandBuffers[currentFrame]);
    if (vkResult != VK_SUCCESS)
        LOG(ERROR, "Failed to end command buffer. Code: %d", vkResult);

    VkSemaphore waitSemaphores[] = {imageAvailable[currentFrame]};
    VkSemaphore signalSemaphores[] = {renderFinished[currentFrame]};
    VkPipelineStageFlags waitStages[] = {VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};
    drawConfig.submitConfig.pWaitSemaphores = waitSemaphores;
    drawConfig.submitConfig.pSignalSemaphores =signalSemaphores;
    drawConfig.submitConfig.pCommandBuffers = &graphicsCommandBuffers[currentFrame];
    drawConfig.submitConfig.pWaitDstStageMask = waitStages;

    vkResult = vkResetFences(device, 1, &inFlightFences[currentFrame]);
    if (vkResult != VK_SUCCESS)
        LOG(ERROR, "Failed to reset for fence. Code: %d", vkResult);

    vkResult = vkQueueSubmit(queues.graphics, 1, &drawConfig.submitConfig, inFlightFences[currentFrame]);
    if (vkResult != VK_SUCCESS)
        LOG(ERROR, "Failed to submit render commands. Code: %d", vkResult);

    drawConfig.presentConfig.pWaitSemaphores = signalSemaphores;
    VkSwapchainKHR swapChains[] = {swapchain};
    drawConfig.presentConfig.pSwapchains = swapChains;
    drawConfig.presentConfig.pImageIndices = &imageIndex;
    vkResult = vkQueuePresentKHR(queues.present, &drawConfig.presentConfig);
    switch (vkResult)
    {
        case VK_SUCCESS:
            break;

        case VK_ERROR_OUT_OF_DATE_KHR:
        case VK_SUBOPTIMAL_KHR:
            LOG(WARNING, "Swapchain needs to be recreated");
            RecreateSwapchain();
            break;

        default:
            LOG(ERROR, "Failed to present swapchain. Code: %d", vkResult);
            break;
    }

    currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
}

// Rotate camera and objects right here to simplify
void vulkan_render::Renderer::UpdateUniformBuffer(uint32_t currentImage)
{
    // Calculate deltaTime to rotate without depending on FPS
    static auto startTime = std::chrono::high_resolution_clock::now();
    auto currentTime = std::chrono::high_resolution_clock::now();
    float deltaTime = std::chrono::duration<float, std::chrono::seconds::period>(currentTime - startTime).count();

    vulkan_data::GPU_GlobalUBO camera;
    vulkan_data::GPU_TranformUBO transform;

    // Rotate objects by 90 degrees per second
    transform.model = glm::rotate(glm::mat4(1.0f), deltaTime * glm::radians(90.0f), glm::vec3(0.0f, 0.0f, 1.0f));
    // Rotate camera by 45 defrees upside-down
    camera.view = glm::lookAt(glm::vec3(2.0f, 2.0f, 2.0f), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f));
    // Perspective projection, vertical FOV is 45 degrees
    camera.proj = glm::perspective(glm::radians(45.0f), surfaceInfo.extent.width / (float) surfaceInfo.extent.height, 0.1f, 10.0f);
    camera.proj[1][1] *= -1; // GLM uses reversed Y like in OpenGL. Fix it for Vulkan

    void* data;
    vkMapMemory(device, vpUniformsMemory[currentImage], 0, sizeof(camera), 0, &data);
    memcpy(data, &camera, sizeof(camera));
    vkUnmapMemory(device, vpUniformsMemory[currentImage]);

    vkMapMemory(device, modelUniformsMemory[currentImage], 0, sizeof(transform), 0, &data);
    memcpy(data, &transform, sizeof(transform));
    vkUnmapMemory(device, modelUniformsMemory[currentImage]);
}

void vulkan_render::Renderer::RegisterObjects(std::span<vulkan_data::RenderObject> newObjects)
{
    vulkan_data::GPU_Vertex *verticesStart = verticesBinded.data();
    uint16_t *indicesStart = indicesBinded.data();

    for (size_t i = 0; i < newObjects.size(); i++)
    {
        vulkan_data::RenderObject &object = newObjects[i];
        object.engineId = static_cast<uint32_t>(objects.size());
        if (objects.contains(object)) continue;

        memcpy(verticesStart, object.mesh.vertices.data(), object.mesh.vertices.size() * sizeof(vulkan_data::GPU_Vertex));
        object.mesh.verticesBufferPosition = i * object.mesh.vertices.size();
        verticesStart += object.mesh.vertices.size();
        CopyBuffer(
                stagingVertices,
                object.mesh.verticesBufferPosition,
                vertices,
                object.mesh.vertices.size() * sizeof(vulkan_data::GPU_Vertex)
        );

        memcpy(indicesStart, object.mesh.indices.data(), object.mesh.indices.size() * sizeof(uint16_t ));
        object.mesh.indicesBufferPosition = i * object.mesh.indices.size();
        indicesStart += object.mesh.indices.size();
        CopyBuffer(
                stagingIndices,
                object.mesh.indicesBufferPosition,
                indices,
                object.mesh.indices.size() * sizeof(uint16_t)
        );

        objects.insert(object);
    }
}

void vulkan_render::Renderer::ResizeWindow(int width, int height)
{
    screenWidth = width;
    screenHeight = height;
    RecreateSwapchain();
}

vulkan_render::Renderer::~Renderer()
{
    vkDeviceWaitIdle(device);

    ReleaseSwapchain();

    vkUnmapMemory(device, stagingVerticesMemory);

    vkDestroyBuffer(device, vertices, nullptr);
    vkDestroyBuffer(device, stagingVertices, nullptr);

    vkDestroyBuffer(device, indices, nullptr);
    vkDestroyBuffer(device, stagingIndices, nullptr);

    vkFreeMemory(device, verticesMemory, nullptr);
    vkFreeMemory(device, stagingVerticesMemory, nullptr);

    vkFreeMemory(device, indicesMemory, nullptr);
    vkFreeMemory(device, stagingIndicesMemory, nullptr);

    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
    {
        vkDestroySemaphore(device, renderFinished[i], nullptr);
        vkDestroySemaphore(device, imageAvailable[i], nullptr);
        vkDestroyFence(device, inFlightFences[i], nullptr);
    }

    vkDestroyCommandPool(device, graphicsCommandPool, nullptr);
    vkDestroyCommandPool(device, transferCommandPool, nullptr);

    vkDestroyDevice(device, nullptr);
    vkDestroySurfaceKHR(vulkan, surface, nullptr);
    vkDestroyInstance(vulkan, nullptr);

    LOG(INFO, "Renderer is released");
}