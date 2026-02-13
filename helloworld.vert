#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(binding = 0) uniform ViewProjectinoUBO {
    mat4 view;
    mat4 proj;
} vp;

layout(binding = 1) uniform ModelUBO {
    mat4 model;
} model;

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inColor;

layout(location = 0) out vec3 fragColor;

void main() {
     gl_Position = vp.proj * vp.view * model.model * vec4(inPosition, 1.0);
    fragColor = inColor;
}