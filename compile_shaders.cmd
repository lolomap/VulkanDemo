@echo off

glslc helloworld.vert -o vert.spv
glslc helloworld.frag -o frag.spv

echo Shaders compiled