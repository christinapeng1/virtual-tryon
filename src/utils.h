#ifndef UTILS_H
#define UTILS_H

#include "mesh.h"
#include <glm/glm.hpp>
#include <string>
#include <cmath>
#include <GLFW/glfw3.h>

// Load mesh from GLB file
Mesh loadMesh(const std::string& path);

// Convert normalized screen point [0..1] to world position
glm::vec3 screenToWorld(glm::vec2 screenPos, float depth, float aspect);

// Draw a sphere at position
void drawSphere(glm::vec3 pos, float radius = 0.06f);

#endif // UTILS_H
