#ifndef MESH_H
#define MESH_H

#include <vector>
#include <glm/glm.hpp>
#include <GLFW/glfw3.h>

struct Vertex {
    glm::vec3 position;
    glm::vec3 normal;
};

struct Mesh {
    std::vector<Vertex> vertices;
    std::vector<unsigned int> indices;
    bool isValid = false;
    glm::vec3 leftShoulderPos{0, 0, 0};
    glm::vec3 rightShoulderPos{0, 0, 0};

    void findShoulders() {
        if (vertices.empty()) return;

        float maxY = -1e9f;
        glm::vec3 minPos = vertices[0].position;
        glm::vec3 maxPos = vertices[0].position;

        for (const auto& v : vertices) {
            minPos = glm::min(minPos, v.position);
            maxPos = glm::max(maxPos, v.position);
            maxY = std::max(maxY, v.position.y);
        }

        float shoulderY = maxY - (maxPos.y - minPos.y) * 0.1f;
        leftShoulderPos = {maxPos.x, shoulderY, 0.0f};
        rightShoulderPos  = {minPos.x, shoulderY, 0.0f};
    }

    void draw() {
        if (!isValid || vertices.empty() || indices.empty()) return;

        glEnableClientState(GL_VERTEX_ARRAY);
        glEnableClientState(GL_NORMAL_ARRAY);

        glVertexPointer(3, GL_FLOAT, sizeof(Vertex), &vertices[0].position);
        glNormalPointer(GL_FLOAT, sizeof(Vertex), &vertices[0].normal);
        glDrawElements(GL_TRIANGLES, (GLsizei)indices.size(), GL_UNSIGNED_INT, indices.data());

        glDisableClientState(GL_NORMAL_ARRAY);
        glDisableClientState(GL_VERTEX_ARRAY);
    }
};

#endif // MESH_H
