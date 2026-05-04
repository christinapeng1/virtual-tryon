
#ifndef MESH_H
#define MESH_H

#include <vector>
#include <string>
#include <glm/glm.hpp>

#ifdef __APPLE__
#include <OpenGL/gl.h>
#else
#include <GL/gl.h>
#endif

struct Vertex {
    glm::vec3 position{0.0f};
    glm::vec3 normal{0.0f, 0.0f, 1.0f};
    glm::vec2 uv{0.0f};
};

enum VertexRegion {
    REGION_UNKNOWN = 0,
    REGION_TORSO,
    REGION_LEFT_UPPER_SLEEVE,
    REGION_LEFT_LOWER_SLEEVE,
    REGION_RIGHT_UPPER_SLEEVE,
    REGION_RIGHT_LOWER_SLEEVE
};

struct Mesh {
    bool isValid = false;

    std::vector<Vertex> bindVertices;
    std::vector<Vertex> vertices;
    std::vector<unsigned int> indices;

    std::vector<std::vector<int>> adjacency;
    std::vector<int> vertexRegion;

    glm::vec3 leftShoulderPos{-0.25f, 0.20f, 0.0f};
    glm::vec3 rightShoulderPos{ 0.25f, 0.20f, 0.0f};
    glm::vec3 leftElbowPos{-0.45f, 0.05f, 0.0f};
    glm::vec3 rightElbowPos{ 0.45f, 0.05f, 0.0f};
    glm::vec3 leftWristPos{-0.58f, -0.08f, 0.0f};
    glm::vec3 rightWristPos{ 0.58f, -0.08f, 0.0f};

    GLuint textureId = 0;
    bool hasTexture = false;
    glm::vec4 materialColor{1.0f, 1.0f, 1.0f, 1.0f};
    bool hasMaterialColor = false;

    void findShoulders();
    void buildAdjacency();
    void computeRegionsFromAnchors();

    void resetToBindPose();
    void uploadDeformedVertices();
    void draw() const;

private:
    mutable GLuint displayListId = 0;
    mutable bool displayListDirty = true;
};

Mesh loadMesh(const std::string& path);

#endif
