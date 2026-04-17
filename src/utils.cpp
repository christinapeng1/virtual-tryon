#include "utils.h"
#include <iostream>
#include <functional>
#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

Mesh loadMesh(const std::string& path) {
    Mesh mesh;

    Assimp::Importer importer;
    const aiScene* scene = importer.ReadFile(
        path,
        aiProcess_Triangulate |
        aiProcess_GenSmoothNormals |
        aiProcess_JoinIdenticalVertices
    );

    if (!scene || !scene->mRootNode) {
        std::cerr << "Failed to load mesh: " << importer.GetErrorString() << "\n";
        return mesh;
    }

    std::function<void(aiNode*)> processMeshes = [&](aiNode* node) {
        for (unsigned int i = 0; i < node->mNumMeshes; ++i) {
            const aiMesh* aMesh = scene->mMeshes[node->mMeshes[i]];
            unsigned int baseIdx = (unsigned int)mesh.vertices.size();

            for (unsigned int v = 0; v < aMesh->mNumVertices; ++v) {
                const auto& pos = aMesh->mVertices[v];
                const auto& norm = aMesh->HasNormals() ? aMesh->mNormals[v] : aiVector3D(0, 0, 1);

                mesh.vertices.push_back({
                    glm::vec3(pos.x, pos.y, pos.z) * 0.1f,
                    glm::vec3(norm.x, norm.y, norm.z)
                });
            }

            for (unsigned int f = 0; f < aMesh->mNumFaces; ++f) {
                const aiFace& face = aMesh->mFaces[f];
                if (face.mNumIndices == 3) {
                    mesh.indices.push_back(baseIdx + face.mIndices[0]);
                    mesh.indices.push_back(baseIdx + face.mIndices[1]);
                    mesh.indices.push_back(baseIdx + face.mIndices[2]);
                }
            }
        }

        for (unsigned int i = 0; i < node->mNumChildren; ++i) {
            processMeshes(node->mChildren[i]);
        }
    };

    processMeshes(scene->mRootNode);
    mesh.isValid = !mesh.vertices.empty() && !mesh.indices.empty();

    if (mesh.isValid) {
        mesh.findShoulders();
    }

    return mesh;
}

glm::vec3 screenToWorld(glm::vec2 screenPos, float depth, float aspect) {
    glm::vec2 ndc = screenPos * 2.0f - 1.0f;
    ndc.y = -ndc.y;

    const float fovY = glm::radians(45.0f);
    float tanHalfFov = glm::tan(fovY * 0.5f);

    float z = depth; // should be negative
    float x = ndc.x * std::abs(z) * tanHalfFov * aspect;
    float y = ndc.y * std::abs(z) * tanHalfFov;

    return glm::vec3(x, y, z);
}

void drawSphere(glm::vec3 pos, float radius) {
    glPushMatrix();
    glTranslatef(pos.x, pos.y, pos.z);

    int stacks = 8, slices = 8;
    for (int i = 0; i < stacks; i++) {
        float phi1 = (float)M_PI * (-0.5f + (float)i / (float)stacks);
        float phi2 = (float)M_PI * (-0.5f + (float)(i + 1) / (float)stacks);

        float y1 = std::sin(phi1);
        float r1 = std::cos(phi1);
        float y2 = std::sin(phi2);
        float r2 = std::cos(phi2);

        glBegin(GL_QUAD_STRIP);
        for (int j = 0; j <= slices; j++) {
            float theta = 2.0f * (float)M_PI * (float)j / (float)slices;
            float x = std::cos(theta);
            float z = std::sin(theta);

            glNormal3f(r1 * x, y1, r1 * z);
            glVertex3f(radius * r1 * x, radius * y1, radius * r1 * z);

            glNormal3f(r2 * x, y2, r2 * z);
            glVertex3f(radius * r2 * x, radius * y2, radius * r2 * z);
        }
        glEnd();
    }

    glPopMatrix();
}
