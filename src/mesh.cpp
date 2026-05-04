#include "mesh.h"

#include <iostream>
#include <queue>
#include <limits>
#include <algorithm>
#include <functional>
#include <cmath>

#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>
#include <opencv2/opencv.hpp>

namespace {

float safeLength(const glm::vec3& v) {
    return glm::length(v);
}

int nearestVertexToPoint(const std::vector<Vertex>& verts, const glm::vec3& p) {
    if (verts.empty()) return -1;
    int bestIdx = 0;
    float bestD2 = std::numeric_limits<float>::max();
    for (int i = 0; i < (int)verts.size(); ++i) {
        glm::vec3 d = verts[i].position - p;
        float d2 = glm::dot(d, d);
        if (d2 < bestD2) {
            bestD2 = d2;
            bestIdx = i;
        }
    }
    return bestIdx;
}

std::vector<float> dijkstraDistances(
    const std::vector<Vertex>& verts,
    const std::vector<std::vector<int>>& adj,
    int seed)
{
    const float INF = std::numeric_limits<float>::infinity();
    std::vector<float> dist(verts.size(), INF);
    if (seed < 0 || seed >= (int)verts.size()) return dist;

    using Node = std::pair<float, int>;
    std::priority_queue<Node, std::vector<Node>, std::greater<Node>> pq;

    dist[seed] = 0.0f;
    pq.push({0.0f, seed});

    while (!pq.empty()) {
        auto item = pq.top();
        pq.pop();
        float d = item.first;
        int u = item.second;
        if (d > dist[u]) continue;

        const glm::vec3 pu = verts[u].position;
        for (int v : adj[u]) {
            const glm::vec3 pv = verts[v].position;
            float w = safeLength(pv - pu);
            float nd = d + w;
            if (nd < dist[v]) {
                dist[v] = nd;
                pq.push({nd, v});
            }
        }
    }
    return dist;
}

void appendUnique(std::vector<int>& out, int v) {
    if (std::find(out.begin(), out.end(), v) == out.end()) {
        out.push_back(v);
    }
}

float distancePointToSegment(const glm::vec3& p, const glm::vec3& a, const glm::vec3& b, float* outT = nullptr) {
    glm::vec3 ab = b - a;
    float denom = glm::dot(ab, ab);
    float t = 0.0f;
    if (denom > 1e-8f) {
        t = glm::clamp(glm::dot(p - a, ab) / denom, 0.0f, 1.0f);
    }
    if (outT) *outT = t;
    glm::vec3 q = a + t * ab;
    return glm::length(p - q);
}

struct ArmCorridorEval {
    bool inside = false;
    bool upper = false;
    bool lower = false;
    float along = 0.0f;
    float radial = 0.0f;
};

ArmCorridorEval evalArmCorridor(
    const glm::vec3& p,
    const glm::vec3& shoulder,
    const glm::vec3& elbow,
    const glm::vec3& wrist,
    float shoulderRadius,
    float elbowRadius,
    float wristRadius,
    float sideSign)
{
    ArmCorridorEval out;

    float tUpper = 0.0f;
    float dUpper = distancePointToSegment(p, shoulder, elbow, &tUpper);
    float allowedUpper = glm::mix(shoulderRadius, elbowRadius, tUpper);

    float tLower = 0.0f;
    float dLower = distancePointToSegment(p, elbow, wrist, &tLower);
    float allowedLower = glm::mix(elbowRadius, wristRadius, tLower);

    const bool upperOk = dUpper <= allowedUpper;
    const bool lowerOk = dLower <= allowedLower;

    // Side gate so the corridor does not cross through the body.
    const float sideDelta = (p.x - shoulder.x) * sideSign;
    const bool sideOk = sideDelta >= -0.03f;

    // Vertices far below the shoulder/armpit line are body, not sleeve.
    const float minArmY = std::min(shoulder.y, std::min(elbow.y, wrist.y));
    const float maxDrop = std::max(0.18f * glm::length(wrist - shoulder), 0.035f);
    const bool heightOk = p.y >= (minArmY - maxDrop);

    if (sideOk && heightOk && (upperOk || lowerOk)) {
        out.inside = true;
        out.upper = upperOk && (!lowerOk || dUpper <= dLower);
        out.lower = lowerOk && (!upperOk || dLower < dUpper);
        out.radial = std::min(dUpper, dLower);
        out.along = out.upper ? tUpper : (1.0f + tLower);
    }

    return out;
}

} // namespace

void Mesh::findShoulders() {
    // This is now only used as a fallback heuristic if calibration hasn't occurred.
    // Calibration via calibrateAnchorsFromPose() will overwrite these values.
    if (bindVertices.empty()) return;

    glm::vec3 minP = bindVertices.front().position;
    glm::vec3 maxP = bindVertices.front().position;

    for (const auto& v : bindVertices) {
        minP = glm::min(minP, v.position);
        maxP = glm::max(maxP, v.position);
    }

    const glm::vec3 size = maxP - minP;
    const float midY = minP.y + size.y * 0.72f;
    const float leftX = minP.x + size.x * 0.22f;
    const float rightX = minP.x + size.x * 0.78f;

    auto nearestToXZBand = [&](float targetX) {
        int bestIdx = 0;
        float bestScore = std::numeric_limits<float>::max();
        for (int i = 0; i < (int)bindVertices.size(); ++i) {
            const glm::vec3 p = bindVertices[i].position;
            float score = std::abs(p.x - targetX) + 1.8f * std::abs(p.y - midY);
            if (score < bestScore) {
                bestScore = score;
                bestIdx = i;
            }
        }
        return bindVertices[bestIdx].position;
    };

    leftShoulderPos = nearestToXZBand(leftX);
    rightShoulderPos = nearestToXZBand(rightX);

    // Initialize with conservative defaults - will be overwritten by calibration
    glm::vec3 shoulderSpan = rightShoulderPos - leftShoulderPos;
    float width = glm::max(glm::length(shoulderSpan), 1e-4f);

    leftElbowPos  = leftShoulderPos  + glm::vec3(-0.1f * width, -0.35f * width, 0.0f);
    rightElbowPos = rightShoulderPos + glm::vec3( 0.1f * width, -0.35f * width, 0.0f);
    leftWristPos  = leftShoulderPos  + glm::vec3(-0.50f * width, -0.75f * width, 0.0f);
    rightWristPos = rightShoulderPos + glm::vec3( 0.50f * width, -0.75f * width, 0.0f);
}

void Mesh::buildAdjacency() {
    adjacency.assign(bindVertices.size(), {});
    for (size_t i = 0; i + 2 < indices.size(); i += 3) {
        int a = (int)indices[i + 0];
        int b = (int)indices[i + 1];
        int c = (int)indices[i + 2];

        if (a < 0 || b < 0 || c < 0 ||
            a >= (int)bindVertices.size() ||
            b >= (int)bindVertices.size() ||
            c >= (int)bindVertices.size()) {
            continue;
        }

        appendUnique(adjacency[a], b); appendUnique(adjacency[a], c);
        appendUnique(adjacency[b], a); appendUnique(adjacency[b], c);
        appendUnique(adjacency[c], a); appendUnique(adjacency[c], b);
    }
}

void Mesh::computeRegionsFromAnchors() {
    vertexRegion.assign(bindVertices.size(), REGION_UNKNOWN);
    if (bindVertices.empty() || adjacency.empty()) return;

    const int ls = nearestVertexToPoint(bindVertices, leftShoulderPos);
    const int le = nearestVertexToPoint(bindVertices, leftElbowPos);
    const int lw = nearestVertexToPoint(bindVertices, leftWristPos);
    const int rs = nearestVertexToPoint(bindVertices, rightShoulderPos);
    const int re = nearestVertexToPoint(bindVertices, rightElbowPos);
    const int rw = nearestVertexToPoint(bindVertices, rightWristPos);

    std::vector<float> dLS = dijkstraDistances(bindVertices, adjacency, ls);
    std::vector<float> dLE = dijkstraDistances(bindVertices, adjacency, le);
    std::vector<float> dLW = dijkstraDistances(bindVertices, adjacency, lw);
    std::vector<float> dRS = dijkstraDistances(bindVertices, adjacency, rs);
    std::vector<float> dRE = dijkstraDistances(bindVertices, adjacency, re);
    std::vector<float> dRW = dijkstraDistances(bindVertices, adjacency, rw);

    const glm::vec3 torsoCenter = 0.5f * (leftShoulderPos + rightShoulderPos);
    const int torsoSeed = nearestVertexToPoint(bindVertices, torsoCenter);
    std::vector<float> dTorso = dijkstraDistances(bindVertices, adjacency, torsoSeed);

    const float shoulderWidth = glm::max(glm::length(rightShoulderPos - leftShoulderPos), 1e-4f);
const float shoulderRadius = 0.26f * shoulderWidth;
const float elbowRadius    = 0.20f * shoulderWidth;
const float wristRadius    = 0.16f * shoulderWidth;

    for (size_t i = 0; i < bindVertices.size(); ++i) {
        const glm::vec3 p = bindVertices[i].position;

        ArmCorridorEval leftEval = evalArmCorridor(
            p,
            leftShoulderPos,
            leftElbowPos,
            leftWristPos,
            shoulderRadius,
            elbowRadius,
            wristRadius,
            -1.0f);

        ArmCorridorEval rightEval = evalArmCorridor(
            p,
            rightShoulderPos,
            rightElbowPos,
            rightWristPos,
            shoulderRadius,
            elbowRadius,
            wristRadius,
            +1.0f);

        const float leftGraph = std::min(dLS[i], std::min(dLE[i], dLW[i]));
        const float rightGraph = std::min(dRS[i], std::min(dRE[i], dRW[i]));
        const float torsoGraph = dTorso[i];

        const bool leftCandidate  = leftEval.inside  && (leftGraph  < torsoGraph * 0.90f);
        const bool rightCandidate = rightEval.inside && (rightGraph < torsoGraph * 0.90f);

        if (leftCandidate && (!rightCandidate || leftGraph <= rightGraph)) {
            vertexRegion[i] = leftEval.lower ? REGION_LEFT_LOWER_SLEEVE : REGION_LEFT_UPPER_SLEEVE;
            continue;
        }

        if (rightCandidate) {
            vertexRegion[i] = rightEval.lower ? REGION_RIGHT_LOWER_SLEEVE : REGION_RIGHT_UPPER_SLEEVE;
            continue;
        }

        vertexRegion[i] = REGION_TORSO;
    }

    auto cleanRegion = [&](VertexRegion region, int s0, int s1, int s2) {
        std::vector<char> visited(bindVertices.size(), 0);
        std::queue<int> q;
        const int seeds[3] = {s0, s1, s2};
        for (int j = 0; j < 3; ++j) {
            int s = seeds[j];
            if (s >= 0 && s < (int)bindVertices.size() && vertexRegion[s] == region) {
                visited[s] = 1;
                q.push(s);
            }
        }

        while (!q.empty()) {
            int u = q.front();
            q.pop();
            for (int v : adjacency[u]) {
                if (!visited[v] && vertexRegion[v] == region) {
                    visited[v] = 1;
                    q.push(v);
                }
            }
        }

        for (size_t i = 0; i < bindVertices.size(); ++i) {
            if (vertexRegion[i] == region && !visited[i]) {
                vertexRegion[i] = REGION_TORSO;
            }
        }
    };

    cleanRegion(REGION_LEFT_UPPER_SLEEVE, ls, le, -1);
    cleanRegion(REGION_LEFT_LOWER_SLEEVE, le, lw, -1);
    cleanRegion(REGION_RIGHT_UPPER_SLEEVE, rs, re, -1);
    cleanRegion(REGION_RIGHT_LOWER_SLEEVE, re, rw, -1);
}

void Mesh::resetToBindPose() {
    vertices = bindVertices;
    displayListDirty = true;
}

void Mesh::uploadDeformedVertices() {
    displayListDirty = true;
}

void Mesh::draw() const {
    if (!isValid || vertices.empty() || indices.empty()) return;

    if (displayListId == 0) {
        displayListId = glGenLists(1);
        displayListDirty = true;
    }

    if (displayListDirty) {
        glNewList(displayListId, GL_COMPILE);
        glBegin(GL_TRIANGLES);
        for (size_t i = 0; i + 2 < indices.size(); i += 3) {
            for (int k = 0; k < 3; ++k) {
                unsigned int idx = indices[i + k];
                if (idx >= vertices.size()) continue;
                const Vertex& v = vertices[idx];
                glNormal3f(v.normal.x, v.normal.y, v.normal.z);
                glTexCoord2f(v.uv.x, v.uv.y);
                glVertex3f(v.position.x, v.position.y, v.position.z);
            }
        }
        glEnd();
        glEndList();
        displayListDirty = false;
    }

    glCallList(displayListId);
}

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
                const auto& uv = aMesh->HasTextureCoords(0) ? aMesh->mTextureCoords[0][v] : aiVector3D(0, 0, 0);

                Vertex vert;
                vert.position = glm::vec3(pos.x, pos.y, pos.z) * 0.1f;
                vert.normal = glm::vec3(norm.x, norm.y, norm.z);
                vert.uv = glm::vec2(uv.x, uv.y);

                mesh.bindVertices.push_back(vert);
                mesh.vertices.push_back(vert);
            }

            for (unsigned int f = 0; f < aMesh->mNumFaces; ++f) {
                const aiFace& face = aMesh->mFaces[f];
                if (face.mNumIndices == 3) {
                    mesh.indices.push_back(baseIdx + face.mIndices[0]);
                    mesh.indices.push_back(baseIdx + face.mIndices[1]);
                    mesh.indices.push_back(baseIdx + face.mIndices[2]);
                }
            }

            if (!mesh.hasTexture && scene->mNumMaterials > 0) {
                const aiMaterial* mat = scene->mMaterials[aMesh->mMaterialIndex];
                aiString texPath;
                if (mat->GetTexture(aiTextureType_DIFFUSE, 0, &texPath) == AI_SUCCESS) {
                    // Check if it's an embedded texture (GLB stores textures this way)
                    const aiTexture* embTex = scene->GetEmbeddedTexture(texPath.C_Str());
                    if (embTex) {
                        cv::Mat img;
                        if (embTex->mHeight == 0) {
                            // Compressed (PNG/JPG) — decode from memory
                            std::vector<uchar> buf(
                                (uchar*)embTex->pcData,
                                (uchar*)embTex->pcData + embTex->mWidth
                            );
                            img = cv::imdecode(buf, cv::IMREAD_COLOR);
                            cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
                        } else {
                            // Raw ARGB8888
                            img = cv::Mat(embTex->mHeight, embTex->mWidth, CV_8UC4, embTex->pcData);
                            cv::cvtColor(img, img, cv::COLOR_BGRA2RGBA);
                        }

                        if (!img.empty()) {
                            glGenTextures(1, &mesh.textureId);
                            glBindTexture(GL_TEXTURE_2D, mesh.textureId);
                            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
                            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
                            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
                            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
                            GLenum fmt = (img.channels() == 4) ? GL_RGBA : GL_RGB;
                            glTexImage2D(GL_TEXTURE_2D, 0, fmt,
                                        img.cols, img.rows, 0,
                                        fmt, GL_UNSIGNED_BYTE, img.data);
                            glBindTexture(GL_TEXTURE_2D, 0);
                            mesh.hasTexture = true;
                        }
                    }
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
        mesh.buildAdjacency();
        mesh.computeRegionsFromAnchors();
        mesh.resetToBindPose();
    }

    return mesh;
}
