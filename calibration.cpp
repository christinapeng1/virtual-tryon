#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <limits>
#include <algorithm>
#include <cmath>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <GLFW/glfw3.h>

#include "mesh.h"

struct CalibrationData {
    int leftShoulder = -1;
    int leftElbow = -1;
    int leftWrist = -1;
    int rightShoulder = -1;
    int rightElbow = -1;
    int rightWrist = -1;
    std::vector<int> vertexRegion;
};

struct OrbitCamera {
    float yaw = 0.0f;
    float pitch = 0.15f;
    float distance = 2.2f;
    glm::vec3 target = glm::vec3(0.0f);

    glm::mat4 view() const {
        glm::vec3 eye;
        eye.x = target.x + distance * std::cos(pitch) * std::sin(yaw);
        eye.y = target.y + distance * std::sin(pitch);
        eye.z = target.z + distance * std::cos(pitch) * std::cos(yaw);
        return glm::lookAt(eye, target, glm::vec3(0, 1, 0));
    }
};

namespace CalibrationRegion {
    enum Region {
        REGION_UNKNOWN = 0,
        REGION_TORSO = 1,
        REGION_LEFT_UPPER_SLEEVE = 2,
        REGION_LEFT_LOWER_SLEEVE = 3,
        REGION_RIGHT_UPPER_SLEEVE = 4,
        REGION_RIGHT_LOWER_SLEEVE = 5
    };
}

static const char* kStageNames[6] = {
    "Click LEFT SHOULDER seam",
    "Click LEFT ELBOW area",
    "Click LEFT WRIST / cuff area",
    "Click RIGHT SHOULDER seam",
    "Click RIGHT ELBOW area",
    "Click RIGHT WRIST / cuff area"
};

static std::vector<std::vector<int>> buildAdjacency(const Mesh& mesh) {
    std::vector<std::vector<int>> adj(mesh.bindVertices.size());
    auto addUnique = [](std::vector<int>& list, int v) {
        if (std::find(list.begin(), list.end(), v) == list.end()) list.push_back(v);
    };

    for (size_t i = 0; i + 2 < mesh.indices.size(); i += 3) {
        int a = (int)mesh.indices[i + 0];
        int b = (int)mesh.indices[i + 1];
        int c = (int)mesh.indices[i + 2];
        addUnique(adj[a], b); addUnique(adj[a], c);
        addUnique(adj[b], a); addUnique(adj[b], c);
        addUnique(adj[c], a); addUnique(adj[c], b);
    }
    return adj;
}

static std::vector<float> dijkstra(const Mesh& mesh, const std::vector<std::vector<int>>& adj, int seed) {
    const float INF = std::numeric_limits<float>::infinity();
    std::vector<float> dist(mesh.bindVertices.size(), INF);
    if (seed < 0 || seed >= (int)mesh.bindVertices.size()) return dist;

    using Node = std::pair<float, int>;
    std::priority_queue<Node, std::vector<Node>, std::greater<Node>> pq;
    dist[seed] = 0.0f;
    pq.push({0.0f, seed});

    while (!pq.empty()) {
        auto [d, u] = pq.top();
        pq.pop();
        if (d > dist[u]) continue;
        glm::vec3 pu = mesh.bindVertices[u].position;
        for (int v : adj[u]) {
            glm::vec3 pv = mesh.bindVertices[v].position;
            float nd = d + glm::length(pv - pu);
            if (nd < dist[v]) {
                dist[v] = nd;
                pq.push({nd, v});
            }
        }
    }
    return dist;
}

static float distancePointToSegment(const glm::vec3& p, const glm::vec3& a, const glm::vec3& b) {
    glm::vec3 ab = b - a;
    float denom = glm::dot(ab, ab);
    float t = 0.0f;
    if (denom > 1e-8f) t = glm::clamp(glm::dot(p - a, ab) / denom, 0.0f, 1.0f);
    glm::vec3 q = a + t * ab;
    return glm::length(p - q);
}

static CalibrationData computeCalibration(const Mesh& mesh,
                                          int ls, int le, int lw,
                                          int rs, int re, int rw) {
    CalibrationData out;
    out.leftShoulder = ls; out.leftElbow = le; out.leftWrist = lw;
    out.rightShoulder = rs; out.rightElbow = re; out.rightWrist = rw;
    out.vertexRegion.assign(mesh.bindVertices.size(), REGION_TORSO);

    auto adj = buildAdjacency(mesh);
    auto dLS = dijkstra(mesh, adj, ls);
    auto dLE = dijkstra(mesh, adj, le);
    auto dLW = dijkstra(mesh, adj, lw);
    auto dRS = dijkstra(mesh, adj, rs);
    auto dRE = dijkstra(mesh, adj, re);
    auto dRW = dijkstra(mesh, adj, rw);

    const glm::vec3 pLS = mesh.bindVertices[ls].position;
    const glm::vec3 pLE = mesh.bindVertices[le].position;
    const glm::vec3 pLW = mesh.bindVertices[lw].position;
    const glm::vec3 pRS = mesh.bindVertices[rs].position;
    const glm::vec3 pRE = mesh.bindVertices[re].position;
    const glm::vec3 pRW = mesh.bindVertices[rw].position;
    const float shoulderWidth = glm::length(pRS - pLS);
    const float leftRadius = std::max(0.28f * shoulderWidth, 0.04f);
    const float rightRadius = leftRadius;

    for (size_t i = 0; i < mesh.bindVertices.size(); ++i) {
        const glm::vec3 p = mesh.bindVertices[i].position;

        bool leftCorridor =
            distancePointToSegment(p, pLS, pLE) <= leftRadius ||
            distancePointToSegment(p, pLE, pLW) <= leftRadius * 1.6f;
        bool rightCorridor =
            distancePointToSegment(p, pRS, pRE) <= rightRadius ||
            distancePointToSegment(p, pRE, pRW) <= rightRadius * 1.6f;

        float leftGraph = std::min({dLS[i], dLE[i], dLW[i]});
        float rightGraph = std::min({dRS[i], dRE[i], dRW[i]});

        if (leftCorridor && (!rightCorridor || leftGraph <= rightGraph)) {
            out.vertexRegion[i] = (dLE[i] <= dLW[i]) ? REGION_LEFT_UPPER_SLEEVE : REGION_LEFT_LOWER_SLEEVE;
        } else if (rightCorridor) {
            out.vertexRegion[i] = (dRE[i] <= dRW[i]) ? REGION_RIGHT_UPPER_SLEEVE : REGION_RIGHT_LOWER_SLEEVE;
        }
    }

    return out;
}

static bool saveCalibrationJson(const std::string& path, const Mesh& mesh, const CalibrationData& c) {
    std::ofstream out(path);
    if (!out) return false;

    auto P = [&](int idx) { return mesh.bindVertices[idx].position; };

    out << "{\n";
    out << "  \"leftShoulderVertex\": " << c.leftShoulder << ",\n";
    out << "  \"leftElbowVertex\": " << c.leftElbow << ",\n";
    out << "  \"leftWristVertex\": " << c.leftWrist << ",\n";
    out << "  \"rightShoulderVertex\": " << c.rightShoulder << ",\n";
    out << "  \"rightElbowVertex\": " << c.rightElbow << ",\n";
    out << "  \"rightWristVertex\": " << c.rightWrist << ",\n";
    out << "  \"anchors\": {\n";
    out << "    \"leftShoulder\": [" << P(c.leftShoulder).x << ", " << P(c.leftShoulder).y << ", " << P(c.leftShoulder).z << "],\n";
    out << "    \"leftElbow\": [" << P(c.leftElbow).x << ", " << P(c.leftElbow).y << ", " << P(c.leftElbow).z << "],\n";
    out << "    \"leftWrist\": [" << P(c.leftWrist).x << ", " << P(c.leftWrist).y << ", " << P(c.leftWrist).z << "],\n";
    out << "    \"rightShoulder\": [" << P(c.rightShoulder).x << ", " << P(c.rightShoulder).y << ", " << P(c.rightShoulder).z << "],\n";
    out << "    \"rightElbow\": [" << P(c.rightElbow).x << ", " << P(c.rightElbow).y << ", " << P(c.rightElbow).z << "],\n";
    out << "    \"rightWrist\": [" << P(c.rightWrist).x << ", " << P(c.rightWrist).y << ", " << P(c.rightWrist).z << "]\n";
    out << "  },\n";
    out << "  \"vertexRegion\": [";
    for (size_t i = 0; i < c.vertexRegion.size(); ++i) {
        if (i) out << ", ";
        out << c.vertexRegion[i];
    }
    out << "]\n";
    out << "}\n";
    return true;
}

static int pickVertex(const Mesh& mesh,
                      double mouseX, double mouseY,
                      int w, int h,
                      const glm::mat4& model,
                      const glm::mat4& view,
                      const glm::mat4& proj)
{
    int bestIdx = -1;
    float bestScore = std::numeric_limits<float>::max();

    for (int i = 0; i < (int)mesh.bindVertices.size(); ++i) {
        glm::vec4 clip = proj * view * model * glm::vec4(mesh.bindVertices[i].position, 1.0f);
        if (clip.w <= 1e-6f) continue;
        glm::vec3 ndc = glm::vec3(clip) / clip.w;
        if (ndc.z < -1.0f || ndc.z > 1.0f) continue;

        float sx = (ndc.x * 0.5f + 0.5f) * w;
        float sy = (1.0f - (ndc.y * 0.5f + 0.5f)) * h;

        float dx = sx - (float)mouseX;
        float dy = sy - (float)mouseY;
        float d2 = dx * dx + dy * dy;

        float score = d2 + 250.0f * (ndc.z + 1.0f);
        if (score < bestScore) {
            bestScore = score;
            bestIdx = i;
        }
    }

    if (bestScore > 60.0f * 60.0f) return -1;
    return bestIdx;
}

static void drawMarker(const glm::vec3& p, float r) {
    glPushMatrix();
    glTranslatef(p.x, p.y, p.z);
    glBegin(GL_LINES);
    glVertex3f(-r, 0, 0); glVertex3f(r, 0, 0);
    glVertex3f(0, -r, 0); glVertex3f(0, r, 0);
    glVertex3f(0, 0, -r); glVertex3f(0, 0, r);
    glEnd();
    glPopMatrix();
}

int main() {
    if (!glfwInit()) return -1;

    GLFWwindow* window = glfwCreateWindow(1280, 900, "Garment Calibration", nullptr, nullptr);
    if (!window) {
        glfwTerminate();
        return -1;
    }

    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    Mesh mesh = loadMesh("../meshes/jeans_denim_jacket.glb");
    if (!mesh.isValid) {
        std::cerr << "Failed to load mesh.\n";
        return -1;
    }

    glEnable(GL_DEPTH_TEST);
    glDisable(GL_CULL_FACE);
    glPointSize(10.0f);

    // Setup lighting
    glEnable(GL_LIGHTING);
    glEnable(GL_LIGHT0);
    glEnable(GL_COLOR_MATERIAL);
    glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);
    
    float lightPos[] = {1.0f, 1.5f, 2.0f, 0.0f};
    float lightAmbient[] = {0.4f, 0.4f, 0.4f, 1.0f};
    float lightDiffuse[] = {0.8f, 0.8f, 0.8f, 1.0f};
    float lightSpecular[] = {0.5f, 0.5f, 0.5f, 1.0f};
    
    glLightfv(GL_LIGHT0, GL_POSITION, lightPos);
    glLightfv(GL_LIGHT0, GL_AMBIENT, lightAmbient);
    glLightfv(GL_LIGHT0, GL_DIFFUSE, lightDiffuse);
    glLightfv(GL_LIGHT0, GL_SPECULAR, lightSpecular);

    OrbitCamera cam;
    bool dragging = false;
    double lastX = 0.0, lastY = 0.0;
    int stage = 0;
    int picked[6] = {-1, -1, -1, -1, -1, -1};
    bool clickLatch = false;

    std::cout << "Calibration mode\n";
    std::cout << "Mouse drag: orbit | Scroll: zoom | Enter: save once all 6 points are picked\n";
    std::cout << kStageNames[stage] << "\n";

    while (!glfwWindowShouldClose(window)) {
        int w, h;
        glfwGetFramebufferSize(window, &w, &h);
        int ww, wh;
        glfwGetWindowSize(window, &ww, &wh);
        float scaleX = (float)w / ww;
        float scaleY = (float)h / wh;
        float aspect = w / (float)std::max(h, 1);

        if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
            glfwSetWindowShouldClose(window, GLFW_TRUE);
        }

        double mx, my;
        glfwGetCursorPos(window, &mx, &my);
        mx *= scaleX;
        my *= scaleY;

        if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS) {
            if (!dragging) {
                dragging = true;
                lastX = mx; lastY = my;
            } else {
                cam.yaw += (float)(mx - lastX) * 0.01f;
                cam.pitch += (float)(my - lastY) * 0.01f;
                cam.pitch = glm::clamp(cam.pitch, -1.3f, 1.3f);
                lastX = mx; lastY = my;
            }
        } else {
            dragging = false;
        }

        if (glfwGetKey(window, GLFW_KEY_EQUAL) == GLFW_PRESS) cam.distance = std::max(0.4f, cam.distance - 0.02f);
        if (glfwGetKey(window, GLFW_KEY_MINUS) == GLFW_PRESS) cam.distance += 0.02f;

        glm::mat4 model(1.0f);
        glm::mat4 view = cam.view();
        glm::mat4 proj = glm::perspective(glm::radians(45.0f), aspect, 0.01f, 100.0f);

        bool leftClick = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS;
        if (leftClick && !clickLatch && stage < 6) {
            int idx = pickVertex(mesh, mx, my, w, h, model, view, proj);
            if (idx >= 0) {
                picked[stage] = idx;
                std::cout << "Picked stage " << stage << ": vertex " << idx << "\n";
                ++stage;
                if (stage < 6) std::cout << kStageNames[stage] << "\n";
                else std::cout << "All 6 points selected. Press Enter to save calibration.json\n";
            }
        }
        clickLatch = leftClick;

        if (stage == 6 && glfwGetKey(window, GLFW_KEY_ENTER) == GLFW_PRESS) {
            CalibrationData c = computeCalibration(mesh, picked[0], picked[1], picked[2], picked[3], picked[4], picked[5]);
            if (saveCalibrationJson("calibration.json", mesh, c)) {
                std::cout << "Saved calibration.json\n";
            } else {
                std::cout << "Failed to save calibration.json\n";
            }
            break;
        }

        glViewport(0, 0, w, h);
        glClearColor(0.08f, 0.08f, 0.10f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glMatrixMode(GL_PROJECTION);
        glLoadMatrixf(glm::value_ptr(proj));
        glMatrixMode(GL_MODELVIEW);
        glm::mat4 mv = view * model;
        glLoadMatrixf(glm::value_ptr(mv));

        glColor3f(0.75f, 0.82f, 0.95f);
        mesh.draw();

        glLineWidth(3.0f);
        for (int i = 0; i < stage; ++i) {
            if (picked[i] < 0) continue;
            const glm::vec3 p = mesh.bindVertices[picked[i]].position;
            if (i < 3) glColor3f(1.0f, 0.25f, 0.25f);
            else glColor3f(0.2f, 1.0f, 1.0f);
            drawMarker(p, 0.03f);
        }

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
