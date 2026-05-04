#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <limits>
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <queue>
#include <functional>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <GLFW/glfw3.h>

#include "mesh.h"

struct Anchor {
    int vertex = -1;
    glm::vec3 position = glm::vec3(0.0f);
    bool isVirtual = false;
};

struct CalibrationData {
    Anchor leftShoulder;
    Anchor leftElbow;
    Anchor leftWrist;
    Anchor rightShoulder;
    Anchor rightElbow;
    Anchor rightWrist;
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
    "Click LEFT SHOULDER seam on mesh",
    "Click LEFT ELBOW area OR blank space where elbow would be",
    "Click LEFT WRIST/cuff area OR blank space where wrist would be",
    "Click RIGHT SHOULDER seam on mesh",
    "Click RIGHT ELBOW area OR blank space where elbow would be",
    "Click RIGHT WRIST/cuff area OR blank space where wrist would be"
};

static std::string getMeshBaseName(const std::string& path) {
    std::filesystem::path p(path);
    return p.stem().string();
}

static std::string getCalibrationPath(const std::string& meshPath) {
    std::string name = getMeshBaseName(meshPath);
    return "../calibrations/calibration_" + name + ".json";
}

static bool stageRequiresMeshVertex(int stage) {
    return stage == 0 || stage == 3; // shoulders must be real mesh anchors
}

static float distancePointToSegment(const glm::vec3& p, const glm::vec3& a, const glm::vec3& b) {
    glm::vec3 ab = b - a;
    float denom = glm::dot(ab, ab);
    float t = 0.0f;

    if (denom > 1e-8f) {
        t = glm::clamp(glm::dot(p - a, ab) / denom, 0.0f, 1.0f);
    }

    glm::vec3 q = a + t * ab;
    return glm::length(p - q);
}

static glm::vec3 safeNormalize(const glm::vec3& v, const glm::vec3& fallback) {
    float len = glm::length(v);
    if (len < 1e-6f) return fallback;
    return v / len;
}

static Anchor makeVertexAnchor(const Mesh& mesh, int idx) {
    Anchor a;
    a.vertex = idx;
    a.position = mesh.bindVertices[idx].position;
    a.isVirtual = false;
    return a;
}

static glm::vec3 screenToWorldOnPlane(
    double mouseX,
    double mouseY,
    int w,
    int h,
    const glm::mat4& view,
    const glm::mat4& proj,
    const glm::vec3& planePoint
) {
    float x = (2.0f * static_cast<float>(mouseX)) / static_cast<float>(w) - 1.0f;
    float y = 1.0f - (2.0f * static_cast<float>(mouseY)) / static_cast<float>(h);

    glm::mat4 invVP = glm::inverse(proj * view);

    glm::vec4 nearClip(x, y, -1.0f, 1.0f);
    glm::vec4 farClip(x, y, 1.0f, 1.0f);

    glm::vec4 nearWorld4 = invVP * nearClip;
    glm::vec4 farWorld4 = invVP * farClip;

    nearWorld4 /= nearWorld4.w;
    farWorld4 /= farWorld4.w;

    glm::vec3 rayOrigin = glm::vec3(nearWorld4);
    glm::vec3 rayDir = glm::normalize(glm::vec3(farWorld4) - rayOrigin);

    glm::mat4 invView = glm::inverse(view);
    glm::vec3 cameraPos = glm::vec3(invView[3]);
    glm::vec3 planeNormal = glm::normalize(cameraPos - planePoint);

    float denom = glm::dot(rayDir, planeNormal);

    if (std::abs(denom) < 1e-6f) {
        return planePoint;
    }

    float t = glm::dot(planePoint - rayOrigin, planeNormal) / denom;
    return rayOrigin + rayDir * t;
}

static CalibrationData computeCalibration(
    const Mesh& mesh,
    const Anchor& ls,
    const Anchor& le,
    const Anchor& lw,
    const Anchor& rs,
    const Anchor& re,
    const Anchor& rw
) {
    CalibrationData out;

    out.leftShoulder = ls;
    out.leftElbow = le;
    out.leftWrist = lw;
    out.rightShoulder = rs;
    out.rightElbow = re;
    out.rightWrist = rw;

    out.vertexRegion.assign(
        mesh.bindVertices.size(),
        CalibrationRegion::REGION_TORSO
    );

    const glm::vec3 pLS = ls.position;
    const glm::vec3 pLE = le.position;
    const glm::vec3 pLW = lw.position;
    const glm::vec3 pRS = rs.position;
    const glm::vec3 pRE = re.position;
    const glm::vec3 pRW = rw.position;

    const float shoulderWidth = glm::max(glm::length(pRS - pLS), 0.1f);

    // Larger radius works better for short sleeves because elbow/wrist may be virtual.
    const float leftRadius = std::max(0.30f * shoulderWidth, 0.055f);
    const float rightRadius = leftRadius;

    glm::vec3 leftAxis = safeNormalize(pLW - pLS, glm::vec3(-1, 0, 0));
    glm::vec3 rightAxis = safeNormalize(pRW - pRS, glm::vec3(1, 0, 0));

    float leftElbowAxial = glm::dot(pLE - pLS, leftAxis);
    float rightElbowAxial = glm::dot(pRE - pRS, rightAxis);

    for (size_t i = 0; i < mesh.bindVertices.size(); ++i) {
        const glm::vec3 p = mesh.bindVertices[i].position;

        bool leftCorridor =
            distancePointToSegment(p, pLS, pLE) <= leftRadius ||
            distancePointToSegment(p, pLE, pLW) <= leftRadius * 1.8f;

        bool rightCorridor =
            distancePointToSegment(p, pRS, pRE) <= rightRadius ||
            distancePointToSegment(p, pRE, pRW) <= rightRadius * 1.8f;

        float leftDist = std::min(
            distancePointToSegment(p, pLS, pLE),
            distancePointToSegment(p, pLE, pLW)
        );

        float rightDist = std::min(
            distancePointToSegment(p, pRS, pRE),
            distancePointToSegment(p, pRE, pRW)
        );

        if (leftCorridor && (!rightCorridor || leftDist <= rightDist)) {
            float axial = glm::dot(p - pLS, leftAxis);

            out.vertexRegion[i] =
                axial <= leftElbowAxial
                    ? CalibrationRegion::REGION_LEFT_UPPER_SLEEVE
                    : CalibrationRegion::REGION_LEFT_LOWER_SLEEVE;
        } else if (rightCorridor) {
            float axial = glm::dot(p - pRS, rightAxis);

            out.vertexRegion[i] =
                axial <= rightElbowAxial
                    ? CalibrationRegion::REGION_RIGHT_UPPER_SLEEVE
                    : CalibrationRegion::REGION_RIGHT_LOWER_SLEEVE;
        }
    }

    return out;
}

static bool saveCalibrationJson(const std::string& path, const CalibrationData& c) {
    std::filesystem::create_directories(std::filesystem::path(path).parent_path());

    std::ofstream out(path);
    if (!out) return false;

    auto P = [](const Anchor& a) {
        return a.position;
    };

    out << "{\n";

    out << "  \"leftShoulderVertex\": " << c.leftShoulder.vertex << ",\n";
    out << "  \"leftElbowVertex\": " << c.leftElbow.vertex << ",\n";
    out << "  \"leftWristVertex\": " << c.leftWrist.vertex << ",\n";
    out << "  \"rightShoulderVertex\": " << c.rightShoulder.vertex << ",\n";
    out << "  \"rightElbowVertex\": " << c.rightElbow.vertex << ",\n";
    out << "  \"rightWristVertex\": " << c.rightWrist.vertex << ",\n";

    out << "  \"virtualAnchors\": {\n";
    out << "    \"leftShoulder\": " << (c.leftShoulder.isVirtual ? "true" : "false") << ",\n";
    out << "    \"leftElbow\": " << (c.leftElbow.isVirtual ? "true" : "false") << ",\n";
    out << "    \"leftWrist\": " << (c.leftWrist.isVirtual ? "true" : "false") << ",\n";
    out << "    \"rightShoulder\": " << (c.rightShoulder.isVirtual ? "true" : "false") << ",\n";
    out << "    \"rightElbow\": " << (c.rightElbow.isVirtual ? "true" : "false") << ",\n";
    out << "    \"rightWrist\": " << (c.rightWrist.isVirtual ? "true" : "false") << "\n";
    out << "  },\n";

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

static int pickVertex(
    const Mesh& mesh,
    double mouseX,
    double mouseY,
    int w,
    int h,
    const glm::mat4& model,
    const glm::mat4& view,
    const glm::mat4& proj
) {
    int bestIdx = -1;
    float bestScore = std::numeric_limits<float>::max();

    for (int i = 0; i < static_cast<int>(mesh.bindVertices.size()); ++i) {
        glm::vec4 clip = proj * view * model * glm::vec4(mesh.bindVertices[i].position, 1.0f);

        if (clip.w <= 1e-6f) continue;

        glm::vec3 ndc = glm::vec3(clip) / clip.w;

        if (ndc.z < -1.0f || ndc.z > 1.0f) continue;

        float sx = (ndc.x * 0.5f + 0.5f) * w;
        float sy = (1.0f - (ndc.y * 0.5f + 0.5f)) * h;

        float dx = sx - static_cast<float>(mouseX);
        float dy = sy - static_cast<float>(mouseY);
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

int main(int argc, char** argv) {
    if (!glfwInit()) return -1;

    GLFWwindow* window = glfwCreateWindow(1280, 900, "Garment Calibration", nullptr, nullptr);
    if (!window) {
        glfwTerminate();
        return -1;
    }

    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    if (argc < 2) {
        std::cerr << "Usage: ./calibrate <mesh_path>\n";
        std::cerr << "Example: ./calibrate ../meshes/t-shirt.glb\n";
        glfwDestroyWindow(window);
        glfwTerminate();
        return -1;
    }

    std::string meshPath = argv[1];
    std::string calibrationPath = getCalibrationPath(meshPath);

    std::cout << "Loading mesh: " << meshPath << "\n";
    std::cout << "Calibration file will be: " << calibrationPath << "\n";

    Mesh mesh = loadMesh(meshPath);
    if (!mesh.isValid) {
        std::cerr << "Failed to load mesh.\n";
        glfwDestroyWindow(window);
        glfwTerminate();
        return -1;
    }

    glEnable(GL_DEPTH_TEST);
    glDisable(GL_CULL_FACE);
    glPointSize(10.0f);

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
    double lastX = 0.0;
    double lastY = 0.0;

    int stage = 0;
    Anchor picked[6];

    bool clickLatch = false;
    bool saveLatch = false;

    std::cout << "Calibration mode\n";
    std::cout << "Right drag: orbit | + / -: zoom\n";
    std::cout << "Shoulders must be clicked on the mesh.\n";
    std::cout << "Elbows/wrists may be clicked on mesh OR blank space for short sleeves.\n";
    std::cout << "Enter: save once all 6 anchors are selected\n";
    std::cout << kStageNames[stage] << "\n";

    while (!glfwWindowShouldClose(window)) {
        int w, h;
        glfwGetFramebufferSize(window, &w, &h);

        int ww, wh;
        glfwGetWindowSize(window, &ww, &wh);

        if (ww <= 0 || wh <= 0) {
            glfwPollEvents();
            continue;
        }

        float scaleX = static_cast<float>(w) / static_cast<float>(ww);
        float scaleY = static_cast<float>(h) / static_cast<float>(wh);
        float aspect = w / static_cast<float>(std::max(h, 1));

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
                lastX = mx;
                lastY = my;
            } else {
                cam.yaw += static_cast<float>(mx - lastX) * 0.01f;
                cam.pitch += static_cast<float>(my - lastY) * 0.01f;
                cam.pitch = glm::clamp(cam.pitch, -1.3f, 1.3f);
                lastX = mx;
                lastY = my;
            }
        } else {
            dragging = false;
        }

        if (glfwGetKey(window, GLFW_KEY_EQUAL) == GLFW_PRESS ||
            glfwGetKey(window, GLFW_KEY_KP_ADD) == GLFW_PRESS) {
            cam.distance = std::max(0.4f, cam.distance - 0.02f);
        }

        if (glfwGetKey(window, GLFW_KEY_MINUS) == GLFW_PRESS ||
            glfwGetKey(window, GLFW_KEY_KP_SUBTRACT) == GLFW_PRESS) {
            cam.distance += 0.02f;
        }

        glm::mat4 model(1.0f);
        glm::mat4 view = cam.view();
        glm::mat4 proj = glm::perspective(glm::radians(45.0f), aspect, 0.01f, 100.0f);

        bool leftClick = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS;

        if (leftClick && !clickLatch && stage < 6) {
            int idx = pickVertex(mesh, mx, my, w, h, model, view, proj);

            if (idx >= 0) {
                picked[stage] = makeVertexAnchor(mesh, idx);

                std::cout << "Picked stage " << stage
                          << ": vertex " << idx << "\n";

                ++stage;

                if (stage < 6) {
                    std::cout << kStageNames[stage] << "\n";
                } else {
                    std::cout << "All 6 anchors selected. Press Enter to save calibration.\n";
                }
            } else if (!stageRequiresMeshVertex(stage)) {
                glm::vec3 planePoint;

                if (stage == 1 || stage == 2) {
                    planePoint = picked[0].position;
                } else {
                    planePoint = picked[3].position;
                }

                glm::vec3 virtualPos = screenToWorldOnPlane(mx, my, w, h, view, proj, planePoint);

                picked[stage].vertex = -1;
                picked[stage].position = virtualPos;
                picked[stage].isVirtual = true;

                std::cout << "Picked stage " << stage
                          << ": virtual point at "
                          << virtualPos.x << ", "
                          << virtualPos.y << ", "
                          << virtualPos.z << "\n";

                ++stage;

                if (stage < 6) {
                    std::cout << kStageNames[stage] << "\n";
                } else {
                    std::cout << "All 6 anchors selected. Press Enter to save calibration.\n";
                }
            } else {
                std::cout << "This stage requires a mesh vertex. Click directly on the mesh.\n";
            }
        }

        clickLatch = leftClick;

        bool savePressed = glfwGetKey(window, GLFW_KEY_ENTER) == GLFW_PRESS;

        if (stage == 6 && savePressed && !saveLatch) {
            CalibrationData c = computeCalibration(
                mesh,
                picked[0],
                picked[1],
                picked[2],
                picked[3],
                picked[4],
                picked[5]
            );

            if (saveCalibrationJson(calibrationPath, c)) {
                std::cout << "Saved " << calibrationPath << "\n";
            } else {
                std::cout << "Failed to save calibration.\n";
            }

            break;
        }

        saveLatch = savePressed;

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

        glDisable(GL_LIGHTING);
        glLineWidth(3.0f);

        for (int i = 0; i < stage; ++i) {
            if (picked[i].isVirtual) {
                glColor3f(1.0f, 0.9f, 0.1f);
                drawMarker(picked[i].position, 0.04f);
            } else {
                if (i < 3) glColor3f(1.0f, 0.25f, 0.25f);
                else glColor3f(0.2f, 1.0f, 1.0f);

                drawMarker(picked[i].position, 0.03f);
            }
        }

        glEnable(GL_LIGHTING);

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}