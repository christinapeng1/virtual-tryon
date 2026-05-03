#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <GLFW/glfw3.h>
#include <opencv2/opencv.hpp>

#include "mesh.h"
#include "pose_tracker.h"
#include "utils.h"

static std::string meshBaseNameFromPath(const std::string& meshPath) {
    std::filesystem::path p(meshPath);
    return p.stem().string();
}

static std::string calibrationPathForMesh(const std::string& meshPath) {
    std::string name = meshBaseNameFromPath(meshPath);
    return "../calibrations/calibration_" + name + ".json";
}

static glm::vec3 smoothVec3(const glm::vec3& prev, const glm::vec3& curr, float alpha) {
    return prev * (1.0f - alpha) + curr * alpha;
}

static glm::vec3 safeNormalize(const glm::vec3& v, const glm::vec3& fallback = glm::vec3(1.0f, 0.0f, 0.0f)) {
    float len = glm::length(v);
    if (len < 1e-6f) return fallback;
    return v / len;
}

static float smoothstepf(float a, float b, float x) {
    if (std::abs(b - a) < 1e-6f) return x < a ? 0.0f : 1.0f;
    float t = glm::clamp((x - a) / (b - a), 0.0f, 1.0f);
    return t * t * (3.0f - 2.0f * t);
}

static float getMeshVerticalHeight(const Mesh& mesh) {
    if (mesh.bindVertices.empty()) return 1.0f;

    float minY = mesh.bindVertices[0].position.y;
    float maxY = mesh.bindVertices[0].position.y;

    for (const auto& v : mesh.bindVertices) {
        minY = glm::min(minY, v.position.y);
        maxY = glm::max(maxY, v.position.y);
    }

    return glm::max(maxY - minY, 1e-5f);
}

static glm::mat4 rotationBetweenVectors(const glm::vec3& from, const glm::vec3& to) {
    glm::vec3 a = safeNormalize(from);
    glm::vec3 b = safeNormalize(to);

    float d = glm::clamp(glm::dot(a, b), -1.0f, 1.0f);
    if (d > 0.9999f) return glm::mat4(1.0f);

    if (d < -0.9999f) {
        glm::vec3 arbitrary = std::abs(a.x) < 0.9f ? glm::vec3(1, 0, 0) : glm::vec3(0, 1, 0);
        glm::vec3 axis = glm::normalize(glm::cross(a, arbitrary));
        return glm::rotate(glm::mat4(1.0f), glm::pi<float>(), axis);
    }

    glm::vec3 axis = glm::normalize(glm::cross(a, b));
    float angle = std::acos(d);
    return glm::rotate(glm::mat4(1.0f), angle, axis);
}

static glm::mat4 makeRigidSegmentTransform(
    const glm::vec3& localA,
    const glm::vec3& localB,
    const glm::vec3& worldA,
    const glm::vec3& worldB,
    float extraScale = 1.0f)
{
    glm::vec3 localDir = localB - localA;
    glm::vec3 worldDir = worldB - worldA;

    float localLen = glm::length(localDir);
    float worldLen = glm::length(worldDir);

    if (localLen < 1e-6f || worldLen < 1e-6f) {
        return glm::translate(glm::mat4(1.0f), worldA - localA);
    }

    glm::mat4 R = rotationBetweenVectors(localDir, worldDir);
    float scale = (worldLen / localLen) * extraScale;
    glm::mat4 S = glm::scale(glm::mat4(1.0f), glm::vec3(scale));

    return glm::translate(glm::mat4(1.0f), worldA) *
           R *
           S *
           glm::translate(glm::mat4(1.0f), -localA);
}

struct SleeveRegion {
    glm::vec3 shoulder;
    glm::vec3 elbow;
    glm::vec3 wrist;
    bool left = true;
};

static void deformSleeveRegion(
    Mesh& mesh,
    const glm::mat4& torsoModel,
    const SleeveRegion& local,
    const SleeveRegion& world,
    float sideSign,
    float torsoBlendWidth,
    float upperBlendWidth,
    float lowerBlendWidth)
{
    glm::mat4 upperT = makeRigidSegmentTransform(
        local.shoulder, local.elbow,
        world.shoulder, world.elbow
    );

    glm::mat4 lowerT = makeRigidSegmentTransform(
        local.elbow, local.wrist,
        world.elbow, world.wrist
    );

    glm::vec3 armAxis = safeNormalize(
        local.wrist - local.shoulder,
        glm::vec3(sideSign, 0.0f, 0.0f)
    );

    float upperLen = glm::max(glm::length(local.elbow - local.shoulder), 1e-5f);

    for (size_t i = 0; i < mesh.vertices.size() && i < mesh.bindVertices.size(); ++i) {
        const glm::vec3 p = mesh.bindVertices[i].position;

        bool isUpper = false;
        bool isLower = false;

        if (sideSign < 0.0f) {
            isUpper = mesh.vertexRegion[i] == REGION_LEFT_UPPER_SLEEVE;
            isLower = mesh.vertexRegion[i] == REGION_LEFT_LOWER_SLEEVE;
        } else {
            isUpper = mesh.vertexRegion[i] == REGION_RIGHT_UPPER_SLEEVE;
            isLower = mesh.vertexRegion[i] == REGION_RIGHT_LOWER_SLEEVE;
        }

        if (!isUpper && !isLower) continue;

        glm::vec3 rel = p - local.shoulder;
        float axial = glm::dot(rel, armAxis);

        float shoulderToUpper = smoothstepf(
            0.0f,
            upperLen * 0.65f,
            axial
        );

        // Make shoulder transition softer and less abrupt.
        shoulderToUpper = shoulderToUpper * shoulderToUpper * (3.0f - 2.0f * shoulderToUpper);

        float upperToLower = smoothstepf(
            upperLen - upperBlendWidth,
            upperLen + lowerBlendWidth,
            axial
        );

        glm::vec4 p4(p, 1.0f);
        glm::vec3 torsoP = glm::vec3(torsoModel * p4);
        glm::vec3 upperP = glm::vec3(upperT * p4);
        glm::vec3 lowerP = glm::vec3(lowerT * p4);

        glm::vec3 atUpper = glm::mix(torsoP, upperP, shoulderToUpper);

        if (isUpper) {
            mesh.vertices[i].position = atUpper;
        } else {
            mesh.vertices[i].position = glm::mix(atUpper, lowerP, upperToLower);
        }
    }
}

static void loadCalibrationIfAvailable(Mesh& mesh, const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        std::cout << "No calibration file found at " << path << ", using defaults\n";
        return;
    }

    std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    file.close();

    size_t anchorsStart = content.find("\"anchors\"");
    if (anchorsStart == std::string::npos) {
        std::cout << "No anchors in calibration file\n";
        return;
    }

    auto parseVec3 = [](const std::string& text, size_t start) -> glm::vec3 {
        size_t bracketStart = text.find("[", start);
        size_t bracketEnd = text.find("]", bracketStart);
        std::string vecStr = text.substr(bracketStart + 1, bracketEnd - bracketStart - 1);

        glm::vec3 v(0.0f);
        int count = std::sscanf(vecStr.c_str(), "%f, %f, %f", &v.x, &v.y, &v.z);
        if (count != 3) std::cout << "Failed to parse vec3\n";
        return v;
    };

    std::cout << "Loading calibration...\n";

    mesh.leftShoulderPos = parseVec3(content, content.find("\"leftShoulder\""));
    mesh.leftElbowPos = parseVec3(content, content.find("\"leftElbow\""));
    mesh.leftWristPos = parseVec3(content, content.find("\"leftWrist\""));
    mesh.rightShoulderPos = parseVec3(content, content.find("\"rightShoulder\""));
    mesh.rightElbowPos = parseVec3(content, content.find("\"rightElbow\""));
    mesh.rightWristPos = parseVec3(content, content.find("\"rightWrist\""));

    size_t regionStart = content.find("\"vertexRegion\"");
    if (regionStart != std::string::npos) {
        size_t arrayStart = content.find("[", regionStart);
        size_t arrayEnd = content.find("]", arrayStart);
        std::string arrayStr = content.substr(arrayStart + 1, arrayEnd - arrayStart - 1);
        std::istringstream iss(arrayStr);

        mesh.vertexRegion.clear();

        int value;
        while (iss >> value) {
            mesh.vertexRegion.push_back(value);
            if (iss.peek() == ',') iss.ignore();
        }

        std::cout << "Loaded " << mesh.vertexRegion.size() << " vertex regions\n";
    }

    std::cout << "Calibration loaded successfully\n";
}

int main(int argc, char** argv) {
    std::cout << "Initializing...\n";

    if (argc < 2) {
        std::cerr << "Usage: ./main <mesh_path>\n";
        std::cerr << "Example: ./main ../meshes/t-shirt.glb\n";
        return -1;
    }

    std::string meshPath = argv[1];
    std::string calibrationPath = calibrationPathForMesh(meshPath);

    std::cout << "Mesh: " << meshPath << "\n";
    std::cout << "Calibration: " << calibrationPath << "\n";

    if (!glfwInit()) return -1;

#ifdef __APPLE__
    glfwWindowHint(GLFW_COCOA_RETINA_FRAMEBUFFER, GLFW_FALSE);
#endif

    GLFWwindow* window = glfwCreateWindow(1280, 720, "AR Single Mesh Sleeve Deform", nullptr, nullptr);
    if (!window) {
        std::cerr << "Window creation failed\n";
        glfwTerminate();
        return -1;
    }

    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    cv::VideoCapture camera(0);
    if (!camera.isOpened()) {
        std::cerr << "Camera failed\n";
        glfwDestroyWindow(window);
        glfwTerminate();
        return -1;
    }

    camera.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
    camera.set(cv::CAP_PROP_FRAME_HEIGHT, 720);

    cv::Mat frame;
    camera >> frame;
    if (frame.empty()) {
        std::cerr << "No camera frame\n";
        glfwDestroyWindow(window);
        glfwTerminate();
        return -1;
    }

    int initW, initH;
    glfwGetWindowSize(window, &initW, &initH);

    cv::flip(frame, frame, 1);
    cv::cvtColor(frame, frame, cv::COLOR_BGR2RGB);
    cv::resize(frame, frame, cv::Size(initW, initH));

    GLuint camTex = 0;
    glGenTextures(1, &camTex);
    glBindTexture(GL_TEXTURE_2D, camTex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, initW, initH, 0, GL_RGB, GL_UNSIGNED_BYTE, frame.data);

    Mesh shirtMesh = loadMesh(meshPath);
    loadCalibrationIfAvailable(shirtMesh, calibrationPath);

    float meshVerticalHeight = getMeshVerticalHeight(shirtMesh);

    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_LIGHTING);
    glEnable(GL_LIGHT0);
    glEnable(GL_COLOR_MATERIAL);
    glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);
    glDisable(GL_CULL_FACE);
    glShadeModel(GL_SMOOTH);

    GLfloat lightPos[] = {0.0f, 0.0f, 1.0f, 0.0f}; // directional light from camera
    GLfloat lightAmbient[] = {0.45f, 0.45f, 0.45f, 1.0f};
    GLfloat lightDiffuse[] = {0.95f, 0.95f, 0.95f, 1.0f};
    GLfloat lightSpecular[] = {0.20f, 0.20f, 0.20f, 1.0f};

    glLightfv(GL_LIGHT0, GL_POSITION, lightPos);
    glLightfv(GL_LIGHT0, GL_AMBIENT, lightAmbient);
    glLightfv(GL_LIGHT0, GL_DIFFUSE, lightDiffuse);
    glLightfv(GL_LIGHT0, GL_SPECULAR, lightSpecular);

    PoseTracker tracker;
    bool trackerReady = tracker.init();
    if (trackerReady) std::cout << "Pose tracker ready\n";

    glm::vec2 leftShoulder(0.35f, 0.40f);
    glm::vec2 rightShoulder(0.65f, 0.40f);
    glm::vec2 leftElbow(0.28f, 0.54f);
    glm::vec2 rightElbow(0.72f, 0.54f);
    glm::vec2 leftWrist(0.24f, 0.70f);
    glm::vec2 rightWrist(0.76f, 0.70f);
    glm::vec2 leftHip(0.35f, 0.70f);
    glm::vec2 rightHip(0.65f, 0.70f);
    glm::vec2 nose(0.50f, 0.30f);

    float rawYawDegrees = 0.0f;
    float smoothedYaw = 0.0f;
    float poseVisibility = 0.0f;
    float worldShoulderWidth = 0.4f;

    float depth = -2.5f;

    glm::vec3 sLS(0.0f), sRS(0.0f), sLE(0.0f), sRE(0.0f);
    glm::vec3 sLW(0.0f), sRW(0.0f), sLH(0.0f), sRH(0.0f), sNose(0.0f);

    bool smoothingInitialized = false;
    const float jointSmoothAlpha = 0.35f;

    while (!glfwWindowShouldClose(window)) {
        int w, h;
        glfwGetWindowSize(window, &w, &h);
        if (w <= 0 || h <= 0) {
            glfwPollEvents();
            continue;
        }

        float aspect = w / static_cast<float>(h);

        if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
            glfwSetWindowShouldClose(window, GLFW_TRUE);
        }

        if (trackerReady) {
            float yaw, vis, wsw;
            if (tracker.getRotation(yaw, vis, wsw)) {
                rawYawDegrees = yaw;
                poseVisibility = vis;
                worldShoulderWidth = wsw;
            }

            glm::vec2 tls, trs, tle, tre, tlw, trw;
            if (tracker.getUpperBody(tls, trs, tle, tre, tlw, trw)) {
                leftShoulder = tls;
                rightShoulder = trs;
                leftElbow = tle;
                rightElbow = tre;
                leftWrist = tlw;
                rightWrist = trw;
            }

            glm::vec2 tlh, trh;
            if (tracker.getHips(tlh, trh)) {
                leftHip = tlh;
                rightHip = trh;
            }

            glm::vec2 tnose;
            if (tracker.getNose(tnose)) {
                nose = tnose;
            }
        }

        leftShoulder = glm::clamp(leftShoulder, glm::vec2(0.0f), glm::vec2(1.0f));
        rightShoulder = glm::clamp(rightShoulder, glm::vec2(0.0f), glm::vec2(1.0f));
        leftElbow = glm::clamp(leftElbow, glm::vec2(0.0f), glm::vec2(1.0f));
        rightElbow = glm::clamp(rightElbow, glm::vec2(0.0f), glm::vec2(1.0f));
        leftWrist = glm::clamp(leftWrist, glm::vec2(0.0f), glm::vec2(1.0f));
        rightWrist = glm::clamp(rightWrist, glm::vec2(0.0f), glm::vec2(1.0f));
        leftHip = glm::clamp(leftHip, glm::vec2(0.0f), glm::vec2(1.0f));
        rightHip = glm::clamp(rightHip, glm::vec2(0.0f), glm::vec2(1.0f));
        nose = glm::clamp(nose, glm::vec2(0.0f), glm::vec2(1.0f));
        depth = glm::clamp(depth, -8.0f, -1.5f);

        camera >> frame;
        if (frame.empty()) break;

        cv::flip(frame, frame, 1);
        cv::cvtColor(frame, frame, cv::COLOR_BGR2RGB);
        cv::resize(frame, frame, cv::Size(w, h));

        glBindTexture(GL_TEXTURE_2D, camTex);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, w, h, GL_RGB, GL_UNSIGNED_BYTE, frame.data);

        glm::vec3 wsLS = screenToWorld(leftShoulder, depth, aspect);
        glm::vec3 wsRS = screenToWorld(rightShoulder, depth, aspect);
        glm::vec3 wsLE = screenToWorld(leftElbow, depth, aspect);
        glm::vec3 wsRE = screenToWorld(rightElbow, depth, aspect);
        glm::vec3 wsLW = screenToWorld(leftWrist, depth, aspect);
        glm::vec3 wsRW = screenToWorld(rightWrist, depth, aspect);
        glm::vec3 wsLH = screenToWorld(leftHip, depth, aspect);
        glm::vec3 wsRH = screenToWorld(rightHip, depth, aspect);
        glm::vec3 wsNose = screenToWorld(nose, depth, aspect);

        if (!smoothingInitialized) {
            sLS = wsLS; sRS = wsRS; sLE = wsLE; sRE = wsRE;
            sLW = wsLW; sRW = wsRW; sLH = wsLH; sRH = wsRH; sNose = wsNose;
            smoothingInitialized = true;
        } else {
            sLS = smoothVec3(sLS, wsLS, jointSmoothAlpha);
            sRS = smoothVec3(sRS, wsRS, jointSmoothAlpha);
            sLE = smoothVec3(sLE, wsLE, jointSmoothAlpha);
            sRE = smoothVec3(sRE, wsRE, jointSmoothAlpha);
            sLW = smoothVec3(sLW, wsLW, jointSmoothAlpha);
            sRW = smoothVec3(sRW, wsRW, jointSmoothAlpha);
            sLH = smoothVec3(sLH, wsLH, jointSmoothAlpha);
            sRH = smoothVec3(sRH, wsRH, jointSmoothAlpha);
            sNose = smoothVec3(sNose, wsNose, jointSmoothAlpha);
        }

        glm::vec3 torsoCenter = (sLS + sRS) * 0.5f;
        glm::vec3 hipCenter = (sLH + sRH) * 0.5f;
        glm::vec3 shoulderVec = sRS - sLS;

        float torsoHeightRef = glm::length(torsoCenter - hipCenter);
        torsoHeightRef = glm::max(torsoHeightRef, 1e-5f);

        glViewport(0, 0, w, h);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glDisable(GL_DEPTH_TEST);
        glDisable(GL_LIGHTING);

        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        glOrtho(0, w, h, 0, -1, 1);

        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();

        glEnable(GL_TEXTURE_2D);
        glBindTexture(GL_TEXTURE_2D, camTex);
        glColor3f(1.0f, 1.0f, 1.0f);

        glBegin(GL_QUADS);
            glTexCoord2f(0.0f, 0.0f); glVertex2f(0.0f, 0.0f);
            glTexCoord2f(1.0f, 0.0f); glVertex2f((GLfloat)w, 0.0f);
            glTexCoord2f(1.0f, 1.0f); glVertex2f((GLfloat)w, (GLfloat)h);
            glTexCoord2f(0.0f, 1.0f); glVertex2f(0.0f, (GLfloat)h);
        glEnd();

        glDisable(GL_TEXTURE_2D);

        glPointSize(10.0f);
        glBegin(GL_POINTS);
            glColor3f(0.0f, 1.0f, 0.0f);
            glVertex2f(leftShoulder.x * w, leftShoulder.y * h);
            glVertex2f(rightShoulder.x * w, rightShoulder.y * h);

            glColor3f(1.0f, 1.0f, 0.0f);
            glVertex2f(leftElbow.x * w, leftElbow.y * h);
            glVertex2f(rightElbow.x * w, rightElbow.y * h);

            glColor3f(1.0f, 0.2f, 0.2f);
            glVertex2f(leftWrist.x * w, leftWrist.y * h);
            glVertex2f(rightWrist.x * w, rightWrist.y * h);
        glEnd();

        glEnable(GL_DEPTH_TEST);
        glEnable(GL_LIGHTING);

        glMatrixMode(GL_PROJECTION);
        glm::mat4 proj = glm::perspective(glm::radians(45.0f), aspect, 0.1f, 100.0f);
        glLoadMatrixf(glm::value_ptr(proj));

        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();
        glLightfv(GL_LIGHT0, GL_POSITION, lightPos);

        if (shirtMesh.isValid) {
            glm::vec3 meshShoulderVec = shirtMesh.rightShoulderPos - shirtMesh.leftShoulderPos;
            glm::vec3 meshShoulderCenter = (shirtMesh.leftShoulderPos + shirtMesh.rightShoulderPos) * 0.5f;
            float meshShoulderWidth = glm::length(meshShoulderVec);

            if (meshShoulderWidth > 1e-5f && torsoHeightRef > 1e-5f) {
                float delta = rawYawDegrees - smoothedYaw;
                if (delta > 180.0f) delta -= 360.0f;
                if (delta < -180.0f) delta += 360.0f;
                smoothedYaw += 0.08f * delta;

                glm::mat4 torsoModel(1.0f);

                torsoModel = glm::translate(torsoModel, -meshShoulderCenter);

                torsoModel =
                    glm::rotate(
                        glm::mat4(1.0f),
                        glm::radians(-smoothedYaw),
                        glm::vec3(0.0f, 1.0f, 0.0f)
                    ) * torsoModel;

                // Stable torso scale:
                // user size = MediaPipe shoulder-center to hip-center.
                // mesh size = mesh vertical height from bind-pose vertices.
                float scaleFactor = torsoHeightRef / glm::max(meshVerticalHeight, 1e-5f);

                // Visual fit multiplier.
                // Increase if jacket is too small, decrease if too big.
                scaleFactor *= 1.35f;

                torsoModel = glm::scale(glm::mat4(1.0f), glm::vec3(scaleFactor)) * torsoModel;

                // Slightly raise the jacket relative to shoulder center.
                glm::vec3 finalPos = torsoCenter + glm::vec3(0.0f, 0.08f * scaleFactor, 0.0f);
                torsoModel = glm::translate(glm::mat4(1.0f), finalPos) * torsoModel;

                shirtMesh.resetToBindPose();

                for (size_t i = 0; i < shirtMesh.vertices.size() && i < shirtMesh.bindVertices.size(); ++i) {
                    glm::vec4 p = torsoModel * glm::vec4(shirtMesh.bindVertices[i].position, 1.0f);
                    shirtMesh.vertices[i].position = glm::vec3(p);
                }

                SleeveRegion leftLocal {
                    shirtMesh.leftShoulderPos,
                    shirtMesh.leftElbowPos,
                    shirtMesh.leftWristPos,
                    true
                };

                SleeveRegion rightLocal {
                    shirtMesh.rightShoulderPos,
                    shirtMesh.rightElbowPos,
                    shirtMesh.rightWristPos,
                    false
                };

                glm::vec3 torsoLeftShoulder =
                    glm::vec3(torsoModel * glm::vec4(shirtMesh.leftShoulderPos, 1.0f));

                glm::vec3 torsoRightShoulder =
                    glm::vec3(torsoModel * glm::vec4(shirtMesh.rightShoulderPos, 1.0f));

                auto makeArmTarget = [&](const glm::vec3& torsoShoulder,
                                         const glm::vec3& screenElbow,
                                         const glm::vec3& screenWrist) {
                    glm::vec3 elbowOffset = screenElbow - torsoShoulder;
                    glm::vec3 wristOffset = screenWrist - torsoShoulder;

                    elbowOffset.z = 0.0f;
                    wristOffset.z = 0.0f;

                    return std::pair<glm::vec3, glm::vec3>(
                        torsoShoulder + elbowOffset,
                        torsoShoulder + wristOffset
                    );
                };

                auto [fixedLE, fixedLW] = makeArmTarget(torsoLeftShoulder, sLE, sLW);
                auto [fixedRE, fixedRW] = makeArmTarget(torsoRightShoulder, sRE, sRW);

                SleeveRegion leftWorld {
                    torsoLeftShoulder,
                    fixedLE,
                    fixedLW,
                    true
                };

                SleeveRegion rightWorld {
                    torsoRightShoulder,
                    fixedRE,
                    fixedRW,
                    false
                };

                deformSleeveRegion(
                    shirtMesh,
                    torsoModel,
                    leftLocal,
                    leftWorld,
                    -1.0f,
                    0.65f,
                    0.18f,
                    0.20f
                );

                deformSleeveRegion(
                    shirtMesh,
                    torsoModel,
                    rightLocal,
                    rightWorld,
                    +1.0f,
                    0.65f,
                    0.18f,
                    0.20f
                );

                shirtMesh.uploadDeformedVertices();

                glColor3f(0.7f, 0.82f, 0.95f);
                shirtMesh.draw();
            }
        }

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    tracker.stop();
    glDeleteTextures(1, &camTex);
    camera.release();
    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}