#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <sstream>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <GLFW/glfw3.h>
#include <opencv2/opencv.hpp>

#include "mesh.h"
#include "pose_tracker.h"
#include "utils.h"

/*
    WHY THIS VERSION FIXES THE BUG
    ==============================
    The previous attempt rendered the same jacket multiple times with different transforms,
    which creates multiple whole shirts.

    This version keeps ONE jacket mesh and deforms only the sleeve vertex regions in place.

    REQUIRED MESH SUPPORT
    =====================
    Your Mesh class needs to expose a bind/rest-pose copy and a writable deformed copy.
    Minimal API expected by this file:

        struct Vertex {
            glm::vec3 position;
            glm::vec3 normal;
            glm::vec2 uv;
        };

        struct Mesh {
            bool isValid = false;

            std::vector<Vertex> bindVertices;   // original vertices from asset
            std::vector<Vertex> vertices;       // deformed vertices for current frame
            std::vector<unsigned int> indices;

            glm::vec3 leftShoulderPos;
            glm::vec3 rightShoulderPos;
            glm::vec3 leftElbowPos;
            glm::vec3 rightElbowPos;
            glm::vec3 leftWristPos;
            glm::vec3 rightWristPos;

            void resetToBindPose();
            void uploadDeformedVertices();
            void draw();
        };

    REQUIRED POSE TRACKER SUPPORT
    =============================
        bool getUpperBody(
            glm::vec2& leftShoulder,
            glm::vec2& rightShoulder,
            glm::vec2& leftElbow,
            glm::vec2& rightElbow,
            glm::vec2& leftWrist,
            glm::vec2& rightWrist);
*/

static glm::vec3 smoothVec3(const glm::vec3& prev, const glm::vec3& curr, float alpha) {
    return prev * (1.0f - alpha) + curr * alpha;
}

static glm::vec3 safeNormalize(const glm::vec3& v, const glm::vec3& fallback = glm::vec3(1.0f, 0.0f, 0.0f)) {
    float len = glm::length(v);
    if (len < 1e-6f) return fallback;
    return v / len;
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

    return glm::translate(glm::mat4(1.0f), worldA) * R * S * glm::translate(glm::mat4(1.0f), -localA);
}

static float smoothstepf(float a, float b, float x) {
    if (std::abs(b - a) < 1e-6f) return x < a ? 0.0f : 1.0f;
    float t = glm::clamp((x - a) / (b - a), 0.0f, 1.0f);
    return t * t * (3.0f - 2.0f * t);
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
    float lowerLen = glm::max(glm::length(local.wrist - local.elbow), 1e-5f);
    
    // Debug: Count how many vertices we're deforming
    int deformCount = 0;
    std::string armSide = (sideSign < 0.0f) ? "LEFT" : "RIGHT";

    for (size_t i = 0; i < mesh.vertices.size() && i < mesh.bindVertices.size(); ++i) {
        const glm::vec3 p = mesh.bindVertices[i].position;

        bool isUpper = false;
        bool isLower = false;

        if (sideSign < 0.0f) {
            isUpper = (mesh.vertexRegion[i] == REGION_LEFT_UPPER_SLEEVE);
            isLower = (mesh.vertexRegion[i] == REGION_LEFT_LOWER_SLEEVE);
        } else {
            isUpper = (mesh.vertexRegion[i] == REGION_RIGHT_UPPER_SLEEVE);
            isLower = (mesh.vertexRegion[i] == REGION_RIGHT_LOWER_SLEEVE);
        }

        if (!isUpper && !isLower) {
            continue;
        }
        
        deformCount++;

        glm::vec3 rel = p - local.shoulder;
        float axial = glm::dot(rel, armAxis);

        float shoulderToUpper = smoothstepf(0.0f, torsoBlendWidth, axial);
        float upperToLower = smoothstepf(upperLen - upperBlendWidth, upperLen + lowerBlendWidth, axial);

        glm::vec4 p4(p, 1.0f);
        glm::vec3 torsoP = glm::vec3(torsoModel * p4);
        glm::vec3 upperP = glm::vec3(upperT * p4);
        glm::vec3 lowerP = glm::vec3(lowerT * p4);
        
        // Blend through all three levels hierarchically
        glm::vec3 finalP = torsoP;
        
        // Step 1: Blend toward upper arm
        glm::vec3 atUpper = glm::mix(torsoP, upperP, shoulderToUpper);
        
        if (isUpper) {
            // Upper arm vertices: use the torso->upper blend
            finalP = atUpper;
        } else {
            // Lower arm vertices: blend from upper toward lower
            finalP = glm::mix(atUpper, lowerP, upperToLower);
        }

        mesh.vertices[i].position = finalP;
    }
    
    std::cout << "Deformed " << deformCount << " vertices for " << armSide << " sleeve\n";
    
    // Debug: Count lower sleeve vertices specifically
    int upperCount = 0, lowerCount = 0;
    if (sideSign < 0.0f) {
        for (size_t i = 0; i < mesh.vertexRegion.size(); ++i) {
            if (mesh.vertexRegion[i] == REGION_LEFT_UPPER_SLEEVE) upperCount++;
            if (mesh.vertexRegion[i] == REGION_LEFT_LOWER_SLEEVE) lowerCount++;
        }
    } else {
        for (size_t i = 0; i < mesh.vertexRegion.size(); ++i) {
            if (mesh.vertexRegion[i] == REGION_RIGHT_UPPER_SLEEVE) upperCount++;
            if (mesh.vertexRegion[i] == REGION_RIGHT_LOWER_SLEEVE) lowerCount++;
        }
    }
    std::cout << "  -> Upper: " << upperCount << ", Lower: " << lowerCount << "\n";
}

static void loadCalibrationIfAvailable(Mesh& mesh, const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        std::cout << "No calibration file found at " << path << ", using defaults\n";
        return;
    }

    std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    file.close();

    // Simple JSON parsing for anchors
    size_t anchorsStart = content.find("\"anchors\"");
    if (anchorsStart == std::string::npos) {
        std::cout << "No anchors in calibration file\n";
        return;
    }

    auto parseVec3 = [](const std::string& text, size_t start) -> glm::vec3 {
        size_t bracketStart = text.find("[", start);
        size_t bracketEnd = text.find("]", bracketStart);
        std::string vecStr = text.substr(bracketStart + 1, bracketEnd - bracketStart - 1);
        
        glm::vec3 v;
        int count = std::sscanf(vecStr.c_str(), "%f, %f, %f", &v.x, &v.y, &v.z);
        if (count != 3) std::cout << "Failed to parse vec3\n";
        return v;
    };

    // Extract each joint position
    std::cout << "Loading calibration...\n";
    mesh.leftShoulderPos = parseVec3(content, content.find("\"leftShoulder\""));
    mesh.leftElbowPos = parseVec3(content, content.find("\"leftElbow\""));
    mesh.leftWristPos = parseVec3(content, content.find("\"leftWrist\""));
    mesh.rightShoulderPos = parseVec3(content, content.find("\"rightShoulder\""));
    mesh.rightElbowPos = parseVec3(content, content.find("\"rightElbow\""));
    mesh.rightWristPos = parseVec3(content, content.find("\"rightWrist\""));
    
    std::cout << "Joint positions loaded:\n"
              << "  Left: shoulder=" << mesh.leftShoulderPos.x << "," << mesh.leftShoulderPos.y 
              << " elbow=" << mesh.leftElbowPos.x << "," << mesh.leftElbowPos.y 
              << " wrist=" << mesh.leftWristPos.x << "," << mesh.leftWristPos.y << "\n"
              << "  Right: shoulder=" << mesh.rightShoulderPos.x << "," << mesh.rightShoulderPos.y 
              << " elbow=" << mesh.rightElbowPos.x << "," << mesh.rightElbowPos.y 
              << " wrist=" << mesh.rightWristPos.x << "," << mesh.rightWristPos.y << "\n";

    // Parse vertexRegion array
    size_t regionStart = content.find("\"vertexRegion\"");
    if (regionStart != std::string::npos) {
        size_t arrayStart = content.find("[", regionStart);
        size_t arrayEnd = content.find("]", arrayStart);
        std::string arrayStr = content.substr(arrayStart + 1, arrayEnd - arrayStart - 1);
        std::istringstream iss(arrayStr);
        
        mesh.vertexRegion.clear();
        int value;
        int leftCount = 0, rightCount = 0;
        while (iss >> value) {
            mesh.vertexRegion.push_back(value);
            if (value == REGION_LEFT_UPPER_SLEEVE || value == REGION_LEFT_LOWER_SLEEVE) leftCount++;
            if (value == REGION_RIGHT_UPPER_SLEEVE || value == REGION_RIGHT_LOWER_SLEEVE) rightCount++;
            if (iss.peek() == ',') iss.ignore();
        }
        std::cout << "Loaded " << mesh.vertexRegion.size() << " vertex regions: " 
                  << leftCount << " left sleeve, " << rightCount << " right sleeve\n";
    }

    std::cout << "Calibration loaded successfully\n";
}

int main() {
    std::cout << "Initializing...\n";

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

    Mesh shirtMesh = loadMesh("../meshes/jeans_denim_jacket.glb");

    // Load calibration data if available
    loadCalibrationIfAvailable(shirtMesh, "calibration.json");

    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_LIGHTING);
    glEnable(GL_LIGHT0);
    glEnable(GL_COLOR_MATERIAL);
    glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);
    glDisable(GL_CULL_FACE);
    glShadeModel(GL_SMOOTH);

    GLfloat lightPos[] = {0.0f, 0.0f, 2.0f, 1.0f};
    GLfloat lightDiffuse[] = {0.9f, 0.9f, 0.9f, 1.0f};
    glLightfv(GL_LIGHT0, GL_POSITION, lightPos);
    glLightfv(GL_LIGHT0, GL_DIFFUSE, lightDiffuse);

    PoseTracker tracker;
    bool trackerReady = tracker.init();
    if (trackerReady) std::cout << "Pose tracker ready\n";

    glm::vec2 leftShoulder(0.35f, 0.40f);
    glm::vec2 rightShoulder(0.65f, 0.40f);
    glm::vec2 leftElbow(0.28f, 0.54f);
    glm::vec2 rightElbow(0.72f, 0.54f);
    glm::vec2 leftWrist(0.24f, 0.70f);
    glm::vec2 rightWrist(0.76f, 0.70f);

    float depth = -2.5f;

    glm::vec3 sLS(0.0f), sRS(0.0f), sLE(0.0f), sRE(0.0f), sLW(0.0f), sRW(0.0f);
    bool smoothingInitialized = false;
    const float jointSmoothAlpha = 0.35f;

    while (!glfwWindowShouldClose(window)) {
        int w, h;
        glfwGetWindowSize(window, &w, &h);
        if (w <= 0 || h <= 0) {
            glfwPollEvents();
            continue;
        }

        float aspect = w / (float)h;

        if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
            glfwSetWindowShouldClose(window, GLFW_TRUE);
        }

        leftShoulder = glm::clamp(leftShoulder, glm::vec2(0.0f), glm::vec2(1.0f));
        rightShoulder = glm::clamp(rightShoulder, glm::vec2(0.0f), glm::vec2(1.0f));
        leftElbow = glm::clamp(leftElbow, glm::vec2(0.0f), glm::vec2(1.0f));
        rightElbow = glm::clamp(rightElbow, glm::vec2(0.0f), glm::vec2(1.0f));
        leftWrist = glm::clamp(leftWrist, glm::vec2(0.0f), glm::vec2(1.0f));
        rightWrist = glm::clamp(rightWrist, glm::vec2(0.0f), glm::vec2(1.0f));
        depth = glm::clamp(depth, -8.0f, -1.5f);

        if (trackerReady) {
            glm::vec2 tls, trs, tle, tre, tlw, trw;
            if (tracker.getUpperBody(tls, trs, tle, tre, tlw, trw)) {
                leftShoulder = tls;
                rightShoulder = trs;
                leftElbow = tle;
                rightElbow = tre;
                leftWrist = tlw;
                rightWrist = trw;
            }
        }

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

        if (!smoothingInitialized) {
            sLS = wsLS; sRS = wsRS; sLE = wsLE; sRE = wsRE; sLW = wsLW; sRW = wsRW;
            smoothingInitialized = true;
        } else {
            sLS = smoothVec3(sLS, wsLS, jointSmoothAlpha);
            sRS = smoothVec3(sRS, wsRS, jointSmoothAlpha);
            sLE = smoothVec3(sLE, wsLE, jointSmoothAlpha);
            sRE = smoothVec3(sRE, wsRE, jointSmoothAlpha);
            sLW = smoothVec3(sLW, wsLW, jointSmoothAlpha);
            sRW = smoothVec3(sRW, wsRW, jointSmoothAlpha);
        }

        glm::vec3 torsoCenter = (sLS + sRS) * 0.5f;
        glm::vec3 shoulderVec = sRS - sLS;
        float shoulderDist = glm::length(shoulderVec);

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

        // Debug visualization spheres - disabled for cleaner view
        // drawSphere(sLS, 0.05f);
        // drawSphere(sRS, 0.05f);
        // drawSphere(sLE, 0.04f);
        // drawSphere(sRE, 0.04f);
        // drawSphere(sLW, 0.035f);
        // drawSphere(sRW, 0.035f);

        if (shirtMesh.isValid) {
            glm::vec3 meshShoulderVec = shirtMesh.rightShoulderPos - shirtMesh.leftShoulderPos;
            glm::vec3 meshShoulderCenter = (shirtMesh.leftShoulderPos + shirtMesh.rightShoulderPos) * 0.5f;
            float meshShoulderWidth = glm::length(meshShoulderVec);

            if (shoulderDist > 1e-5f && meshShoulderWidth > 1e-5f) {
                glm::vec3 targetDir = glm::normalize(shoulderVec);
                glm::vec3 meshDir = glm::normalize(meshShoulderVec);

                glm::mat4 torsoModel(1.0f);
                torsoModel = glm::translate(torsoModel, -meshShoulderCenter);

                float d = glm::clamp(glm::dot(meshDir, targetDir), -1.0f, 1.0f);
                float angle = std::acos(d);
                glm::vec3 axis = glm::cross(meshDir, targetDir);
                if (glm::length(axis) > 1e-6f && angle > 1e-6f) {
                    torsoModel = glm::rotate(glm::mat4(1.0f), angle, glm::normalize(axis)) * torsoModel;
                }

                float scaleFactor = shoulderDist / meshShoulderWidth;
                scaleFactor *= 1.0f;  // Reduced from 1.5f to make shirt smaller
                torsoModel = glm::scale(glm::mat4(1.0f), glm::vec3(scaleFactor, scaleFactor, scaleFactor)) * torsoModel;

                glm::vec3 finalPos = torsoCenter + glm::vec3(0.0f, -0.1f * scaleFactor, 0.0f);  // Changed to negative to move down
                torsoModel = glm::translate(glm::mat4(1.0f), finalPos) * torsoModel;

                // Single-mesh path: start from bind pose, convert all vertices by torso placement,
                // then overwrite only sleeve regions with arm-aware deformation.
                shirtMesh.resetToBindPose();
                for (size_t i = 0; i < shirtMesh.vertices.size() && i < shirtMesh.bindVertices.size(); ++i) {
                    glm::vec4 p = torsoModel * glm::vec4(shirtMesh.bindVertices[i].position, 1.0f);
                    shirtMesh.vertices[i].position = glm::vec3(p);
                }

                SleeveRegion leftLocal { shirtMesh.leftShoulderPos, shirtMesh.leftElbowPos, shirtMesh.leftWristPos, true };
                SleeveRegion rightLocal{ shirtMesh.rightShoulderPos, shirtMesh.rightElbowPos, shirtMesh.rightWristPos, false };
                SleeveRegion leftWorld { sLS, sLE, sLW, true };
                SleeveRegion rightWorld{ sRS, sRE, sRW, false };

                deformSleeveRegion(
                    shirtMesh,
                    torsoModel,
                    leftLocal,
                    leftWorld,
                    -1.0f,
                    0.12f,
                    0.08f,
                    0.16f
                );

                deformSleeveRegion(
                    shirtMesh,
                    torsoModel,
                    rightLocal,
                    rightWorld,
                    +1.0f,
                    0.12f,
                    0.08f,
                    0.16f
                );

                shirtMesh.uploadDeformedVertices();

                // DEBUG: Visualize mesh's elbow and wrist positions (transformed by torso)
                glColor3f(1.0f, 0.0f, 1.0f);  // Magenta for mesh elbows
                glm::vec4 meshLeftElbow4 = torsoModel * glm::vec4(shirtMesh.leftElbowPos, 1.0f);
                glm::vec4 meshRightElbow4 = torsoModel * glm::vec4(shirtMesh.rightElbowPos, 1.0f);
                drawSphere(glm::vec3(meshLeftElbow4), 0.04f);
                drawSphere(glm::vec3(meshRightElbow4), 0.04f);
                
                glColor3f(0.0f, 1.0f, 1.0f);  // Cyan for mesh wrists
                glm::vec4 meshLeftWrist4 = torsoModel * glm::vec4(shirtMesh.leftWristPos, 1.0f);
                glm::vec4 meshRightWrist4 = torsoModel * glm::vec4(shirtMesh.rightWristPos, 1.0f);
                drawSphere(glm::vec3(meshLeftWrist4), 0.035f);
                drawSphere(glm::vec3(meshRightWrist4), 0.035f);

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
