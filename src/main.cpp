#include <iostream>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <GLFW/glfw3.h>
#include <opencv2/opencv.hpp>

#include "mesh.h"
#include "pose_tracker.h"
#include "utils.h"

int main() {
    std::cout << "Initializing...\n";

    if (!glfwInit()) {
        return -1;
    }

#ifdef __APPLE__
    glfwWindowHint(GLFW_COCOA_RETINA_FRAMEBUFFER, GLFW_FALSE);
#endif

    GLFWwindow* window = glfwCreateWindow(1280, 720, "AR Mesh on Torso", nullptr, nullptr);
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

    // Camera texture
    GLuint camTex = 0;
    glGenTextures(1, &camTex);
    glBindTexture(GL_TEXTURE_2D, camTex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, initW, initH, 0, GL_RGB, GL_UNSIGNED_BYTE, frame.data);

    // Load mesh
    Mesh shirtMesh = loadMesh("../meshes/jeans_denim_jacket.glb");

    // Setup
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

    // Pose tracker
    PoseTracker tracker;
    bool trackerReady = tracker.init();
    if (trackerReady) std::cout << "Pose tracker ready\n";

    glm::vec2 leftShoulder(0.35f, 0.4f);
    glm::vec2 rightShoulder(0.65f, 0.4f);
    float depth = -2.5f;

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
        depth = glm::clamp(depth, -8.0f, -1.5f);

        if (trackerReady) {
            glm::vec2 tl, tr;
            if (tracker.getShoulders(tl, tr)) {
                leftShoulder = tl;
                rightShoulder = tr;
            }
        }

        camera >> frame;
        if (frame.empty()) break;

        cv::flip(frame, frame, 1);
        cv::cvtColor(frame, frame, cv::COLOR_BGR2RGB);
        cv::resize(frame, frame, cv::Size(w, h));

        glBindTexture(GL_TEXTURE_2D, camTex);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, w, h, GL_RGB, GL_UNSIGNED_BYTE, frame.data);

        glm::vec3 worldLeft = screenToWorld(leftShoulder, depth, aspect);
        glm::vec3 worldRight = screenToWorld(rightShoulder, depth, aspect);
        glm::vec3 torsoCenter = (worldLeft + worldRight) * 0.5f;

        glm::vec3 shoulderVec = worldRight - worldLeft;
        float shoulderDist = glm::length(shoulderVec);

        glViewport(0, 0, w, h);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // ---------- Draw camera background in 2D ----------
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

        // Draw 2D shoulder points
        glPointSize(10.0f);
        glColor3f(0.0f, 1.0f, 0.0f);
        glBegin(GL_POINTS);
            glVertex2f(leftShoulder.x * w,  leftShoulder.y * h);
            glVertex2f(rightShoulder.x * w, rightShoulder.y * h);
        glEnd();

        // ---------- Switch to 3D ----------
        glEnable(GL_DEPTH_TEST);
        glEnable(GL_LIGHTING);

        glMatrixMode(GL_PROJECTION);
        glm::mat4 proj = glm::perspective(glm::radians(45.0f), aspect, 0.1f, 100.0f);
        glLoadMatrixf(glm::value_ptr(proj));

        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();

        glLightfv(GL_LIGHT0, GL_POSITION, lightPos);

        // Draw joint spheres
        glColor3f(0.25f, 0.55f, 1.0f);
        drawSphere(worldLeft, 0.06f);
        drawSphere(worldRight, 0.06f);

        // Draw mesh at torso - anchored to shoulders
        if (shirtMesh.isValid) {
            glColor3f(0.7f, 0.82f, 0.95f);

            glm::vec3 meshShoulderVec = shirtMesh.rightShoulderPos - shirtMesh.leftShoulderPos;
            glm::vec3 meshShoulderCenter = (shirtMesh.leftShoulderPos + shirtMesh.rightShoulderPos) * 0.5f;
            float meshShoulderWidth = glm::length(meshShoulderVec);

            if (shoulderDist > 1e-5f && meshShoulderWidth > 1e-5f) {
                glm::vec3 targetDir = glm::normalize(shoulderVec);
                glm::vec3 meshDir = glm::normalize(meshShoulderVec);

                glm::mat4 model(1.0f);

                // Move mesh local shoulder center to origin
                model = glm::translate(model, -meshShoulderCenter);

                // Rotate mesh shoulder line to tracked shoulder line
                float d = glm::clamp(glm::dot(meshDir, targetDir), -1.0f, 1.0f);
                float angle = std::acos(d);
                glm::vec3 axis = glm::cross(meshDir, targetDir);

                if (glm::length(axis) > 1e-6f && angle > 1e-6f) {
                    model = glm::rotate(glm::mat4(1.0f), angle, glm::normalize(axis)) * model;
                }

                // Scale mesh to actual shoulder width
                float scaleFactor = shoulderDist / meshShoulderWidth;
                scaleFactor *= 3.f;

                // Flip Y if the asset needs it
                model = glm::scale(glm::mat4(1.0f), glm::vec3(scaleFactor, scaleFactor, scaleFactor)) * model;

                // Place it at torso center with a slight downward offset
                glm::vec3 finalPos = torsoCenter + glm::vec3(0.0f, 0.08f * scaleFactor, 0.0f);
                model = glm::translate(glm::mat4(1.0f), finalPos) * model;

                glPushMatrix();
                glMultMatrixf(glm::value_ptr(model));
                shirtMesh.draw();
                glPopMatrix();
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