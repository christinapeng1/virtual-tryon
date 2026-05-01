#include "pose_tracker.h"
#include <iostream>
#include <chrono>
#include <cstring>

PoseTracker::PoseTracker() : pipe(nullptr), running(false), hasPose(false),
    latestNose(0.0f),
    latestYaw(0.0f),
    latestVisibility(0.0f),
    latestWorldShoulderWidth(0.0f),
    latestLeftShoulder(0.0f), latestRightShoulder(0.0f),
    latestLeftElbow(0.0f), latestRightElbow(0.0f),
    latestLeftWrist(0.0f), latestRightWrist(0.0f),
    latestLeftHip(0.0f), latestRightHip(0.0f) {}


PoseTracker::~PoseTracker() {
    
    stop();
}

bool PoseTracker::init() {
    const char* scriptPath = "../pose_tracker.py";
    std::string cmd = std::string("python3 ") + scriptPath;

    pipe = popen(cmd.c_str(), "r");
    if (!pipe) {
        std::cerr << "Error: Failed to start pose tracker\n";
        return false;
    }

    running = true;
    readerThread = std::thread(&PoseTracker::readerLoop, this);
    return true;
}

bool PoseTracker::getRotation(float& yaw, float& visibility, float& worldShoulderWidth) {
    std::lock_guard<std::mutex> lock(dataMutex);
    if (!hasPose) return false;

    yaw = latestYaw;
    visibility = latestVisibility;
    worldShoulderWidth = latestWorldShoulderWidth;
    return true;
}

void PoseTracker::stop() {
    running = false;
    if (readerThread.joinable()) readerThread.join();
    if (pipe) {
        pclose(pipe);
        pipe = nullptr;
    }
}

bool PoseTracker::getShoulders(glm::vec2& leftShoulder, glm::vec2& rightShoulder) {
    std::lock_guard<std::mutex> lock(dataMutex);
    if (!hasPose) return false;
    leftShoulder = latestLeftShoulder;
    rightShoulder = latestRightShoulder;
    return true;
}

bool PoseTracker::getHips(glm::vec2& leftHip, glm::vec2& rightHip) {
    std::lock_guard<std::mutex> lock(dataMutex);
    if (!hasPose) return false;
    leftHip = latestLeftHip;
    rightHip = latestRightHip;
    return true;
}

bool PoseTracker::getNose(glm::vec2& nose) {
    std::lock_guard<std::mutex> lock(dataMutex);
    if (!hasPose) return false;
    nose = latestNose;
    return true;
}

bool PoseTracker::getUpperBody(glm::vec2& leftShoulder, glm::vec2& rightShoulder,
                              glm::vec2& leftElbow, glm::vec2& rightElbow,
                              glm::vec2& leftWrist, glm::vec2& rightWrist) {
    std::lock_guard<std::mutex> lock(dataMutex);
    if (!hasPose) return false;
    leftShoulder = latestLeftShoulder;
    rightShoulder = latestRightShoulder;
    leftElbow = latestLeftElbow;
    rightElbow = latestRightElbow;
    leftWrist = latestLeftWrist;
    rightWrist = latestRightWrist;
    return true;
}
void PoseTracker::readerLoop() {
    char buffer[512];

    while (running && pipe) {
        if (fgets(buffer, sizeof(buffer), pipe) == nullptr) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }

        float nx, ny, lsx, lsy, rsx, rsy;
        float lex, ley, rex, rey, lwx, lwy, rwx, rwy;
        float lhx, lhy, rhx, rhy;

        float yaw = 0.0f;
        float visibility = 0.0f;
        float worldShoulderWidth = 0.4f;

        int result = sscanf(
            buffer,
            "%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f",
            &nx, &ny,
            &lsx, &lsy,
            &rsx, &rsy,
            &lex, &ley,
            &rex, &rey,
            &lwx, &lwy,
            &rwx, &rwy,
            &lhx, &lhy,
            &rhx, &rhy,
            &yaw,
            &visibility,
            &worldShoulderWidth
        );

        if (result == 18 || result == 21) {
            std::lock_guard<std::mutex> lock(dataMutex);

            latestNose = {nx, ny};
            latestLeftShoulder = {lsx, lsy};
            latestRightShoulder = {rsx, rsy};
            latestLeftElbow = {lex, ley};
            latestRightElbow = {rex, rey};
            latestLeftWrist = {lwx, lwy};
            latestRightWrist = {rwx, rwy};
            latestLeftHip = {lhx, lhy};
            latestRightHip = {rhx, rhy};

            if (result == 21) {
                latestYaw = yaw;
                latestVisibility = visibility;
                latestWorldShoulderWidth = worldShoulderWidth;
            }

            hasPose = true;
        } else {
            std::cerr << "Pose parse failed. result=" << result
                      << " line=" << buffer << std::endl;
        }
    }
}