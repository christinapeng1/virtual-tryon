#include "pose_tracker.h"
#include <iostream>
#include <chrono>
#include <cstring>

PoseTracker::PoseTracker() : pipe(nullptr), running(false), hasPose(false),
    latestLeftShoulder(0.0f), latestRightShoulder(0.0f) {}

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

void PoseTracker::readerLoop() {
    char buffer[256];
    while (running && pipe) {
        if (fgets(buffer, sizeof(buffer), pipe) == nullptr) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }

        float lsx, lsy, rsx, rsy;
        int result = sscanf(buffer, "%f,%f,%f,%f", &lsx, &lsy, &rsx, &rsy);
        if (result == 4) {
            std::lock_guard<std::mutex> lock(dataMutex);
            latestLeftShoulder = {lsx, lsy};
            latestRightShoulder = {rsx, rsy};
            hasPose = true;
        }
    }
}
