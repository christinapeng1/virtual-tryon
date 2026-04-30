#include "pose_tracker.h"
#include <iostream>
#include <chrono>
#include <cstring>

PoseTracker::PoseTracker()
    : pipe(nullptr), running(false), hasPose(false),
      latestYaw(0.0f), latestVisibility(0.0f), latestWorldShoulderWidth(0.0f) {}

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

bool PoseTracker::getShoulders(
    glm::vec2& ls2d, glm::vec2& rs2d,
    glm::vec3& ls3d, glm::vec3& rs3d,
    float& yaw, float& vis, float& wsw)
{
    std::lock_guard<std::mutex> lock(dataMutex);
    if (!hasPose) return false;
    ls2d = latestLeftShoulder2D;
    rs2d = latestRightShoulder2D;
    ls3d = latestLeftShoulder3D;
    rs3d = latestRightShoulder3D;
    yaw  = latestYaw;
    vis  = latestVisibility;
    wsw  = latestWorldShoulderWidth;
    return true;
}

void PoseTracker::readerLoop() {
    char buffer[512];
    while (running && pipe) {
        if (fgets(buffer, sizeof(buffer), pipe) == nullptr) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }

        float lsx, lsy, rsx, rsy;
        float wlsx, wlsy, wlsz, wrsx, wrsy, wrsz;
        float yaw, vis, wsw;

        int r = sscanf(buffer, "%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f",
            &lsx, &lsy, &rsx, &rsy,
            &wlsx, &wlsy, &wlsz,
            &wrsx, &wrsy, &wrsz,
            &yaw, &vis, &wsw);

        if (r == 13) {
            std::lock_guard<std::mutex> lock(dataMutex);
            latestLeftShoulder2D  = {lsx, lsy};
            latestRightShoulder2D = {rsx, rsy};
            latestLeftShoulder3D = {wlsx, wlsy, wlsz};
            latestRightShoulder3D = {wrsx, wrsy, wrsz};
            latestYaw = yaw;
            latestVisibility = vis;
            latestWorldShoulderWidth = wsw;
            hasPose = true;
        }
    }
}