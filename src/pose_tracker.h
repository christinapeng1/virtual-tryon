#ifndef POSE_TRACKER_H
#define POSE_TRACKER_H

#include <glm/glm.hpp>
#include <thread>
#include <mutex>
#include <atomic>
#include <cstdio>

class PoseTracker {
public:
    PoseTracker();
    ~PoseTracker();

    bool init();
    void stop();
    bool getShoulders(
        glm::vec2& leftShoulder2D,
        glm::vec2& rightShoulder2D,
        glm::vec3& leftShoulder3D,
        glm::vec3& rightShoulder3D,
        float& yawDegrees,
        float& visibility,
        float& worldShoulderWidth
    );

private:
    void readerLoop();

    FILE* pipe;
    std::atomic<bool> running;
    std::thread readerThread;
    std::mutex dataMutex;
    glm::vec2 latestLeftShoulder2D;
    glm::vec2 latestRightShoulder2D;
    glm::vec3 latestLeftShoulder3D;
    glm::vec3 latestRightShoulder3D;
    float latestYaw;
    float latestVisibility;
    float latestWorldShoulderWidth;
    bool hasPose;
};

#endif // POSE_TRACKER_H
