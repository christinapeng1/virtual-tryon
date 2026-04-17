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
    bool getShoulders(glm::vec2& leftShoulder, glm::vec2& rightShoulder);

private:
    void readerLoop();

    FILE* pipe;
    std::atomic<bool> running;
    std::thread readerThread;
    std::mutex dataMutex;
    glm::vec2 latestLeftShoulder;
    glm::vec2 latestRightShoulder;
    bool hasPose;
};

#endif // POSE_TRACKER_H
