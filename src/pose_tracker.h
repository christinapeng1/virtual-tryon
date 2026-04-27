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
    bool getUpperBody(glm::vec2& leftShoulder, glm::vec2& rightShoulder,
                      glm::vec2& leftElbow, glm::vec2& rightElbow,
                      glm::vec2& leftWrist, glm::vec2& rightWrist);

private:
    void readerLoop();

    FILE* pipe;
    std::atomic<bool> running;
    std::thread readerThread;
    std::mutex dataMutex;
    glm::vec2 latestLeftShoulder;
    glm::vec2 latestRightShoulder;
    glm::vec2 latestLeftElbow;
    glm::vec2 latestRightElbow;
    glm::vec2 latestLeftWrist;
    glm::vec2 latestRightWrist;
    bool hasPose;
};

#endif // POSE_TRACKER_H
