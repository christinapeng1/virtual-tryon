#include "utils.h"
#include <iostream>
#include <cmath>

glm::vec3 screenToWorld(glm::vec2 screenPos, float depth, float aspect) {
    glm::vec2 ndc = screenPos * 2.0f - 1.0f;
    ndc.y = -ndc.y;

    const float fovY = glm::radians(45.0f);
    float tanHalfFov = glm::tan(fovY * 0.5f);

    float z = depth; // should be negative
    float x = ndc.x * std::abs(z) * tanHalfFov * aspect;
    float y = ndc.y * std::abs(z) * tanHalfFov;

    return glm::vec3(x, y, z);
}

void drawSphere(glm::vec3 pos, float radius) {
    glPushMatrix();
    glTranslatef(pos.x, pos.y, pos.z);

    int stacks = 8, slices = 8;
    for (int i = 0; i < stacks; i++) {
        float phi1 = (float)M_PI * (-0.5f + (float)i / (float)stacks);
        float phi2 = (float)M_PI * (-0.5f + (float)(i + 1) / (float)stacks);

        float y1 = std::sin(phi1);
        float r1 = std::cos(phi1);
        float y2 = std::sin(phi2);
        float r2 = std::cos(phi2);

        glBegin(GL_QUAD_STRIP);
        for (int j = 0; j <= slices; j++) {
            float theta = 2.0f * (float)M_PI * (float)j / (float)slices;
            float x = std::cos(theta);
            float z = std::sin(theta);

            glNormal3f(r1 * x, y1, r1 * z);
            glVertex3f(radius * r1 * x, radius * y1, radius * r1 * z);

            glNormal3f(r2 * x, y2, r2 * z);
            glVertex3f(radius * r2 * x, radius * y2, radius * r2 * z);
        }
        glEnd();
    }

    glPopMatrix();
}
