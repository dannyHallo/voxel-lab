#pragma once

#include "utils/incl/GlmIncl.hpp" // IWYU pragma: export
#include "window/Window.hpp"

// An abstract camera class that processes input and calculates the
// corresponding Euler Angles, Vectors and Matrices for use in OpenGL
class Camera {

public:
  Camera(Window *window, glm::vec3 camPosition = glm::vec3(0.F, 0.1F, 0.F),
         glm::vec3 wrdUp = glm::vec3(0.F, 1.F, 0.F), float camYaw = 270, float camPitch = 0,
         float vFov = 60.F)
      : _position(camPosition), _worldUp(wrdUp), _yaw(camYaw), _pitch(camPitch), _fov(vFov),
        _window(window) {
    _updateCameraVectors();
  }

  [[nodiscard]] glm::mat4 getViewMatrix() const {
    return glm::lookAt(_position, _position + _front, _up);
  }

  void processInput(double deltaTime);

  // processes input received from any keyboard-like input system. Accepts input
  // parameter in the form of camera defined ENUM (to abstract it from windowing
  // systems)
  void processKeyboard(double deltaTime);

  // processes input received from a mouse input system. Expects the offset
  // value in both the x and y direction.
  void handleMouseMovement(float xoffset, float yoffset);

  // processes input received from a mouse scroll-wheel event. Only requires
  // input on the vertical wheel-axis void processMouseScroll(float yoffset);
  [[nodiscard]] glm::mat4 getProjectionMatrix(float aspectRatio, float zNear = 0.1F,
                                              float zFar = 10000) const;

  [[nodiscard]] glm::dmat4 getProjectionMatrixDouble(float aspectRatio, float zNear = 0.1F,
                                                     float zFar = 10000) const;

  [[nodiscard]] glm::vec3 getPosition() const { return _position; }
  [[nodiscard]] glm::vec3 getFront() const { return _front; }
  [[nodiscard]] glm::vec3 getUp() const { return _up; }
  [[nodiscard]] glm::vec3 getRight() const { return _right; }
  [[nodiscard]] float getVFov() const { return _fov; }

private:
  glm::vec3 _position;
  glm::vec3 _worldUp;

  glm::vec3 _front = glm::vec3(0.F);
  glm::vec3 _up    = glm::vec3(0.F);
  glm::vec3 _right = glm::vec3(0.F);

  // euler angles
  float _yaw;
  float _pitch;
  float _fov;

  float _movementSpeedMultiplier = 1.F;

  // window is owned by ApplicationContext
  Window *_window;

  // calculates the front vector from the Camera's (updated) Euler Angles
  void _updateCameraVectors();
  [[nodiscard]] bool canMove() const {

    return _window->getCursorState() == CursorState::kInvisible;
  }
};
