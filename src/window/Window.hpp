#pragma once

#include "GLFW/glfw3.h"

#ifdef APIENTRY
#undef APIENTRY
#endif

#include "volk.h"

#include "CursorInfo.hpp"

#include <functional>
#include <map>
#include <vector>

enum class WindowStyle { kNone, kFullScreen, kMaximized, kHover };
enum class CursorState { kNone, kInvisible, kVisible };

class Window {
public:
  Window(WindowStyle windowStyle, int widthIfWindowed = 400, int heightIfWindowed = 300);
  ~Window();

  // disable move and copy
  Window(const Window &)            = delete;
  Window &operator=(const Window &) = delete;
  Window(Window &&)                 = delete;
  Window &operator=(Window &&)      = delete;

  [[nodiscard]] GLFWwindow *getGlWindow() const { return _window; }
  [[nodiscard]] GLFWmonitor *getMonitor() const { return _monitor; }
  [[nodiscard]] bool isInputBitActive(int inputBit) {
    return _keyInputMap.contains(inputBit) && _keyInputMap[inputBit];
  }

  [[nodiscard]] WindowStyle getWindowStyle() const { return _windowStyle; }
  [[nodiscard]] CursorState getCursorState() const { return _cursorState; }
  [[nodiscard]] bool windowSizeChanged() const { return _windowSizeChanged; }

  // be careful to use these two functions, you might want to query the
  // framebuffer size, not the window size
  [[nodiscard]] int getWindowWidth() const {
    int width  = 0;
    int height = 0;
    glfwGetWindowSize(_window, &width, &height);
    return width;
  }

  [[nodiscard]] int getWindowHeight() const {
    int width  = 0;
    int height = 0;
    glfwGetWindowSize(_window, &width, &height);
    return height;
  }

  [[nodiscard]] int getFrameBufferWidth() const {
    int width  = 0;
    int height = 0;
    glfwGetFramebufferSize(_window, &width, &height);
    return width;
  }

  [[nodiscard]] int getFrameBufferHeight() const {
    int width  = 0;
    int height = 0;
    glfwGetFramebufferSize(_window, &width, &height);
    return height;
  }

  [[nodiscard]] int getCursorXPos() const {
    double xPos = NAN;
    double yPos = NAN;
    glfwGetCursorPos(_window, &xPos, &yPos);
    return static_cast<int>(xPos);
  }

  [[nodiscard]] int getCursorYPos() const {
    double xPos = NAN;
    double yPos = NAN;
    glfwGetCursorPos(_window, &xPos, &yPos);
    return static_cast<int>(yPos);
  }

  void toggleWindowStyle();

  void setWindowStyle(WindowStyle newStyle);

  void setWindowSizeChanged(bool windowSizeChanged) { _windowSizeChanged = windowSizeChanged; }

  void showCursor();
  void hideCursor();
  void toggleCursor();

  void disableInputBit(int bitToBeDisabled) { _keyInputMap[bitToBeDisabled] = false; }

  void addCursorMoveCallback(std::function<void(CursorMoveInfo const &)> callback);
  void addCursorButtonCallback(std::function<void(CursorInfo const &)> callback);

  CursorInfo getCursorInfo() const { return _cursorInfo; }

private:
  WindowStyle _windowStyle = WindowStyle::kNone;
  CursorState _cursorState = CursorState::kInvisible;

  int _widthIfWindowed;
  int _heightIfWindowed;
  std::map<int, bool> _keyInputMap;

  bool _windowSizeChanged = false;

  GLFWwindow *_window   = nullptr;
  GLFWmonitor *_monitor = nullptr;

  CursorInfo _cursorInfo;

  // these are used to restore maximized window to its original size and pos
  int _titleBarHeight                  = 0;
  int _maximizedFullscreenClientWidth  = 0;
  int _maximizedFullscreenClientHeight = 0;

  std::vector<std::function<void(CursorMoveInfo)>> _cursorMoveCallbacks;
  std::vector<std::function<void(CursorInfo)>> _cursorButtonCallbacks;

  // these functions are restricted to be static functions
  static void _keyCallback(GLFWwindow *window, int key, int scancode, int action, int mods);
  static void _cursorPosCallback(GLFWwindow *window, double xPos, double yPos);
  static void _mouseButtonCallback(GLFWwindow *window, int button, int action, int mods);
  static void _frameBufferResizeCallback(GLFWwindow *window, int width, int height);
};