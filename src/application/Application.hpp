#pragma once

#include "app-context/VulkanApplicationContext.hpp"

#include "utils/logger/Logger.hpp"

#include <cstdint>
#include <memory>
#include <vector>

class Window;
class SvoBuilder;
class SvoTracer;
class ImguiManager;
class Camera;
class FpsSink;
class ShaderFileWatchListener;
class Application {
public:
  Application();
  ~Application();

  // disable move and copy
  Application(const Application &)            = delete;
  Application &operator=(const Application &) = delete;
  Application(Application &&)                 = delete;
  Application &operator=(Application &&)      = delete;

  Camera *getCamera();
  void run();

private:
  Logger _logger;

  std::unique_ptr<FpsSink> _fpsSink;
  std::unique_ptr<ImguiManager> _imguiManager;
  std::unique_ptr<Camera> _camera;
  std::unique_ptr<Window> _window;
  std::unique_ptr<ShaderFileWatchListener> _shaderFileWatchListener;

  VulkanApplicationContext *_appContext;

  std::unique_ptr<SvoBuilder> _svoBuilder;
  std::unique_ptr<SvoTracer> _svoTracer;

  // semaphores and fences for synchronization
  VkFence _svoBuildingDoneFence = VK_NULL_HANDLE;
  std::vector<VkSemaphore> _imageAvailableSemaphores;
  std::vector<VkSemaphore> _renderFinishedSemaphores;
  std::vector<VkFence> _framesInFlightFences;

  void _createSemaphoresAndFences();
  void _onSwapchainResize();
  void _waitForTheWindowToBeResumed();
  void _drawFrame();
  void _mainLoop();
  void _init();
  void _cleanup();
  bool _needToToggleWindowStyle();
};