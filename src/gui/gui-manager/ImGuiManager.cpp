#include "ImGuiManager.hpp"

#include "app-context/VulkanApplicationContext.hpp"
#include "gui/gui-elements/FpsGui.hpp"
#include "render-context/RenderSystem.hpp"
#include "utils/config/RootDir.h"
#include "utils/logger/Logger.hpp"
#include "window/Window.hpp"

#include "imgui/backends/imgui_impl_glfw.h"
#include "imgui/backends/imgui_impl_vulkan.h"

float constexpr kImguiFontSize = 22.0F;

namespace {
void check_vk_result(VkResult resultCode) {
  assert(resultCode == VK_SUCCESS && "check_vk_result failed");
}

void comboSelector(std::string const &comboLabel, std::vector<std::string> const &outputItems,
                   uint32_t &selectedIdx) {
  assert(selectedIdx < outputItems.size() && "selectedIdx is out of range");
  char const *currentSelectedItem = outputItems[selectedIdx].c_str();
  if (ImGui::BeginCombo(comboLabel.c_str(), currentSelectedItem)) {
    for (int n = 0; n < outputItems.size(); n++) {
      bool isSelected         = n == selectedIdx;
      std::string const &item = outputItems[n];
      if (ImGui::Selectable(item.c_str(), isSelected)) {
        currentSelectedItem = item.c_str();
        selectedIdx         = n;
      }
      if (isSelected) {
        ImGui::SetItemDefaultFocus();
      }
    }
    ImGui::EndCombo();
  }
}

} // namespace

ImGuiManager::ImGuiManager(VulkanApplicationContext *appContext, Window *window, Logger *logger,
                           int framesInFlight)
    : _appContext(appContext), _window(window), _logger(logger),
      _fpsGui(std::make_unique<FpsGui>()) {

  _createGuiCommandBuffers(framesInFlight);
  _createGuiRenderPass();
  _createFramebuffers();
  _createGuiDescripterPool();

  _initImgui();
}

ImGuiManager::~ImGuiManager() {

  for (auto &guiCommandBuffer : _guiCommandBuffers) {
    vkFreeCommandBuffers(_appContext->getDevice(), _appContext->getGuiCommandPool(), 1,
                         &guiCommandBuffer);
  }

  vkDestroyRenderPass(_appContext->getDevice(), _guiPass, nullptr);

  _cleanupFrameBuffers();

  vkDestroyDescriptorPool(_appContext->getDevice(), _guiDescriptorPool, nullptr);
  ImGui_ImplVulkan_Shutdown();
}

void ImGuiManager::_cleanupFrameBuffers() {
  for (auto &guiFrameBuffer : _guiFrameBuffers) {
    vkDestroyFramebuffer(_appContext->getDevice(), guiFrameBuffer, nullptr);
  }
}

void ImGuiManager::cleanupSwapchainDimensionRelatedResources() { _cleanupFrameBuffers(); }

void ImGuiManager::createSwapchainDimensionRelatedResources() { _createFramebuffers(); }

void ImGuiManager::_initImgui() {
  // setup Dear ImGui context
  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
  ImGuiIO &io = ImGui::GetIO();
  io.Fonts->AddFontFromFileTTF((kPathToResourceFolder + "/fonts/OverpassMono-Medium.ttf").c_str(),
                               kImguiFontSize);

  io.ConfigFlags |= ImGuiWindowFlags_NoNavInputs;

  ImGui::StyleColorsClassic();

  // Setup Platform/Renderer bindings
  ImGui_ImplGlfw_InitForVulkan(_window->getGlWindow(), true);

  ImGui_ImplVulkan_InitInfo info = {};
  info.Instance                  = _appContext->getVkInstance();
  info.PhysicalDevice            = _appContext->getPhysicalDevice();
  info.Device                    = _appContext->getDevice();
  info.QueueFamily               = _appContext->getQueueFamilyIndices().graphicsFamily;
  info.Queue                     = _appContext->getGraphicsQueue();
  info.PipelineCache             = VK_NULL_HANDLE;
  info.DescriptorPool            = _guiDescriptorPool;
  info.Allocator                 = VK_NULL_HANDLE;
  info.MinImageCount             = static_cast<uint32_t>(_appContext->getSwapchainSize());
  info.ImageCount                = static_cast<uint32_t>(_appContext->getSwapchainSize());
  info.CheckVkResultFn           = check_vk_result;
  if (!ImGui_ImplVulkan_Init(&info, _guiPass)) {
    _logger->print("failed to init impl");
  }

  // Create fonts texture
  VkCommandBuffer commandBuffer = RenderSystem::beginSingleTimeCommands(
      _appContext->getDevice(), _appContext->getCommandPool());

  if (!ImGui_ImplVulkan_CreateFontsTexture(commandBuffer)) {
    _logger->print("failed to create fonts texture");
  }
  RenderSystem::endSingleTimeCommands(_appContext->getDevice(), _appContext->getCommandPool(),
                                      _appContext->getGraphicsQueue(), commandBuffer);
}

void ImGuiManager::_createGuiDescripterPool() {
  int constexpr kMaxDescriptorCount           = 1000;
  std::vector<VkDescriptorPoolSize> poolSizes = {
      {VK_DESCRIPTOR_TYPE_SAMPLER, kMaxDescriptorCount},
      {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, kMaxDescriptorCount},
      {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, kMaxDescriptorCount},
      {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, kMaxDescriptorCount},
      {VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER, kMaxDescriptorCount},
      {VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER, kMaxDescriptorCount},
      {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, kMaxDescriptorCount},
      {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, kMaxDescriptorCount},
      {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, kMaxDescriptorCount},
      {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC, kMaxDescriptorCount},
      {VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT, kMaxDescriptorCount},
  };

  VkDescriptorPoolCreateInfo poolInfo = {};
  poolInfo.sType                      = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  // this descriptor pool is created only for once, so we can set the flag to allow individual
  // descriptor sets to be de-allocated
  poolInfo.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
  // imgui actually uses only 1 descriptor set
  poolInfo.maxSets       = kMaxDescriptorCount * poolSizes.size();
  poolInfo.pPoolSizes    = poolSizes.data();
  poolInfo.poolSizeCount = poolSizes.size();

  VkResult result =
      vkCreateDescriptorPool(_appContext->getDevice(), &poolInfo, nullptr, &_guiDescriptorPool);
  assert(result == VK_SUCCESS && "vkCreateDescriptorPool failed");
}

void ImGuiManager::_createGuiCommandBuffers(int framesInFlight) {
  _guiCommandBuffers.resize(framesInFlight);
  VkCommandBufferAllocateInfo allocInfo{};
  allocInfo.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  allocInfo.commandPool        = _appContext->getGuiCommandPool();
  allocInfo.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  allocInfo.commandBufferCount = (uint32_t)_guiCommandBuffers.size();

  VkResult result =
      vkAllocateCommandBuffers(_appContext->getDevice(), &allocInfo, _guiCommandBuffers.data());
  assert(result == VK_SUCCESS && "vkAllocateCommandBuffers failed");
}

void ImGuiManager::_createGuiRenderPass() {
  // Imgui Pass, right after the main pass
  VkAttachmentDescription attachment = {};
  attachment.format                  = _appContext->getSwapchainImageFormat();
  attachment.samples                 = VK_SAMPLE_COUNT_1_BIT;
  // Load onto the current render pass
  attachment.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
  // Store img until display time
  attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
  // No stencil
  attachment.stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
  attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
  attachment.initialLayout  = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
  // Present image right after this pass
  attachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

  VkAttachmentReference colorAttachment = {};
  colorAttachment.attachment            = 0;
  colorAttachment.layout                = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

  VkSubpassDescription subpass = {};
  subpass.pipelineBindPoint    = VK_PIPELINE_BIND_POINT_GRAPHICS;
  subpass.colorAttachmentCount = 1;
  subpass.pColorAttachments    = &colorAttachment;

  VkSubpassDependency dependency = {};
  dependency.srcSubpass          = VK_SUBPASS_EXTERNAL;
  dependency.dstSubpass          = 0;
  dependency.srcStageMask        = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
  dependency.dstStageMask        = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
  dependency.srcAccessMask       = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
  dependency.dstAccessMask       = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

  VkRenderPassCreateInfo renderPassCreateInfo = {};
  renderPassCreateInfo.sType                  = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
  renderPassCreateInfo.attachmentCount        = 1;
  renderPassCreateInfo.pAttachments           = &attachment;
  renderPassCreateInfo.subpassCount           = 1;
  renderPassCreateInfo.pSubpasses             = &subpass;
  renderPassCreateInfo.dependencyCount        = 1;
  renderPassCreateInfo.pDependencies          = &dependency;

  VkResult result =
      vkCreateRenderPass(_appContext->getDevice(), &renderPassCreateInfo, nullptr, &_guiPass);
  assert(result == VK_SUCCESS && "vkCreateRenderPass failed");
}

void ImGuiManager::_createFramebuffers() {
  // Create gui frame buffers for gui pass to use
  // Each frame buffer will have an attachment of VkImageView, in this case, the
  // attachments are mSwapchainImageViews
  _guiFrameBuffers.resize(_appContext->getSwapchainSize());

  uint32_t const w = _appContext->getSwapchainExtentWidth();
  uint32_t const h = _appContext->getSwapchainExtentHeight();

  // Iterate through image views
  for (size_t i = 0; i < _appContext->getSwapchainSize(); i++) {
    VkImageView attachment = _appContext->getSwapchainImageViews()[i];

    VkFramebufferCreateInfo frameBufferCreateInfo{};
    frameBufferCreateInfo.sType           = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
    frameBufferCreateInfo.renderPass      = _guiPass;
    frameBufferCreateInfo.attachmentCount = 1;
    frameBufferCreateInfo.pAttachments    = &attachment;
    frameBufferCreateInfo.width           = w;
    frameBufferCreateInfo.height          = h;
    frameBufferCreateInfo.layers          = 1;

    VkResult result = vkCreateFramebuffer(_appContext->getDevice(), &frameBufferCreateInfo, nullptr,
                                          &_guiFrameBuffers[i]);
    assert(result == VK_SUCCESS && "vkCreateFramebuffer failed");
  }
}

void ImGuiManager::recordGuiCommandBuffer(size_t currentFrame, uint32_t swapchainImageIndex) {
  VkCommandBuffer commandBuffer = _guiCommandBuffers[currentFrame];

  VkCommandBufferBeginInfo beginInfo{};
  beginInfo.sType            = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  beginInfo.flags            = 0;       // Optional
  beginInfo.pInheritanceInfo = nullptr; // Optional

  // A call to vkBeginCommandBuffer will implicitly reset the command buffer
  VkResult result = vkBeginCommandBuffer(commandBuffer, &beginInfo);
  assert(result == VK_SUCCESS && "vkBeginCommandBuffer failed");

  VkRenderPassBeginInfo renderPassInfo = {};
  renderPassInfo.sType                 = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
  renderPassInfo.renderPass            = _guiPass;
  renderPassInfo.framebuffer           = _guiFrameBuffers[swapchainImageIndex];
  renderPassInfo.renderArea.extent     = _appContext->getSwapchainExtent();

  VkClearValue clearValue{};
  clearValue.color = {{0.0F, 0.0F, 0.0F, 1.0F}};

  renderPassInfo.clearValueCount = 1;
  renderPassInfo.pClearValues    = &clearValue;

  vkCmdBeginRenderPass(commandBuffer, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

  // Record Imgui Draw Data and draw funcs into command buffer
  ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), commandBuffer);

  vkCmdEndRenderPass(commandBuffer);

  result = vkEndCommandBuffer(commandBuffer);
  assert(result == VK_SUCCESS && "vkEndCommandBuffer failed");
}

void ImGuiManager::_configMenu() {

  if (ImGui::BeginMenu("Config")) {
    // ImGui::SeparatorText("Gradient Projection");
    // ImGui::Checkbox("Use Gradient Projection", &_useGradientProjection);

    // ImGui::SeparatorText("Rtx");
    // ImGui::Checkbox("Moving Light Source", &_movingLightSource);
    // std::vector<std::string> const outputItems{"Combined", "Direct Only", "Indirect Only"};
    // comboSelector("Output Type", outputItems, _outputType);
    // float constexpr kDragSpeed = 0.01F;
    // ImGui::DragFloat("Offset X", &_offsetX, kDragSpeed, -1.0F, 1.0F);
    // ImGui::DragFloat("Offset Y", &_offsetY, kDragSpeed, -1.0F, 1.0F);

    // ImGui::SeparatorText("Stratum Filter");
    // ImGui::Checkbox("Use Stratum Filter", &_useStratumFiltering);

    // ImGui::SeparatorText("Temporal Blend");
    // ImGui::Checkbox("Temporal Accumulation", &_useTemporalBlend);
    // ImGui::Checkbox("Use normal test", &_useNormalTest);
    // ImGui::SliderFloat("Normal threhold", &_normalThreshold, 0.0F, 1.0F);
    // ImGui::SliderFloat("Blending Alpha", &_blendingAlpha, 0.0F, 1.0F);

    // ImGui::SeparatorText("Variance Estimation");
    // ImGui::Checkbox("Variance Calculation", &_useVarianceEstimation);
    // ImGui::Checkbox("Skip Stopping Functions", &_skipStoppingFunctions);
    // ImGui::Checkbox("Use Temporal Variance", &_useTemporalVariance);
    // int constexpr kMaxVarianceKernalSize = 15;
    // ImGui::SliderInt("Variance Kernel Size", &_varianceKernelSize, 1, kMaxVarianceKernalSize);
    // ImGui::SliderFloat("Variance Phi Gaussian", &_variancePhiGaussian, 0.0F, 1.0F);
    // ImGui::SliderFloat("Variance Phi Depth", &_variancePhiDepth, 0.0F, 1.0F);

    // ImGui::SeparatorText("A-Trous");
    // ImGui::Checkbox("A-Trous", &_useATrous);
    // ImGui::SliderInt("A-Trous times", &_iCap, 0, kATrousSize);
    // ImGui::Checkbox("Use variance guided filtering", &_useVarianceGuidedFiltering);
    // ImGui::Checkbox("Use gradient in depth", &_useGradientInDepth);
    // ImGui::SliderFloat("Luminance Phi", &_phiLuminance, 0.0F, 1.0F);
    // ImGui::SliderFloat("Phi Depth", &_phiDepth, 0.0F, 1.0F);
    // float constexpr kPhiNormalMax = 200.0F;
    // ImGui::SliderFloat("Phi Normal", &_phiNormal, 0.0F, kPhiNormalMax);
    // ImGui::Checkbox("Ignore Luminance For First Iteration", &_ignoreLuminanceAtFirstIteration);
    // ImGui::Checkbox("Changing luminance phi", &_changingLuminancePhi);
    // ImGui::Checkbox("Use jitter", &_useJittering);

    // ImGui::SeparatorText("Post Processing");
    // std::vector<std::string> displayItems{"Color",      "Variance", "RawCol", "Stratum",
    //                                       "Visibility", "Gradient", "Custom"};
    // comboSelector("Display Type", displayItems, _displayType);

    ImGui::Text("nothing is here");

    ImGui::EndMenu();
  }
}

void ImGuiManager::_fpsMenu(float fps) {
  std::string const kFpsString = std::to_string(static_cast<int>(fps));

  // calculate the right-aligned position for the FPS menu
  auto windowWidth      = ImGui::GetWindowContentRegionMax().x;
  auto fpsMenuWidth     = ImGui::CalcTextSize(kFpsString.c_str()).x;
  auto rightAlignedPosX = windowWidth - fpsMenuWidth;

  // set the cursor position to the calculated position
  ImGui::SetCursorPosX(rightAlignedPosX);
  ImGui::SetNextItemWidth(fpsMenuWidth);
  if (ImGui::BeginMenu("##FpsMenu")) {
    _fpsGui->update(fps);
    ImGui::EndMenu();
  }

  ImGui::SetCursorPosX(rightAlignedPosX);

  ImGui::Text("%s", kFpsString.c_str());
}

void ImGuiManager::_syncMousePosition() {
  auto &io = ImGui::GetIO();
  // the mousePos is not synced correctly when the window is not focused
  // so we set it manually here
  io.MousePos = ImVec2(static_cast<float>(_window->getCursorXPos()),
                       static_cast<float>(_window->getCursorYPos()));
}

void ImGuiManager::update(float fps) {
  _syncMousePosition();

  ImGui_ImplVulkan_NewFrame();
  // handles the user input, and the resizing of the window
  ImGui_ImplGlfw_NewFrame();

  ImGui::NewFrame();

  ImGui::BeginMainMenuBar();
  _configMenu();

  _fpsMenu(fps);
  ImGui::EndMainMenuBar();

  auto *mainGuiViewPort = ImGui::GetMainViewport();
  float height          = mainGuiViewPort->Size.y;

  const float kStatsWindowWidth  = 200.0F;
  const float kStatsWindowHeight = 80.0F;

  ImGui::SetNextWindowPos(ImVec2(0, height - kStatsWindowHeight));
  ImGui::Begin("Stats", nullptr,
               ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove |
                   ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse |
                   ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoSavedSettings |
                   ImGuiWindowFlags_NoInputs | ImGuiWindowFlags_NoFocusOnAppearing |
                   ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoNavFocus);

  ImGui::SetWindowSize(ImVec2(kStatsWindowWidth, kStatsWindowHeight));
  ImGui::Text("fps : %.2f", fps);
  float constexpr kMsPerSecond = 1000.0F;
  ImGui::Text("frame t: %.2f", kMsPerSecond / fps);

  ImGui::End();

  ImGui::Render();
}