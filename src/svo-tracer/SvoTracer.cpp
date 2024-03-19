#include "SvoTracer.hpp"

#include "SvoTracerDataGpu.hpp"
#include "app-context/VulkanApplicationContext.hpp"
#include "file-watcher/ShaderChangeListener.hpp"
#include "memory/Buffer.hpp"
#include "memory/BufferBundle.hpp"
#include "memory/Image.hpp"
#include "pipeline/ComputePipeline.hpp"
#include "pipeline/DescriptorSetBundle.hpp"
#include "svo-builder/SvoBuilder.hpp"
#include "utils/camera/Camera.hpp"
#include "utils/config/RootDir.h"
#include "utils/file-io/ShaderFileReader.hpp"
#include "utils/toml-config/TomlConfigReader.hpp"

#include <string>

namespace {
float halton(int base, int index) {
  float f = 1.F;
  float r = 0.F;
  int i   = index;

  while (i > 0) {
    f = f / static_cast<float>(base);
    r = r + f * static_cast<float>(i % base);
    i = i / base;
  }
  return r;
};

std::string _makeShaderFullPath(std::string const &shaderName) {
  return kPathToResourceFolder + "shaders/svo-tracer/" + shaderName;
}
}; // namespace

SvoTracer::SvoTracer(VulkanApplicationContext *appContext, Logger *logger, size_t framesInFlight,
                     Camera *camera, ShaderCompiler *shaderCompiler,
                     ShaderChangeListener *shaderChangeListener, TomlConfigReader *tomlConfigReader)
    : _appContext(appContext), _logger(logger), _camera(camera), _shaderCompiler(shaderCompiler),
      _shaderChangeListener(shaderChangeListener), _tomlConfigReader(tomlConfigReader),
      _uboData(tomlConfigReader), _framesInFlight(framesInFlight) {
  _loadConfig();
  _updateImageResolutions();
}

SvoTracer::~SvoTracer() {
  vkDestroySampler(_appContext->getDevice(), _defaultSampler, nullptr);

  for (auto &commandBuffer : _tracingCommandBuffers) {
    vkFreeCommandBuffers(_appContext->getDevice(), _appContext->getCommandPool(), 1,
                         &commandBuffer);
  }
}

void SvoTracer::_loadConfig() {
  _aTrousSizeMax  = _tomlConfigReader->getConfig<uint32_t>("SvoTracer.aTrousSizeMax");
  _beamResolution = _tomlConfigReader->getConfig<uint32_t>("SvoTracer.beamResolution");
  _taaSamplingOffsetSize =
      _tomlConfigReader->getConfig<uint32_t>("SvoTracer.taaSamplingOffsetSize");
  _taaUpscaleRatio     = _tomlConfigReader->getConfig<float>("SvoTracer.taaUpscaleRatio");
  _nearestUpscaleRatio = _tomlConfigReader->getConfig<float>("SvoTracer.nearestUpscaleRatio");
}

void SvoTracer::_updateImageResolutions() {
  float h2m = 1 / _nearestUpscaleRatio;
  float m2l = 1 / _taaUpscaleRatio;

  _highResWidth  = _appContext->getSwapchainExtentWidth();
  _highResHeight = _appContext->getSwapchainExtentHeight();

  _midResWidth  = static_cast<uint32_t>(static_cast<float>(_highResWidth) * h2m);
  _midResHeight = static_cast<uint32_t>(static_cast<float>(_highResHeight) * h2m);

  _lowResWidth  = static_cast<uint32_t>(static_cast<float>(_midResWidth) * m2l);
  _lowResHeight = static_cast<uint32_t>(static_cast<float>(_midResHeight) * m2l);

  _logger->info("High res: {}x{}", _highResWidth, _highResHeight);
  _logger->info("Mid res: {}x{}", _midResWidth, _midResHeight);
  _logger->info("Low res: {}x{}", _lowResWidth, _lowResHeight);
}

void SvoTracer::init(SvoBuilder *svoBuilder) {
  _svoBuilder = svoBuilder;

  _createDefaultSampler();

  // images
  _createImages();
  _createImageForwardingPairs();

  // buffers
  _createBuffersAndBufferBundles();
  _initBufferData();

  // pipelines
  _createDescriptorSetBundle();
  _createPipelines();

  // create command buffers
  _recordRenderingCommandBuffers();
  _recordDeliveryCommandBuffers();

  _createTaaSamplingOffsets();
}

void SvoTracer::onSwapchainResize() {
  _updateImageResolutions();

  // images
  _createSwapchainRelatedImages();
  _createImageForwardingPairs();

  // pipelines
  _createDescriptorSetBundle();
  _updatePipelinesDescriptorBundles();

  _recordRenderingCommandBuffers();
  _recordDeliveryCommandBuffers();
}

void SvoTracer::_createTaaSamplingOffsets() {
  _subpixOffsets.resize(_taaSamplingOffsetSize);
  for (int i = 0; i < _taaSamplingOffsetSize; i++) {
    // _subpixOffsets[i] = {halton(2, i + 1) - 0.5F, halton(3, i + 1) - 0.5F};
    _subpixOffsets[i] = {0, 0};
  }
}

void SvoTracer::update() { _recordRenderingCommandBuffers(); }

void SvoTracer::_createDefaultSampler() {
  VkSamplerCreateInfo samplerInfo{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
  samplerInfo.magFilter               = VK_FILTER_LINEAR; // For bilinear interpolation
  samplerInfo.minFilter               = VK_FILTER_LINEAR; // For bilinear interpolation
  samplerInfo.addressModeU            = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
  samplerInfo.addressModeV            = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
  samplerInfo.addressModeW            = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
  samplerInfo.anisotropyEnable        = VK_FALSE;
  samplerInfo.maxAnisotropy           = 1;
  samplerInfo.borderColor             = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
  samplerInfo.unnormalizedCoordinates = VK_FALSE;
  samplerInfo.compareEnable           = VK_FALSE;
  samplerInfo.compareOp               = VK_COMPARE_OP_ALWAYS;
  samplerInfo.mipmapMode              = VK_SAMPLER_MIPMAP_MODE_LINEAR;
  samplerInfo.mipLodBias              = 0.0F;
  samplerInfo.minLod                  = 0.0F;
  samplerInfo.maxLod                  = 0.0F;

  if (vkCreateSampler(_appContext->getDevice(), &samplerInfo, nullptr, &_defaultSampler) !=
      VK_SUCCESS) {
    throw std::runtime_error("failed to create texture sampler!");
  }
}

void SvoTracer::_createImages() {
  _createResourseImages();
  _createSwapchainRelatedImages();
}

void SvoTracer::_createResourseImages() { _createBlueNoiseImages(); }

void SvoTracer::_createSwapchainRelatedImages() {
  _createFullSizedImages();
  _createStratumSizedImages();
}

void SvoTracer::_createBlueNoiseImages() {
  constexpr int kBlueNoiseArraySize = 64;

  std::vector<std::string> filenames{};
  filenames.reserve(kBlueNoiseArraySize);
  for (int i = 0; i < kBlueNoiseArraySize; i++) {
    filenames.emplace_back(kPathToResourceFolder +
                           "/textures/stbn/vec2_2d_1d/"
                           "stbn_vec2_2Dx1D_128x128x64_" +
                           std::to_string(i) + ".png");
  }
  _vec2BlueNoise = std::make_unique<Image>(filenames, VK_IMAGE_USAGE_STORAGE_BIT |
                                                          VK_IMAGE_USAGE_TRANSFER_DST_BIT);

  filenames.clear();
  filenames.reserve(kBlueNoiseArraySize);
  for (int i = 0; i < kBlueNoiseArraySize; i++) {
    filenames.emplace_back(kPathToResourceFolder +
                           "/textures/stbn/unitvec3_cosine_2d_1d/"
                           "stbn_unitvec3_cosine_2Dx1D_128x128x64_" +
                           std::to_string(i) + ".png");
  }
  _weightedCosineBlueNoise = std::make_unique<Image>(
      filenames, VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
}

void SvoTracer::_createFullSizedImages() {
  _backgroundImage = std::make_unique<Image>(_lowResWidth, _lowResHeight, 1, VK_FORMAT_R32_UINT,
                                             VK_IMAGE_USAGE_STORAGE_BIT);

  // w = 16 -> 3, w = 17 -> 4
  _beamDepthImage = std::make_unique<Image>(
      std::ceil(static_cast<float>(_lowResWidth) / static_cast<float>(_beamResolution)) + 1,
      std::ceil(static_cast<float>(_lowResHeight) / static_cast<float>(_beamResolution)) + 1, 1,
      VK_FORMAT_R32_SFLOAT, VK_IMAGE_USAGE_STORAGE_BIT);

  _rawImage = std::make_unique<Image>(_lowResWidth, _lowResHeight, 1, VK_FORMAT_R32_UINT,
                                      VK_IMAGE_USAGE_STORAGE_BIT);

  _depthImage = std::make_unique<Image>(_lowResWidth, _lowResHeight, 1, VK_FORMAT_R32_SFLOAT,
                                        VK_IMAGE_USAGE_STORAGE_BIT);

  _octreeVisualizationImage =
      std::make_unique<Image>(_lowResWidth, _lowResHeight, 1, VK_FORMAT_R32_UINT,
                              VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);

  _hitImage = std::make_unique<Image>(_lowResWidth, _lowResHeight, 1, VK_FORMAT_R8_UINT,
                                      VK_IMAGE_USAGE_STORAGE_BIT);

  _temporalHistLengthImage = std::make_unique<Image>(_lowResWidth, _lowResHeight, 1,
                                                     VK_FORMAT_R8_UINT, VK_IMAGE_USAGE_STORAGE_BIT);

  _motionImage = std::make_unique<Image>(_lowResWidth, _lowResHeight, 1,
                                         VK_FORMAT_R16G16B16A16_SFLOAT, VK_IMAGE_USAGE_STORAGE_BIT);
  _normalImage =
      std::make_unique<Image>(_lowResWidth, _lowResHeight, 1, VK_FORMAT_R32_UINT,
                              VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
  _lastNormalImage =
      std::make_unique<Image>(_lowResWidth, _lowResHeight, 1, VK_FORMAT_R32_UINT,
                              VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);

  _positionImage =
      std::make_unique<Image>(_lowResWidth, _lowResHeight, 1, VK_FORMAT_R32G32B32A32_SFLOAT,
                              VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT);

  _lastPositionImage =
      std::make_unique<Image>(_lowResWidth, _lowResHeight, 1, VK_FORMAT_R32G32B32A32_SFLOAT,
                              VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);

  _voxHashImage =
      std::make_unique<Image>(_lowResWidth, _lowResHeight, 1, VK_FORMAT_R32_UINT,
                              VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
  _lastVoxHashImage =
      std::make_unique<Image>(_lowResWidth, _lowResHeight, 1, VK_FORMAT_R32_UINT,
                              VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);

  _accumedImage =
      std::make_unique<Image>(_lowResWidth, _lowResHeight, 1, VK_FORMAT_R32_UINT,
                              VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
  _lastAccumedImage =
      std::make_unique<Image>(_lowResWidth, _lowResHeight, 1, VK_FORMAT_R32_UINT,
                              VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);

  _taaImage = std::make_unique<Image>(_midResWidth, _midResHeight, 1, VK_FORMAT_R16G16B16A16_SFLOAT,
                                      VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT);

  _lastTaaImage = std::make_unique<Image>(
      _midResWidth, _midResHeight, 1, VK_FORMAT_R16G16B16A16_SFLOAT,
      VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
      _defaultSampler);

  _blittedImage = std::make_unique<Image>(_lowResWidth, _lowResHeight, 1, VK_FORMAT_R32_UINT,
                                          VK_IMAGE_USAGE_STORAGE_BIT);

  // _varianceHistImage =
  //     std::make_unique<Image>(lw, lh, 1, VK_FORMAT_R32G32B32A32_SFLOAT,

  //                             VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
  // _lastVarianceHistImage =
  //     std::make_unique<Image>(lw, lh, 1, VK_FORMAT_R32G32B32A32_SFLOAT,
  // VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);

  // both of the ping and pong can be dumped to the render target image and the lastAccumedImage
  _aTrousPingImage = std::make_unique<Image>(_lowResWidth, _lowResHeight, 1, VK_FORMAT_R32_UINT,
                                             VK_IMAGE_USAGE_STORAGE_BIT);

  _aTrousPongImage = std::make_unique<Image>(_lowResWidth, _lowResHeight, 1, VK_FORMAT_R32_UINT,
                                             VK_IMAGE_USAGE_STORAGE_BIT);

  _aTrousFinalResultImage = std::make_unique<Image>(_lowResWidth, _lowResHeight, 1,
                                                    VK_FORMAT_R32_UINT, VK_IMAGE_USAGE_STORAGE_BIT);

  _renderTargetImage = std::make_unique<Image>(
      _highResWidth, _highResHeight, 1, VK_FORMAT_R8G8B8A8_UNORM,
      VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
          VK_IMAGE_USAGE_TRANSFER_SRC_BIT);

  // _seedImage =
  //     std::make_unique<Image>(w, h, VK_FORMAT_R32G32B32A32_UINT, VK_IMAGE_USAGE_STORAGE_BIT);

  // _aTrousInputImage =
  //     std::make_unique<Image>(w, h, VK_FORMAT_R32G32B32A32_SFLOAT,
  //                             VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);

  // _aTrousOutputImage =
  //     std::make_unique<Image>(w, h, VK_FORMAT_R32G32B32A32_SFLOAT,
  //                             VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
  //                                 VK_IMAGE_USAGE_TRANSFER_DST_BIT);

  // _depthImagePrev = std::make_unique<Image>(
  //     w, h, VK_FORMAT_R32_SFLOAT, VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);

  // _gradientImage = std::make_unique<Image>(
  //     w, h, VK_FORMAT_R32G32_SFLOAT, VK_IMAGE_USAGE_STORAGE_BIT |
  //     VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
  // _gradientImagePrev = std::make_unique<Image>(
  //     w, h, VK_FORMAT_R32G32_SFLOAT, VK_IMAGE_USAGE_STORAGE_BIT |
  //     VK_IMAGE_USAGE_TRANSFER_DST_BIT);

  // _varianceImage = std::make_unique<Image>(w, h, VK_FORMAT_R32_SFLOAT,
  // VK_IMAGE_USAGE_STORAGE_BIT);
}

void SvoTracer::_createStratumSizedImages() {
  // auto w = _appContext->getSwapchainExtentWidth();
  // auto h = _appContext->getSwapchainExtentHeight();

  // constexpr float kStratumResolutionScale = 1.0F / 3.0F;

  // uint32_t perStratumImageWidth  = ceil(static_cast<float>(w) * kStratumResolutionScale);
  // uint32_t perStratumImageHeight = ceil(static_cast<float>(h) * kStratumResolutionScale);

  // _stratumOffsetImage =
  //     std::make_unique<Image>(perStratumImageWidth, perStratumImageHeight, VK_FORMAT_R32_UINT,
  //                             VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);

  // _perStratumLockingImage =
  //     std::make_unique<Image>(perStratumImageWidth, perStratumImageHeight, VK_FORMAT_R32_UINT,

  //                             VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);

  // _visibilityImage = std::make_unique<Image>(
  //     perStratumImageWidth, perStratumImageHeight, VK_FORMAT_R32G32B32A32_SFLOAT,
  //     VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);

  // _seedVisibilityImage = std::make_unique<Image>(
  //     perStratumImageWidth, perStratumImageHeight, VK_FORMAT_R32G32B32A32_UINT,
  //     VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);

  // _temporalGradientNormalizationImagePing = std::make_unique<Image>(
  //     perStratumImageWidth, perStratumImageHeight, VK_FORMAT_R32G32B32A32_SFLOAT,
  //     VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);

  // _temporalGradientNormalizationImagePong = std::make_unique<Image>(
  //     perStratumImageWidth, perStratumImageHeight, VK_FORMAT_R32G32B32A32_SFLOAT,
  //     VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
}

void SvoTracer::_createImageForwardingPairs() {
  _normalForwardingPair = std::make_unique<ImageForwardingPair>(
      _normalImage->getVkImage(), _lastNormalImage->getVkImage(), _lowResWidth, _lowResHeight,
      VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
      VK_IMAGE_LAYOUT_GENERAL);

  _positionForwardingPair = std::make_unique<ImageForwardingPair>(
      _positionImage->getVkImage(), _lastPositionImage->getVkImage(), _lowResWidth, _lowResHeight,
      VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
      VK_IMAGE_LAYOUT_GENERAL);

  _voxHashForwardingPair = std::make_unique<ImageForwardingPair>(
      _voxHashImage->getVkImage(), _lastVoxHashImage->getVkImage(), _lowResWidth, _lowResHeight,
      VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
      VK_IMAGE_LAYOUT_GENERAL);

  _accumedForwardingPair = std::make_unique<ImageForwardingPair>(
      _accumedImage->getVkImage(), _lastAccumedImage->getVkImage(), _lowResWidth, _lowResHeight,
      VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
      VK_IMAGE_LAYOUT_GENERAL);

  _taaForwardingPair = std::make_unique<ImageForwardingPair>(
      _taaImage->getVkImage(), _lastTaaImage->getVkImage(), _midResWidth, _midResHeight,
      VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
      VK_IMAGE_LAYOUT_GENERAL);

  // creating forwarding pairs to copy the image result each frame to a specific swapchain
  _targetForwardingPairs.clear();
  for (int i = 0; i < _appContext->getSwapchainImagesCount(); i++) {
    _targetForwardingPairs.emplace_back(std::make_unique<ImageForwardingPair>(
        _renderTargetImage->getVkImage(), _appContext->getSwapchainImages()[i], _highResWidth,
        _highResHeight, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
        VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL));
  }
}

// these buffers are modified by the CPU side every frame, and we have multiple frames in flight,
// so we need to create multiple copies of them, they are fairly small though
void SvoTracer::_createBuffersAndBufferBundles() {
  // buffers
  _sceneInfoBuffer = std::make_unique<Buffer>(
      sizeof(G_SceneInfo), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, MemoryStyle::kDedicated);

  _aTrousIterationBuffer = std::make_unique<Buffer>(
      sizeof(uint32_t), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
      MemoryStyle::kDedicated);

  _aTrousIterationStagingBuffers.clear();
  _aTrousIterationStagingBuffers.reserve(_aTrousSizeMax);
  for (int i = 0; i < _aTrousSizeMax; i++) {
    _aTrousIterationStagingBuffers.emplace_back(std::make_unique<Buffer>(
        sizeof(uint32_t), VK_BUFFER_USAGE_TRANSFER_SRC_BIT, MemoryStyle::kDedicated));
  }

  // buffer bundles
  _renderInfoBufferBundle =
      std::make_unique<BufferBundle>(_framesInFlight, sizeof(G_RenderInfo),
                                     VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, MemoryStyle::kHostVisible);

  _environmentInfoBufferBundle =
      std::make_unique<BufferBundle>(_framesInFlight, sizeof(G_EnvironmentInfo),
                                     VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, MemoryStyle::kHostVisible);

  _twickableParametersBufferBundle =
      std::make_unique<BufferBundle>(_framesInFlight, sizeof(G_TwickableParameters),
                                     VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, MemoryStyle::kHostVisible);

  _temporalFilterInfoBufferBundle =
      std::make_unique<BufferBundle>(_framesInFlight, sizeof(G_TemporalFilterInfo),
                                     VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, MemoryStyle::kHostVisible);

  _spatialFilterInfoBufferBundle =
      std::make_unique<BufferBundle>(_framesInFlight, sizeof(G_SpatialFilterInfo),
                                     VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, MemoryStyle::kHostVisible);
}

void SvoTracer::_initBufferData() {
  G_SceneInfo sceneData = {_beamResolution, _svoBuilder->getVoxelLevelCount(),
                           _svoBuilder->getChunksDim()};
  _sceneInfoBuffer->fillData(&sceneData);

  for (uint32_t i = 0; i < _aTrousSizeMax; i++) {
    uint32_t aTrousIteration = i;
    _aTrousIterationStagingBuffers[i]->fillData(&aTrousIteration);
  }
}

void SvoTracer::_recordRenderingCommandBuffers() {
  for (auto &commandBuffer : _tracingCommandBuffers) {
    vkFreeCommandBuffers(_appContext->getDevice(), _appContext->getCommandPool(), 1,
                         &commandBuffer);
  }
  _tracingCommandBuffers.clear();

  _tracingCommandBuffers.resize(_framesInFlight); //  change this later on, because it is
                                                  //  bounded to the swapchain image
  VkCommandBufferAllocateInfo allocInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
  allocInfo.commandPool        = _appContext->getCommandPool();
  allocInfo.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  allocInfo.commandBufferCount = (uint32_t)_tracingCommandBuffers.size();

  vkAllocateCommandBuffers(_appContext->getDevice(), &allocInfo, _tracingCommandBuffers.data());

  VkMemoryBarrier uboWritingBarrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
  uboWritingBarrier.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
  uboWritingBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

  // create the general memory barrier
  VkMemoryBarrier memoryBarrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
  memoryBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
  memoryBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;

  for (uint32_t frameIndex = 0; frameIndex < _tracingCommandBuffers.size(); frameIndex++) {
    auto &cmdBuffer = _tracingCommandBuffers[frameIndex];

    VkCommandBufferBeginInfo beginInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    vkBeginCommandBuffer(cmdBuffer, &beginInfo);

    // make all host writes to the ubo visible to the shaders
    vkCmdPipelineBarrier(cmdBuffer,
                         VK_PIPELINE_STAGE_HOST_BIT,           // source stage
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, // destination stage
                         0,                                    // dependency flags
                         1,                                    // memory barrier count
                         &uboWritingBarrier,                   // memory barriers
                         0,                                    // buffer memory barrier count
                         nullptr,                              // buffer memory barriers
                         0,                                    // image memory barrier count
                         nullptr                               // image memory barriers
    );

    // _renderTargetImage->clearImage(cmdBuffer);

    _svoCourseBeamPipeline->recordCommand(
        cmdBuffer, frameIndex,
        static_cast<uint32_t>(
            std::ceil(static_cast<float>(_lowResWidth) / static_cast<float>(_beamResolution))) +
            1,
        static_cast<uint32_t>(
            std::ceil(static_cast<float>(_lowResHeight) / static_cast<float>(_beamResolution))) +
            1,
        1);

    vkCmdPipelineBarrier(cmdBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &memoryBarrier, 0, nullptr, 0,
                         nullptr);

    _svoTracingPipeline->recordCommand(cmdBuffer, frameIndex, _lowResWidth, _lowResHeight, 1);

    vkCmdPipelineBarrier(cmdBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &memoryBarrier, 0, nullptr, 0,
                         nullptr);

    _temporalFilterPipeline->recordCommand(cmdBuffer, frameIndex, _lowResWidth, _lowResHeight, 1);

    vkCmdPipelineBarrier(cmdBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &memoryBarrier, 0, nullptr, 0,
                         nullptr);

    for (int i = 0; i < _aTrousSizeMax; i++) {
      VkBufferCopy bufCopy = {
          0,                                 // srcOffset
          0,                                 // dstOffset,
          _aTrousIterationBuffer->getSize(), // size
      };

      vkCmdCopyBuffer(cmdBuffer, _aTrousIterationStagingBuffers[i]->getVkBuffer(),
                      _aTrousIterationBuffer->getVkBuffer(), 1, &bufCopy);

      VkMemoryBarrier bufferCopyMemoryBarrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
      bufferCopyMemoryBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
      bufferCopyMemoryBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

      vkCmdPipelineBarrier(cmdBuffer,
                           VK_PIPELINE_STAGE_TRANSFER_BIT,       // source stage
                           VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, // destination stage
                           0,                                    // dependency flags
                           1,                                    // memory barrier count
                           &bufferCopyMemoryBarrier,             // memory barriers
                           0,                                    // buffer memory barrier count
                           nullptr,                              // buffer memory barriers
                           0,                                    // image memory barrier count
                           nullptr                               // image memory barriers
      );

      _aTrousPipeline->recordCommand(cmdBuffer, frameIndex, _lowResWidth, _lowResHeight, 1);

      vkCmdPipelineBarrier(cmdBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                           VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &memoryBarrier, 0, nullptr,
                           0, nullptr);
    }

    _backgroundBlitPipeline->recordCommand(cmdBuffer, frameIndex, _lowResWidth, _lowResHeight, 1);

    vkCmdPipelineBarrier(cmdBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &memoryBarrier, 0, nullptr, 0,
                         nullptr);

    _taaUpscalingPipeline->recordCommand(cmdBuffer, frameIndex, _midResWidth, _midResHeight, 1);

    vkCmdPipelineBarrier(cmdBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &memoryBarrier, 0, nullptr, 0,
                         nullptr);

    _nearestUpscalingPipeline->recordCommand(cmdBuffer, frameIndex, _highResWidth, _highResHeight,
                                             1);

    // copy to history images
    _normalForwardingPair->forwardCopy(cmdBuffer);
    _positionForwardingPair->forwardCopy(cmdBuffer);
    _voxHashForwardingPair->forwardCopy(cmdBuffer);
    _accumedForwardingPair->forwardCopy(cmdBuffer);
    _taaForwardingPair->forwardCopy(cmdBuffer);

    vkEndCommandBuffer(cmdBuffer);
  }
}

void SvoTracer::_recordDeliveryCommandBuffers() {
  for (auto &commandBuffer : _deliveryCommandBuffers) {
    vkFreeCommandBuffers(_appContext->getDevice(), _appContext->getCommandPool(), 1,
                         &commandBuffer);
  }
  _deliveryCommandBuffers.clear();

  _deliveryCommandBuffers.resize(_appContext->getSwapchainImagesCount());

  VkCommandBufferAllocateInfo allocInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
  allocInfo.commandPool        = _appContext->getCommandPool();
  allocInfo.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  allocInfo.commandBufferCount = static_cast<uint32_t>(_deliveryCommandBuffers.size());

  vkAllocateCommandBuffers(_appContext->getDevice(), &allocInfo, _deliveryCommandBuffers.data());

  VkMemoryBarrier deliveryMemoryBarrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
  deliveryMemoryBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
  deliveryMemoryBarrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;

  for (size_t imageIndex = 0; imageIndex < _deliveryCommandBuffers.size(); imageIndex++) {
    auto &cmdBuffer = _deliveryCommandBuffers[imageIndex];

    VkCommandBufferBeginInfo beginInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    vkBeginCommandBuffer(cmdBuffer, &beginInfo);

    // // make all host writes to the ubo visible to the shaders
    // vkCmdPipelineBarrier(cmdBuffer,
    //                      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, // source stage
    //                      VK_PIPELINE_STAGE_TRANSFER_BIT,       // destination stage
    //                      0,                                    // dependency flags
    //                      1,                                    // memory barrier count
    //                      &deliveryMemoryBarrier,               // memory barriers
    //                      0,                                    // buffer memory barrier count
    //                      nullptr,                              // buffer memory barriers
    //                      0,                                    // image memory barrier count
    //                      nullptr                               // image memory barriers
    // );

    _targetForwardingPairs[imageIndex]->forwardCopy(cmdBuffer);
    vkEndCommandBuffer(cmdBuffer);
  }
}

void SvoTracer::updateUboData(size_t currentFrame) {
  static uint32_t currentSample = 0;
  // identity matrix
  static glm::mat4 vMatPrev{1.0F};
  static glm::mat4 vMatPrevInv{1.0F};
  static glm::mat4 pMatPrev{1.0F};
  static glm::mat4 pMatPrevInv{1.0F};
  static glm::mat4 vpMatPrev{1.0F};
  static glm::mat4 vpMatPrevInv{1.0F};

  auto currentTime = static_cast<float>(glfwGetTime());

  auto vMat    = _camera->getViewMatrix();
  auto vMatInv = glm::inverse(vMat);
  auto pMat =
      _camera->getProjectionMatrix(static_cast<float>(_appContext->getSwapchainExtentWidth()) /
                                   static_cast<float>(_appContext->getSwapchainExtentHeight()));
  auto pMatInv  = glm::inverse(pMat);
  auto vpMat    = pMat * vMat;
  auto vpMatInv = glm::inverse(vpMat);

  G_RenderInfo renderInfo = {
      _camera->getPosition(),
      _camera->getFront(),
      _camera->getUp(),
      _camera->getRight(),
      _subpixOffsets[currentSample % _taaSamplingOffsetSize],
      vMat,
      vMatInv,
      vMatPrev,
      vMatPrevInv,
      pMat,
      pMatInv,
      pMatPrev,
      pMatPrevInv,
      vpMat,
      vpMatInv,
      vpMatPrev,
      vpMatPrevInv,
      glm::uvec2(_lowResWidth, _lowResHeight),
      glm::vec2(1.F / static_cast<float>(_lowResWidth), 1.F / static_cast<float>(_lowResHeight)),
      glm::uvec2(_midResWidth, _midResHeight),
      glm::vec2(1.F / static_cast<float>(_midResWidth), 1.F / static_cast<float>(_midResHeight)),
      glm::uvec2(_highResWidth, _highResHeight),
      glm::vec2(1.F / static_cast<float>(_highResWidth), 1.F / static_cast<float>(_highResHeight)),
      _camera->getVFov(),
      currentSample,
      currentTime,
  };
  _renderInfoBufferBundle->getBuffer(currentFrame)->fillData(&renderInfo);

  vMatPrev     = vMat;
  vMatPrevInv  = vMatInv;
  pMatPrev     = pMat;
  pMatPrevInv  = pMatInv;
  vpMatPrev    = vpMat;
  vpMatPrevInv = vpMatInv;

  G_EnvironmentInfo environmentInfo{};
  environmentInfo.sunAngleA    = _uboData.sunAngleA;
  environmentInfo.sunAngleB    = _uboData.sunAngleB;
  environmentInfo.sunColor     = _uboData.sunColor;
  environmentInfo.sunLuminance = _uboData.sunLuminance;
  environmentInfo.sunSize      = _uboData.sunSize;
  _environmentInfoBufferBundle->getBuffer(currentFrame)->fillData(&environmentInfo);

  G_TwickableParameters twickableParameters{};
  twickableParameters.visualizeOctree   = _uboData.visualizeOctree;
  twickableParameters.beamOptimization  = _uboData.beamOptimization;
  twickableParameters.traceSecondaryRay = _uboData.traceSecondaryRay;
  twickableParameters.taa               = _uboData.taa;
  _twickableParametersBufferBundle->getBuffer(currentFrame)->fillData(&twickableParameters);

  G_TemporalFilterInfo temporalFilterInfo{};
  temporalFilterInfo.temporalAlpha       = _uboData.temporalAlpha;
  temporalFilterInfo.temporalPositionPhi = _uboData.temporalPositionPhi;
  _temporalFilterInfoBufferBundle->getBuffer(currentFrame)->fillData(&temporalFilterInfo);

  G_SpatialFilterInfo spatialFilterInfo{};
  spatialFilterInfo.aTrousIterationCount = static_cast<uint32_t>(_uboData.aTrousIterationCount);
  spatialFilterInfo.useVarianceGuidedFiltering = _uboData.useVarianceGuidedFiltering;
  spatialFilterInfo.phiC                       = _uboData.phiC;
  spatialFilterInfo.phiN                       = _uboData.phiN;
  spatialFilterInfo.phiP                       = _uboData.phiP;
  spatialFilterInfo.phiZ                       = _uboData.phiZ;
  spatialFilterInfo.phiZTolerance              = _uboData.phiZTolerance;
  spatialFilterInfo.changingLuminancePhi       = _uboData.changingLuminancePhi;
  _spatialFilterInfoBufferBundle->getBuffer(currentFrame)->fillData(&spatialFilterInfo);

  currentSample++;
}

void SvoTracer::_createDescriptorSetBundle() {
  _descriptorSetBundle = std::make_unique<DescriptorSetBundle>(_appContext, _framesInFlight,
                                                               VK_SHADER_STAGE_COMPUTE_BIT);

  _descriptorSetBundle->bindUniformBufferBundle(0, _renderInfoBufferBundle.get());
  _descriptorSetBundle->bindUniformBufferBundle(31, _environmentInfoBufferBundle.get());
  _descriptorSetBundle->bindUniformBufferBundle(1, _twickableParametersBufferBundle.get());
  _descriptorSetBundle->bindUniformBufferBundle(27, _temporalFilterInfoBufferBundle.get());
  _descriptorSetBundle->bindUniformBufferBundle(23, _spatialFilterInfoBufferBundle.get());

  _descriptorSetBundle->bindStorageImage(2, _vec2BlueNoise.get());
  _descriptorSetBundle->bindStorageImage(3, _weightedCosineBlueNoise.get());

  _descriptorSetBundle->bindStorageImage(37, _svoBuilder->getChunksImage());
  _descriptorSetBundle->bindStorageImage(29, _backgroundImage.get());
  _descriptorSetBundle->bindStorageImage(4, _beamDepthImage.get());
  _descriptorSetBundle->bindStorageImage(5, _rawImage.get());
  _descriptorSetBundle->bindStorageImage(6, _depthImage.get());
  _descriptorSetBundle->bindStorageImage(8, _octreeVisualizationImage.get());
  _descriptorSetBundle->bindStorageImage(28, _hitImage.get());
  _descriptorSetBundle->bindStorageImage(30, _temporalHistLengthImage.get());
  _descriptorSetBundle->bindStorageImage(32, _motionImage.get());
  _descriptorSetBundle->bindStorageImage(9, _normalImage.get());
  _descriptorSetBundle->bindStorageImage(10, _lastNormalImage.get());
  _descriptorSetBundle->bindStorageImage(7, _positionImage.get());
  _descriptorSetBundle->bindStorageImage(26, _lastPositionImage.get());
  _descriptorSetBundle->bindStorageImage(11, _voxHashImage.get());
  _descriptorSetBundle->bindStorageImage(12, _lastVoxHashImage.get());
  _descriptorSetBundle->bindStorageImage(13, _accumedImage.get());
  _descriptorSetBundle->bindStorageImage(14, _lastAccumedImage.get());

  _descriptorSetBundle->bindStorageImage(33, _taaImage.get());
  _descriptorSetBundle->bindStorageImage(34, _lastTaaImage.get());

  _descriptorSetBundle->bindStorageImage(35, _blittedImage.get());

  _descriptorSetBundle->bindImageSampler(36, _lastTaaImage.get());

  // _descriptorSetBundle->bindStorageImage(15, _varianceHistImage.get());
  // _descriptorSetBundle->bindStorageImage(16, _lastVarianceHistImage.get());

  // atrous ping and pong
  _descriptorSetBundle->bindStorageImage(17, _aTrousPingImage.get());
  _descriptorSetBundle->bindStorageImage(18, _aTrousPongImage.get());
  _descriptorSetBundle->bindStorageImage(25, _aTrousFinalResultImage.get());

  _descriptorSetBundle->bindStorageImage(19, _renderTargetImage.get());

  _descriptorSetBundle->bindStorageBuffer(20, _sceneInfoBuffer.get());
  _descriptorSetBundle->bindStorageBuffer(21, _svoBuilder->getAppendedOctreeBuffer());
  // _descriptorSetBundle->bindStorageBuffer(22, _svoBuilder->getPaletteBuffer());
  _descriptorSetBundle->bindStorageBuffer(24, _aTrousIterationBuffer.get());

  _descriptorSetBundle->create();
}

void SvoTracer::_createPipelines() {
  _svoCourseBeamPipeline = std::make_unique<ComputePipeline>(
      _appContext, _logger, this, _makeShaderFullPath("svoCoarseBeam.comp"), WorkGroupSize{8, 8, 1},
      _descriptorSetBundle.get(), _shaderCompiler, _shaderChangeListener);
  _svoCourseBeamPipeline->compileAndCacheShaderModule(false);
  _svoCourseBeamPipeline->build();

  _svoTracingPipeline = std::make_unique<ComputePipeline>(
      _appContext, _logger, this, _makeShaderFullPath("svoTracing.comp"), WorkGroupSize{8, 8, 1},
      _descriptorSetBundle.get(), _shaderCompiler, _shaderChangeListener);
  _svoTracingPipeline->compileAndCacheShaderModule(false);
  _svoTracingPipeline->build();

  _temporalFilterPipeline = std::make_unique<ComputePipeline>(
      _appContext, _logger, this, _makeShaderFullPath("temporalFilter.comp"),
      WorkGroupSize{8, 8, 1}, _descriptorSetBundle.get(), _shaderCompiler, _shaderChangeListener);
  _temporalFilterPipeline->compileAndCacheShaderModule(false);
  _temporalFilterPipeline->build();

  _aTrousPipeline = std::make_unique<ComputePipeline>(
      _appContext, _logger, this, _makeShaderFullPath("aTrous.comp"), WorkGroupSize{8, 8, 1},
      _descriptorSetBundle.get(), _shaderCompiler, _shaderChangeListener);
  _aTrousPipeline->compileAndCacheShaderModule(false);
  _aTrousPipeline->build();

  _backgroundBlitPipeline = std::make_unique<ComputePipeline>(
      _appContext, _logger, this, _makeShaderFullPath("backgroundBlit.comp"),
      WorkGroupSize{8, 8, 1}, _descriptorSetBundle.get(), _shaderCompiler, _shaderChangeListener);
  _backgroundBlitPipeline->compileAndCacheShaderModule(false);
  _backgroundBlitPipeline->build();

  _taaUpscalingPipeline = std::make_unique<ComputePipeline>(
      _appContext, _logger, this, _makeShaderFullPath("taaUpscaling.comp"), WorkGroupSize{8, 8, 1},
      _descriptorSetBundle.get(), _shaderCompiler, _shaderChangeListener);
  _taaUpscalingPipeline->compileAndCacheShaderModule(false);
  _taaUpscalingPipeline->build();

  _nearestUpscalingPipeline = std::make_unique<ComputePipeline>(
      _appContext, _logger, this, _makeShaderFullPath("nearestUpscaling.comp"),
      WorkGroupSize{8, 8, 1}, _descriptorSetBundle.get(), _shaderCompiler, _shaderChangeListener);
  _nearestUpscalingPipeline->compileAndCacheShaderModule(false);
  _nearestUpscalingPipeline->build();
}

void SvoTracer::_updatePipelinesDescriptorBundles() {
  _svoCourseBeamPipeline->updateDescriptorSetBundle(_descriptorSetBundle.get());
  _svoTracingPipeline->updateDescriptorSetBundle(_descriptorSetBundle.get());
  _temporalFilterPipeline->updateDescriptorSetBundle(_descriptorSetBundle.get());
  _aTrousPipeline->updateDescriptorSetBundle(_descriptorSetBundle.get());
  _backgroundBlitPipeline->updateDescriptorSetBundle(_descriptorSetBundle.get());
  _taaUpscalingPipeline->updateDescriptorSetBundle(_descriptorSetBundle.get());
  _nearestUpscalingPipeline->updateDescriptorSetBundle(_descriptorSetBundle.get());
}