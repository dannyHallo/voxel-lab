#include "ComputePipeline.hpp"
#include "app-context/VulkanApplicationContext.hpp"
#include "pipeline/DescriptorSetBundle.hpp"
#include "utils/logger/Logger.hpp"

#include "utils/config/RootDir.h"
#include "utils/file-io/ShaderFileReader.hpp"
#include "utils/shader-compiler/ShaderCompiler.hpp"

#include <cassert>
#include <memory>
#include <vector>

ComputePipeline::ComputePipeline(VulkanApplicationContext *appContext, Logger *logger,
                                 Scheduler *scheduler, std::string shaderFileName,
                                 WorkGroupSize workGroupSize,
                                 DescriptorSetBundle *descriptorSetBundle,
                                 ShaderChangeListener *shaderChangeListener, bool needToRebuildSvo)
    : Pipeline(appContext, logger, scheduler, std::move(shaderFileName), descriptorSetBundle,
               VK_SHADER_STAGE_COMPUTE_BIT, shaderChangeListener, needToRebuildSvo),
      _workGroupSize(workGroupSize), _shaderCompiler(std::make_unique<ShaderCompiler>(logger)) {}

ComputePipeline::~ComputePipeline() = default;

void ComputePipeline::build() {
  _cleanupPipelineAndLayout();

  VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
  pipelineLayoutInfo.sType          = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  pipelineLayoutInfo.setLayoutCount = 1;
  // this is why the compute pipeline requires the descriptor set layout to be specified
  pipelineLayoutInfo.pSetLayouts = &_descriptorSetBundle->getDescriptorSetLayout();

  vkCreatePipelineLayout(_appContext->getDevice(), &pipelineLayoutInfo, nullptr, &_pipelineLayout);

  if (_cachedShaderModule == VK_NULL_HANDLE) {
    _logger->error("failed to build the pipeline because of a null shader module: {}",
                   _shaderFileName);
  }

  VkPipelineShaderStageCreateInfo shaderStageInfo{};
  shaderStageInfo.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  shaderStageInfo.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
  shaderStageInfo.module = _cachedShaderModule;
  shaderStageInfo.pName  = "main"; // name of the entry function of current shader

  VkComputePipelineCreateInfo computePipelineCreateInfo{};
  computePipelineCreateInfo.sType  = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
  computePipelineCreateInfo.layout = _pipelineLayout;
  computePipelineCreateInfo.flags  = 0;
  computePipelineCreateInfo.stage  = shaderStageInfo;

  vkCreateComputePipelines(_appContext->getDevice(), VK_NULL_HANDLE, 1, &computePipelineCreateInfo,
                           nullptr, &_pipeline);
}

bool ComputePipeline::buildAndCacheShaderModule(bool allowBuildFail) {
  auto const shaderSourceCode =
      ShaderFileReader::readShaderSourceCode(kRootDir + "src/shaders/" + _shaderFileName, _logger);
  auto const _shaderCode = _shaderCompiler->compileComputeShader(_shaderFileName, shaderSourceCode);

  if (!allowBuildFail && !_shaderCode.has_value()) {
    _logger->error("failed to compile the shader: {}", _shaderFileName);
    exit(0);
  }

  if (_shaderCode.has_value()) {
    _cleanupShaderModule();
    _cachedShaderModule = _createShaderModule(_shaderCode.value());
    return true;
  }
  return false;
}

void ComputePipeline::recordCommand(VkCommandBuffer commandBuffer, uint32_t currentFrame,
                                    uint32_t threadCountX, uint32_t threadCountY,
                                    uint32_t threadCountZ) {
  _bind(commandBuffer, currentFrame);
  vkCmdDispatch(commandBuffer, std::ceil((float)threadCountX / (float)_workGroupSize.x),
                std::ceil((float)threadCountY / (float)_workGroupSize.y),
                std::ceil((float)threadCountZ / (float)_workGroupSize.z));
}

void ComputePipeline::recordIndirectCommand(VkCommandBuffer commandBuffer, uint32_t currentFrame,
                                            VkBuffer indirectBuffer) {
  _bind(commandBuffer, currentFrame);
  vkCmdDispatchIndirect(commandBuffer, indirectBuffer, 0);
}