#include "Material.hpp"
#include "material/DescriptorSetBundle.hpp"

#include <cassert>
#include <map>
#include <memory>
#include <vector>

static const std::map<VkShaderStageFlags, VkPipelineBindPoint> kShaderStageFlagsToBindPoint{
    {VK_SHADER_STAGE_VERTEX_BIT, VK_PIPELINE_BIND_POINT_GRAPHICS},
    {VK_SHADER_STAGE_FRAGMENT_BIT, VK_PIPELINE_BIND_POINT_GRAPHICS},
    {VK_SHADER_STAGE_COMPUTE_BIT, VK_PIPELINE_BIND_POINT_COMPUTE}};

void Material::init() { _createPipeline(); }

VkShaderModule Material::_createShaderModule(const std::vector<uint32_t> &code) {
  VkShaderModuleCreateInfo createInfo{};
  createInfo.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
  createInfo.pCode    = code.data();
  createInfo.codeSize = sizeof(uint32_t) * code.size(); // size in BYTES

  VkShaderModule shaderModule = VK_NULL_HANDLE;
  VkResult result =
      vkCreateShaderModule(_appContext->getDevice(), &createInfo, nullptr, &shaderModule);
  assert(result == VK_SUCCESS && "Material::createShaderModule: failed to create shader module");

  return shaderModule;
}

Material::~Material() {
  vkDestroyPipeline(_appContext->getDevice(), _pipeline, nullptr);
  vkDestroyPipelineLayout(_appContext->getDevice(), _pipelineLayout, nullptr);
}

void Material::addStorageImage(Image *storageImage) { _storageImages.emplace_back(storageImage); }

void Material::addUniformBufferBundle(BufferBundle *uniformBufferBundle) {
  _uniformBufferBundles.emplace_back(uniformBufferBundle);
}

void Material::addStorageBufferBundle(BufferBundle *storageBufferBundle) {
  _storageBufferBundles.emplace_back(storageBufferBundle);
}

void Material::bind(VkCommandBuffer commandBuffer, size_t currentFrame) {
  _bind(kShaderStageFlagsToBindPoint.at(_shaderStageFlags), commandBuffer, currentFrame);
}

void Material::_bind(VkPipelineBindPoint pipelineBindPoint, VkCommandBuffer commandBuffer,
                     size_t currentFrame) {
  vkCmdBindDescriptorSets(commandBuffer, pipelineBindPoint, _pipelineLayout, 0, 1,
                          &_descriptorSetBundle->getDescriptorSet(currentFrame), 0, nullptr);
  vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, _pipeline);
}