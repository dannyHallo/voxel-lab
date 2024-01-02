#include "Material.hpp"
#include "utils/io/Readfile.hpp"
#include "utils/logger/Logger.hpp"

#include <cassert>
#include <map>
#include <memory>
#include <vector>

static std::map<VkShaderStageFlags, VkPipelineBindPoint> kShaderStageFlagsToBindPoint{
    {VK_SHADER_STAGE_VERTEX_BIT, VK_PIPELINE_BIND_POINT_GRAPHICS},
    {VK_SHADER_STAGE_FRAGMENT_BIT, VK_PIPELINE_BIND_POINT_GRAPHICS},
    {VK_SHADER_STAGE_COMPUTE_BIT, VK_PIPELINE_BIND_POINT_COMPUTE}};

VkShaderModule Material::_createShaderModule(const std::vector<char> &code) {
  VkShaderModuleCreateInfo createInfo{};
  createInfo.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
  createInfo.codeSize = code.size();
  createInfo.pCode    = reinterpret_cast<const uint32_t *>(code.data());

  VkShaderModule shaderModule = VK_NULL_HANDLE;
  VkResult result =
      vkCreateShaderModule(_appContext->getDevice(), &createInfo, nullptr, &shaderModule);
  assert(result == VK_SUCCESS && "Material::createShaderModule: failed to create shader module");

  return shaderModule;
}

Material::~Material() {
  vkDestroyDescriptorSetLayout(_appContext->getDevice(), _descriptorSetLayout, nullptr);
  vkDestroyPipeline(_appContext->getDevice(), _pipeline, nullptr);
  vkDestroyPipelineLayout(_appContext->getDevice(), _pipelineLayout, nullptr);

  // mDescriptorSets be automatically freed when the descriptor pool is
  // destroyed
  vkDestroyDescriptorPool(_appContext->getDevice(), _descriptorPool, nullptr);
}

void Material::addStorageImage(Image *storageImage) { _storageImages.emplace_back(storageImage); }

void Material::addUniformBufferBundle(BufferBundle *uniformBufferBundle) {
  _uniformBufferBundles.emplace_back(uniformBufferBundle);
}

void Material::addStorageBufferBundle(BufferBundle *storageBufferBundle) {
  _storageBufferBundles.emplace_back(storageBufferBundle);
}

void Material::_createDescriptorSetLayout() {
  // creates descriptor set layout that will be used to create every descriptor
  // set features are extracted from buffer bundles, not buffers, thus to be
  // reused in descriptor set creation
  std::vector<VkDescriptorSetLayoutBinding> bindings{};
  uint32_t bindingNo = 0;

  // each binding is for each buffer BUNDLE
  for (int i = 0; i < _uniformBufferBundles.size(); i++) {
    VkDescriptorSetLayoutBinding uboLayoutBinding{};
    uboLayoutBinding.binding         = bindingNo++;
    uboLayoutBinding.descriptorCount = 1;
    uboLayoutBinding.descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    uboLayoutBinding.stageFlags      = _shaderStageFlags;
    bindings.push_back(uboLayoutBinding);
  }

  for (int i = 0; i < _storageImages.size(); i++) {
    VkDescriptorSetLayoutBinding samplerLayoutBinding{};
    samplerLayoutBinding.binding         = bindingNo++;
    samplerLayoutBinding.descriptorCount = 1;
    samplerLayoutBinding.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    samplerLayoutBinding.stageFlags      = _shaderStageFlags;
    bindings.push_back(samplerLayoutBinding);
  }

  for (int i = 0; i < _storageBufferBundles.size(); i++) {
    VkDescriptorSetLayoutBinding storageBufferBinding{};
    storageBufferBinding.binding         = bindingNo++;
    storageBufferBinding.descriptorCount = 1;
    storageBufferBinding.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    storageBufferBinding.stageFlags      = _shaderStageFlags;
    bindings.push_back(storageBufferBinding);
  }

  VkDescriptorSetLayoutCreateInfo layoutInfo{};
  layoutInfo.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
  layoutInfo.pBindings    = bindings.data();

  VkResult result = vkCreateDescriptorSetLayout(_appContext->getDevice(), &layoutInfo, nullptr,
                                                &_descriptorSetLayout);
  assert(result == VK_SUCCESS &&
         "Material::initDescriptorSetLayout: failed to create descriptor set layout");
}

void Material::_createDescriptorPool() {
  std::vector<VkDescriptorPoolSize> poolSizes{};
  uint32_t swapchainSize = _appContext->getSwapchainSize();

  // https://www.reddit.com/r/vulkan/comments/8u9zqr/having_trouble_understanding_descriptor_pool/
  // pool sizes info indicates how many descriptors of a certain type can be
  // allocated from the pool - NOT THE SET!
  if (_uniformBufferBundles.size() > 0) {
    poolSizes.emplace_back(
        VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                             static_cast<uint32_t>(swapchainSize * _uniformBufferBundles.size())});
  }
  if (_storageImages.size() > 0) {
    poolSizes.emplace_back(
        VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                             static_cast<uint32_t>(swapchainSize * _storageImages.size())});
  }
  if (_storageBufferBundles.size() > 0) {
    poolSizes.emplace_back(
        VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                             static_cast<uint32_t>(swapchainSize * _storageBufferBundles.size())});
  }

  VkDescriptorPoolCreateInfo poolInfo{};
  poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  // the max number of descriptor sets that can be allocated from this pool
  poolInfo.maxSets       = swapchainSize;
  poolInfo.pPoolSizes    = poolSizes.data();
  poolInfo.poolSizeCount = poolSizes.size();

  VkResult result =
      vkCreateDescriptorPool(_appContext->getDevice(), &poolInfo, nullptr, &_descriptorPool);
  assert(result == VK_SUCCESS && "Material::initDescriptorPool: failed to create descriptor pool");
}

// chop the buffer bundles here, and create the descriptor sets for each of the
// swapchains buffers are loaded to descriptor sets
void Material::_createDescriptorSets() {
  uint32_t swapchainSize = _appContext->getSwapchainSize();

  std::vector<VkDescriptorSetLayout> layouts(swapchainSize, _descriptorSetLayout);

  VkDescriptorSetAllocateInfo allocInfo{};
  allocInfo.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
  allocInfo.descriptorPool     = _descriptorPool;
  allocInfo.descriptorSetCount = swapchainSize;
  allocInfo.pSetLayouts        = layouts.data();

  _descriptorSets.resize(swapchainSize);
  VkResult res =
      vkAllocateDescriptorSets(_appContext->getDevice(), &allocInfo, _descriptorSets.data());
  assert(res == VK_SUCCESS && "Material::initDescriptorSets: failed to allocate descriptor sets");

  for (size_t j = 0; j < swapchainSize; j++) {
    VkDescriptorSet &dstSet = _descriptorSets[j];

    std::vector<VkWriteDescriptorSet> descriptorWrites{};
    descriptorWrites.reserve(_uniformBufferBundles.size() + _storageImages.size() +
                             _storageBufferBundles.size());

    uint32_t bindingNo = 0;

    std::vector<VkDescriptorBufferInfo> uniformBufferInfos{};
    uniformBufferInfos.reserve(_uniformBufferBundles.size());
    for (auto &uniformBufferBundle : _uniformBufferBundles) {
      uniformBufferInfos.emplace_back(uniformBufferBundle->getBuffer(j)->getDescriptorInfo());
    }

    for (size_t i = 0; i < _uniformBufferBundles.size(); i++) {
      VkWriteDescriptorSet descriptorWrite{};
      descriptorWrite.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
      descriptorWrite.dstSet          = dstSet;
      descriptorWrite.dstBinding      = bindingNo++;
      descriptorWrite.dstArrayElement = 0;
      descriptorWrite.descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
      descriptorWrite.descriptorCount = 1;
      descriptorWrite.pBufferInfo     = &uniformBufferInfos[i];
      descriptorWrites.push_back(descriptorWrite);
    }

    std::vector<VkDescriptorImageInfo> storageImageInfos{};
    storageImageInfos.reserve(_storageImages.size());
    for (auto &storageImage : _storageImages) {
      storageImageInfos.push_back(storageImage->getDescriptorInfo(VK_IMAGE_LAYOUT_GENERAL));
    }

    for (size_t i = 0; i < _storageImages.size(); i++) {
      VkWriteDescriptorSet descriptorWrite{};
      descriptorWrite.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
      descriptorWrite.dstSet          = dstSet;
      descriptorWrite.dstBinding      = bindingNo++;
      descriptorWrite.dstArrayElement = 0;
      descriptorWrite.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
      descriptorWrite.descriptorCount = 1;
      descriptorWrite.pImageInfo      = &storageImageInfos[i];
      descriptorWrites.push_back(descriptorWrite);
    }

    std::vector<VkDescriptorBufferInfo> storageBufferInfos{};
    storageBufferInfos.reserve(_storageBufferBundles.size());
    for (auto &storageBufferBundle : _storageBufferBundles) {
      storageBufferInfos.push_back(storageBufferBundle->getBuffer(j)->getDescriptorInfo());
    }

    for (size_t i = 0; i < _storageBufferBundles.size(); i++) {
      VkWriteDescriptorSet descriptorWrite{};
      descriptorWrite.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
      descriptorWrite.dstSet          = dstSet;
      descriptorWrite.dstBinding      = bindingNo++;
      descriptorWrite.dstArrayElement = 0;
      descriptorWrite.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      descriptorWrite.descriptorCount = 1;
      descriptorWrite.pBufferInfo     = &storageBufferInfos[i];

      descriptorWrites.push_back(descriptorWrite);
    }

    vkUpdateDescriptorSets(_appContext->getDevice(), static_cast<uint32_t>(descriptorWrites.size()),
                           descriptorWrites.data(), 0, nullptr);
  }
}

void Material::bind(VkCommandBuffer commandBuffer, size_t currentFrame) {
  assert(kShaderStageFlagsToBindPoint.contains(_shaderStageFlags) &&
         "Material::bind: shader stage flag not supported");
  _bind(kShaderStageFlagsToBindPoint[_shaderStageFlags], commandBuffer, currentFrame);
}

void Material::_bind(VkPipelineBindPoint pipelineBindPoint, VkCommandBuffer commandBuffer,
                     size_t currentFrame) {
  vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, _pipelineLayout, 0, 1,
                          &_descriptorSets[currentFrame], 0, nullptr);

  vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, _pipeline);
}