#include "Buffer.hpp"
#include "utils/Logger.hpp"

Buffer::Buffer(VkDeviceSize size, VkBufferUsageFlags usage,
               VmaMemoryUsage memoryUsage, const void *data)
    : mVkBuffer(VK_NULL_HANDLE), mAllocation(VK_NULL_HANDLE), mSize(size) {
  allocate(size, usage, memoryUsage);
  fillData(data);
}

Buffer::~Buffer() {
  if (mVkBuffer != VK_NULL_HANDLE) {
    vmaDestroyBuffer(VulkanApplicationContext::getInstance()->getAllocator(),
                     mVkBuffer, mAllocation);
    mVkBuffer = VK_NULL_HANDLE;
  }
}

void Buffer::allocate(VkDeviceSize size, VkBufferUsageFlags usage,
                      VmaMemoryUsage memoryUsage) {
  mSize = size;

  VkBufferCreateInfo bufferInfo{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
  bufferInfo.size  = size;
  bufferInfo.usage = usage;
  bufferInfo.sharingMode =
      VK_SHARING_MODE_EXCLUSIVE; // used in only one queue family

  VmaAllocationCreateInfo vmaallocInfo = {};
  vmaallocInfo.usage                   = memoryUsage;

  VkResult result = vmaCreateBuffer(
      VulkanApplicationContext::getInstance()->getAllocator(), &bufferInfo,
      &vmaallocInfo, &mVkBuffer, &mAllocation, nullptr);
  Logger::checkStep("vmaCreateBuffer", result);
}

void Buffer::fillData(const void *data) {
  // a pointer to the first byte of the allocated memory
  void *mappedData;
  vmaMapMemory(VulkanApplicationContext::getInstance()->getAllocator(),
               mAllocation, &mappedData);

  if (data != nullptr)
    memcpy(mappedData, data, mSize);
  else
    memset(mappedData, 0, mSize);

  vmaUnmapMemory(VulkanApplicationContext::getInstance()->getAllocator(),
                 mAllocation);
}

std::shared_ptr<Buffer> BufferBundle::getBuffer(size_t index) {
  if (index < 0 || index >= mBuffers.size()) {
    Logger::throwError("BufferBundle::getBuffer: index out of range");
    return nullptr; // (unreachable code)
  }
  return mBuffers[index];
}

void BufferBundle::fillData(const void *data) {
  for (auto &buffer : mBuffers) buffer->fillData(data);
}