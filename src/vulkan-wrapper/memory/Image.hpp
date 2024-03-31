#pragma once

#include "volk.h"

#include "vma/vk_mem_alloc.h"

#include <string>
#include <vector>

// the wrapper class of VkImage and its corresponding VkImageView, handles
// memory allocation
class Image {
public:
  // create a blank image
  Image(uint32_t width, uint32_t height, uint32_t depth, VkFormat format, VkImageUsageFlags usage,
        VkSampler sampler                = VK_NULL_HANDLE,
        VkImageLayout initialImageLayout = VK_IMAGE_LAYOUT_GENERAL,
        VkSampleCountFlagBits numSamples = VK_SAMPLE_COUNT_1_BIT,
        VkImageTiling tiling             = VK_IMAGE_TILING_OPTIMAL,
        VkImageAspectFlags aspectFlags   = VK_IMAGE_ASPECT_COLOR_BIT);

  // create an image from a file, VK_FORMAT_R8G8B8A8_UNORM is the only format that stb_image
  // supports, so the created image format is fixed, and only 2D images are supported.
  Image(const std::string &filename, VkImageUsageFlags usage, VkSampler sampler = VK_NULL_HANDLE,
        VkImageLayout initialImageLayout = VK_IMAGE_LAYOUT_GENERAL,
        VkSampleCountFlagBits numSamples = VK_SAMPLE_COUNT_1_BIT,
        VkImageTiling tiling             = VK_IMAGE_TILING_OPTIMAL,
        VkImageAspectFlags aspectFlags   = VK_IMAGE_ASPECT_COLOR_BIT);

  // create a texture array from a set of image files, all images should be in
  // the same dimension and the same format..
  Image(const std::vector<std::string> &filenames, VkImageUsageFlags usage,
        VkSampler sampler                = VK_NULL_HANDLE,
        VkImageLayout initialImageLayout = VK_IMAGE_LAYOUT_GENERAL,
        VkSampleCountFlagBits numSamples = VK_SAMPLE_COUNT_1_BIT,
        VkImageTiling tiling             = VK_IMAGE_TILING_OPTIMAL,
        VkImageAspectFlags aspectFlags   = VK_IMAGE_ASPECT_COLOR_BIT);

  ~Image();

  // disable move and copy
  Image(const Image &)            = delete;
  Image &operator=(const Image &) = delete;
  Image(Image &&)                 = delete;
  Image &operator=(Image &&)      = delete;

  VkImage &getVkImage() { return _vkImage; }
  // VmaAllocation &getAllocation() { return mAllocation; }
  // VkImageView &getImageView() { return mVkImageView; }
  [[nodiscard]] VkDescriptorImageInfo getDescriptorInfo(VkImageLayout imageLayout) const;
  [[nodiscard]] uint32_t getWidth() const { return _width; }
  [[nodiscard]] uint32_t getHeight() const { return _height; }

  void clearImage(VkCommandBuffer commandBuffer);

  static VkImageView createImageView(VkDevice device, const VkImage &image, VkFormat format,
                                     VkImageAspectFlags aspectFlags, uint32_t imageDepth = 1,
                                     uint32_t layerCount = 1);

private:
  VkImage _vkImage          = VK_NULL_HANDLE;
  VkImageView _vkImageView  = VK_NULL_HANDLE;
  VkSampler _vkSampler      = VK_NULL_HANDLE;
  VmaAllocation _allocation = VK_NULL_HANDLE;
  VkImageLayout _currentImageLayout;
  uint32_t _layerCount;
  VkFormat _format;

  uint32_t _width;
  uint32_t _height;
  uint32_t _depth;

  void _copyDataToImage(unsigned char *imageData, uint32_t layerToCopyTo = 0);

  // creates an image with VK_IMAGE_LAYOUT_UNDEFINED initially
  VkResult _createImage(VkSampleCountFlagBits numSamples, VkImageTiling tiling,
                        VkImageUsageFlags usage);

  void _transitionImageLayout(VkImageLayout newLayout);
};

// storing the pointer of a pair of imgs, support for easy dumping
class ImageForwardingPair {
public:
  ImageForwardingPair(VkImage image1, VkImage image2, uint32_t width, uint32_t height,
                      VkImageLayout image1BeforeCopy, VkImageLayout image2BeforeCopy,
                      VkImageLayout image1AfterCopy, VkImageLayout image2AfterCopy);

  void forwardCopy(VkCommandBuffer commandBuffer);

private:
  VkImage _image1;
  VkImage _image2;

  VkImageCopy _copyRegion{};
  VkImageMemoryBarrier _image1BeforeCopy{};
  VkImageMemoryBarrier _image2BeforeCopy{};
  VkImageMemoryBarrier _image1AfterCopy{};
  VkImageMemoryBarrier _image2AfterCopy{};
};