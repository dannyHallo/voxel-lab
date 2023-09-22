#include "DeviceCreator.h"

#include "Common.h"
#include "utils/Logger.h"

#include <set>

namespace {
bool queueIndicesAreFilled(const QueueFamilyIndices &indices) {
  return indices.computeFamily != -1 && indices.transferFamily != -1 &&
         indices.graphicsFamily != -1 && indices.presentFamily != -1;
}

bool findQueueFamilies(QueueFamilyIndices &indices,
                       const VkPhysicalDevice &physicalDevice,
                       const VkSurfaceKHR &surface) {
  uint32_t queueFamilyCount = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount,
                                           nullptr);
  std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
  vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount,
                                           queueFamilies.data());

  for (uint32_t i = 0; i < queueFamilyCount; ++i) {
    const auto &queueFamily = queueFamilies[i];

    if (indices.computeFamily == -1) {
      if ((queueFamily.queueFlags & VK_QUEUE_COMPUTE_BIT) != 0) {
        indices.computeFamily = i;
      }
    }

    if (indices.transferFamily == -1) {
      if ((queueFamily.queueFlags & VK_QUEUE_TRANSFER_BIT) != 0) {
        indices.transferFamily = i;
      }
    }

    if (indices.graphicsFamily == -1) {
      if ((queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT) != 0) {
        uint32_t presentSupport = 0;
        vkGetPhysicalDeviceSurfaceSupportKHR(physicalDevice, i, surface,
                                             &presentSupport);
        if (presentSupport != 0) {
          indices.graphicsFamily = i;
          indices.presentFamily  = i;
        }
      }
    }

    if (queueIndicesAreFilled(indices)) {
      return true;
    }
  }
  return false;
}

VkSampleCountFlagBits getDeviceMaxUsableSampleCount(VkPhysicalDevice device) {
  VkPhysicalDeviceProperties physicalDeviceProperties;
  vkGetPhysicalDeviceProperties(device, &physicalDeviceProperties);

  VkSampleCountFlags counts =
      physicalDeviceProperties.limits.framebufferColorSampleCounts &
      physicalDeviceProperties.limits.framebufferDepthSampleCounts;
  if ((counts & VK_SAMPLE_COUNT_64_BIT) != 0) {
    return VK_SAMPLE_COUNT_64_BIT;
  }
  if ((counts & VK_SAMPLE_COUNT_32_BIT) != 0) {
    return VK_SAMPLE_COUNT_32_BIT;
  }
  if ((counts & VK_SAMPLE_COUNT_16_BIT) != 0) {
    return VK_SAMPLE_COUNT_16_BIT;
  }
  if ((counts & VK_SAMPLE_COUNT_8_BIT) != 0) {
    return VK_SAMPLE_COUNT_8_BIT;
  }
  if ((counts & VK_SAMPLE_COUNT_4_BIT) != 0) {
    return VK_SAMPLE_COUNT_4_BIT;
  }
  if ((counts & VK_SAMPLE_COUNT_2_BIT) != 0) {
    return VK_SAMPLE_COUNT_2_BIT;
  }

  return VK_SAMPLE_COUNT_1_BIT;
}

bool checkDeviceExtensionSupport(
    const VkPhysicalDevice &physicalDevice,
    const std::vector<const char *> &requiredDeviceExtensions) {
  uint32_t extensionCount = 0;
  vkEnumerateDeviceExtensionProperties(physicalDevice, nullptr, &extensionCount,
                                       nullptr);

  std::vector<VkExtensionProperties> availableExtensions(extensionCount);
  vkEnumerateDeviceExtensionProperties(physicalDevice, nullptr, &extensionCount,
                                       availableExtensions.data());

  std::set<std::string> availableExtensionsSet{};
  for (const auto &extension : availableExtensions) {
    availableExtensionsSet.insert(
        static_cast<const char *>(extension.extensionName));
  }

  Logger::print("available device extensions count",
                availableExtensions.size());
  Logger::print();
  Logger::print("using device extensions", requiredDeviceExtensions.size());
  for (const auto &extensionName : requiredDeviceExtensions) {
    Logger::print("\t", extensionName);
  }
  Logger::print();
  Logger::print();

  std::vector<std::string> unavailableExtensionNames{};
  for (const auto &requiredExtension : requiredDeviceExtensions) {
    if (availableExtensionsSet.find(requiredExtension) ==
        availableExtensionsSet.end()) {
      unavailableExtensionNames.emplace_back(requiredExtension);
    }
  }

  if (unavailableExtensionNames.empty()) {
    return true;
  }

  Logger::print("the following device extensions are not available:");
  for (const auto &unavailableExtensionName : unavailableExtensionNames) {
    Logger::print("\t", unavailableExtensionName.c_str());
  }
  return false;
}

// this function is also called in swapchain creation step
// so check if this overhead can be eliminated
// query for physical device's swapchain sepport details
SwapchainSupportDetails querySwapchainSupport(VkSurfaceKHR surface,
                                              VkPhysicalDevice physicalDevice) {
  // get swapchain support details using surface and device info
  SwapchainSupportDetails details;

  // get capabilities
  vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physicalDevice, surface,
                                            &details.capabilities);

  // get surface format
  uint32_t formatCount = 0;
  vkGetPhysicalDeviceSurfaceFormatsKHR(physicalDevice, surface, &formatCount,
                                       nullptr);
  if (formatCount != 0) {
    details.formats.resize(formatCount);
    vkGetPhysicalDeviceSurfaceFormatsKHR(physicalDevice, surface, &formatCount,
                                         details.formats.data());
  }

  // get available presentation modes
  uint32_t presentModeCount = 0;
  vkGetPhysicalDeviceSurfacePresentModesKHR(physicalDevice, surface,
                                            &presentModeCount, nullptr);
  if (presentModeCount != 0) {
    details.presentModes.resize(presentModeCount);
    vkGetPhysicalDeviceSurfacePresentModesKHR(physicalDevice, surface,
                                              &presentModeCount,
                                              details.presentModes.data());
  }

  return details;
}

void checkDeviceSuitable(
    const VkSurfaceKHR &surface, const VkPhysicalDevice &physicalDevice,
    const std::vector<const char *> &requiredDeviceExtensions) {
  // Check if the queue family is valid
  QueueFamilyIndices indices{};
  bool indicesAreFilled = findQueueFamilies(indices, physicalDevice, surface);
  // Check extension support
  bool extensionSupported =
      checkDeviceExtensionSupport(physicalDevice, requiredDeviceExtensions);
  bool swapChainAdequate = false;
  if (extensionSupported) {
    SwapchainSupportDetails swapChainSupport =
        querySwapchainSupport(surface, physicalDevice);
    swapChainAdequate = !swapChainSupport.formats.empty() &&
                        !swapChainSupport.presentModes.empty();
  }

  // Query for device features if needed
  // VkPhysicalDeviceFeatures supportedFeatures;
  // vkGetPhysicalDeviceFeatures(physicalDevice, &supportedFeatures);
  if (indicesAreFilled && extensionSupported && swapChainAdequate) {
    return;
  }

  Logger::throwError("physical device not suitable");
}

// helper function to customize the physical device ranking mechanism, returns
// the physical device with the highest score the marking criteria should be
// further optimized
VkPhysicalDevice
selectBestDevice(const std::vector<VkPhysicalDevice> &physicalDevices,
                 const VkSurfaceKHR &surface,
                 const std::vector<const char *> &requiredDeviceExtensions) {
  VkPhysicalDevice bestDevice = VK_NULL_HANDLE;

  static constexpr uint32_t kDiscreteGpuMark   = 100;
  static constexpr uint32_t kIntegratedGpuMark = 20;

  // Give marks to all devices available, returns the best usable device
  std::vector<uint32_t> deviceMarks(physicalDevices.size());
  size_t deviceId = 0;

  Logger::print("-------------------------------------------------------");

  for (const auto &physicalDevice : physicalDevices) {

    VkPhysicalDeviceProperties deviceProperty;
    vkGetPhysicalDeviceProperties(physicalDevice, &deviceProperty);

    // Discrete GPU will mark better
    if (deviceProperty.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
      deviceMarks[deviceId] += kDiscreteGpuMark;
    } else if (deviceProperty.deviceType ==
               VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU) {
      deviceMarks[deviceId] += kIntegratedGpuMark;
    }

    VkPhysicalDeviceMemoryProperties memoryProperty;
    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memoryProperty);

    auto *heapsPointer =
        static_cast<VkMemoryHeap *>(memoryProperty.memoryHeaps);
    auto heaps = std::vector<VkMemoryHeap>(
        heapsPointer, heapsPointer + memoryProperty.memoryHeapCount);

    size_t deviceMemory = 0;
    for (const auto &heap : heaps) {
      // At least one heap has this flag
      if ((heap.flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) != 0) {
        deviceMemory += heap.size;
      }
    }

    // MSAA
    VkSampleCountFlagBits msaaSamples =
        getDeviceMaxUsableSampleCount(physicalDevice);

    std::cout << "Device " << deviceId << "    "
              << static_cast<const char *>(deviceProperty.deviceName)
              << "    Memory in bytes: " << deviceMemory
              << "    MSAA max sample count: " << msaaSamples
              << "    Mark: " << deviceMarks[deviceId] << "\n";

    deviceId++;
  }

  Logger::print("-------------------------------------------------------");
  Logger::print();

  uint32_t bestMark = 0;
  deviceId          = 0;

  for (const auto &deviceMark : deviceMarks) {
    if (deviceMark > bestMark) {
      bestMark   = deviceMark;
      bestDevice = physicalDevices[deviceId];
    }

    deviceId++;
  }

  if (bestDevice == VK_NULL_HANDLE) {
    Logger::throwError("no suitable GPU found.");
  } else {
    VkPhysicalDeviceProperties bestDeviceProperty;
    vkGetPhysicalDeviceProperties(bestDevice, &bestDeviceProperty);
    std::cout << "Selected: "
              << static_cast<const char *>(bestDeviceProperty.deviceName)
              << std::endl;
    Logger::print();

    checkDeviceSuitable(surface, bestDevice, requiredDeviceExtensions);
  }
  return bestDevice;
}
} // namespace

// pick the most suitable physical device, and create logical device from it
void DeviceCreator::create(
    VkPhysicalDevice &physicalDevice, VkDevice &device,
    QueueFamilyIndices &indices, VkQueue &graphicsQueue, VkQueue &presentQueue,
    VkQueue &computeQueue, VkQueue &transferQueue, const VkInstance &instance,
    VkSurfaceKHR surface,
    const std::vector<const char *> &requiredDeviceExtensions) {
  // pick the physical device with the best performance
  {
    physicalDevice = VK_NULL_HANDLE;

    uint32_t deviceCount = 0;
    vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);
    if (deviceCount == 0) {
      Logger::throwError("failed to find GPUs with Vulkan support!");
    }

    std::vector<VkPhysicalDevice> physicalDevices(deviceCount);
    vkEnumeratePhysicalDevices(instance, &deviceCount, physicalDevices.data());

    physicalDevice =
        selectBestDevice(physicalDevices, surface, requiredDeviceExtensions);
  }

  // create logical device from the physical device we've picked
  {
    findQueueFamilies(indices, physicalDevice, surface);

    std::set<uint32_t> queueFamilyIndicesSet = {
        indices.graphicsFamily, indices.presentFamily, indices.computeFamily,
        indices.transferFamily};

    std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
    float queuePriority = 1.F; // ranges from 0 - 1.;
    for (uint32_t queueFamilyIndex : queueFamilyIndicesSet) {
      VkDeviceQueueCreateInfo queueCreateInfo{
          VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
      queueCreateInfo.queueFamilyIndex = queueFamilyIndex;
      queueCreateInfo.queueCount       = 1;
      queueCreateInfo.pQueuePriorities = &queuePriority;
      queueCreateInfos.push_back(queueCreateInfo);
    }

    VkPhysicalDeviceFeatures2 physicalDeviceFeatures{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};

    VkPhysicalDeviceBufferDeviceAddressFeatures bufferDeviceAddress = {
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES};
    bufferDeviceAddress.bufferDeviceAddress = VK_TRUE;

    VkPhysicalDeviceRayTracingPipelineFeaturesKHR rayTracingPipeline = {
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR};
    rayTracingPipeline.pNext              = &bufferDeviceAddress;
    rayTracingPipeline.rayTracingPipeline = VK_TRUE;

    VkPhysicalDeviceAccelerationStructureFeaturesKHR rayTracingStructure = {
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR};
    rayTracingStructure.pNext                 = &rayTracingPipeline;
    rayTracingStructure.accelerationStructure = VK_TRUE;

    VkPhysicalDeviceDescriptorIndexingFeatures descriptorIndexing = {
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_FEATURES};
    // descriptorIndexing.pNext = &rayTracingStructure; // uncomment this to
    // enable the features above

    physicalDeviceFeatures.pNext = &descriptorIndexing;

    vkGetPhysicalDeviceFeatures2(
        physicalDevice,
        &physicalDeviceFeatures); // enable all the features our GPU has

    VkDeviceCreateInfo deviceCreateInfo{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    deviceCreateInfo.pNext = &physicalDeviceFeatures;
    deviceCreateInfo.queueCreateInfoCount =
        static_cast<uint32_t>(queueCreateInfos.size());
    deviceCreateInfo.pQueueCreateInfos = queueCreateInfos.data();
    // createInfo.pEnabledFeatures = &deviceFeatures;
    deviceCreateInfo.pEnabledFeatures = nullptr;

    // enabling device extensions
    deviceCreateInfo.enabledExtensionCount =
        static_cast<uint32_t>(requiredDeviceExtensions.size());
    deviceCreateInfo.ppEnabledExtensionNames = requiredDeviceExtensions.data();

    // The enabledLayerCount and ppEnabledLayerNames fields of
    // VkDeviceCreateInfo are ignored by up-to-date implementations.
    deviceCreateInfo.enabledLayerCount   = 0;
    deviceCreateInfo.ppEnabledLayerNames = nullptr;

    VkResult result =
        vkCreateDevice(physicalDevice, &deviceCreateInfo, nullptr, &device);
    Logger::checkStep("vkCreateDevice", result);

    // reduce loading overhead by specifing only one device is used
    volkLoadDevice(device);

    vkGetDeviceQueue(device, indices.graphicsFamily, 0, &graphicsQueue);
    vkGetDeviceQueue(device, indices.presentFamily, 0, &presentQueue);
    vkGetDeviceQueue(device, indices.computeFamily, 0, &computeQueue);
    vkGetDeviceQueue(device, indices.transferFamily, 0, &transferQueue);

    // // if raytracing support requested - let's get raytracing properties to
    // // know shader header size and max recursion
    // mRTProps =
    // {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR};
    // VkPhysicalDeviceProperties2
    // devProps{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2}; devProps.pNext
    // = &mRTProps; devProps.properties = {};

    // vkGetPhysicalDeviceProperties2(physicalDevice, &devProps);
  }
}