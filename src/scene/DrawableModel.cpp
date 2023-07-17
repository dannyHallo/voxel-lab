#pragma once

#include "utils/vulkan.h"

#include "DrawableModel.h"
#include "Material.h"
#include "Mesh.h"
#include "memory/Buffer.h"

#include <memory>
#include <vector>

DrawableModel::DrawableModel(std::shared_ptr<Material> material, std::string modelPath) : mMaterial(material) {
  Mesh m(modelPath);

  initVertexBuffer(m);
  initIndexBuffer(m);
}

DrawableModel::DrawableModel(std::shared_ptr<Material> material, MeshType type) : mMaterial(material) {
  Mesh m(type);

  initVertexBuffer(m);
  initIndexBuffer(m);
}

std::shared_ptr<Material> DrawableModel::getMaterial() { return mMaterial; }

void DrawableModel::drawCommand(VkCommandBuffer &commandBuffer, size_t currentFrame) {
  mMaterial->bind(commandBuffer, currentFrame);
  VkBuffer vertexBuffers[] = {mVertexBuffer->getVkBuffer()};
  VkDeviceSize offsets[]   = {0};

  vkCmdBindVertexBuffers(commandBuffer, 0, 1, vertexBuffers, offsets);
  vkCmdBindIndexBuffer(commandBuffer, mIndexBuffer->getVkBuffer(), 0, VK_INDEX_TYPE_UINT32);

  VkDeviceSize indirect_offset = 0;
  uint32_t draw_stride         = sizeof(VkDrawIndirectCommand);

  // execute the draw command buffer on each section as defined by the array of draws
  // vkCmdDrawIndexedIndirect(commandBuffer, indirectBuffer.buffer, indirect_offset, 1, draw_stride);
  vkCmdDrawIndexed(commandBuffer, static_cast<uint32_t>(mNumIndices), 1, 0, 0, 0);
}

void DrawableModel::initVertexBuffer(const Mesh &mesh) {
  mVertexBuffer = std::make_shared<Buffer>(sizeof(Vertex) * mesh.vertices.size(), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                                           VMA_MEMORY_USAGE_CPU_TO_GPU, mesh.vertices.data());
}

void DrawableModel::initIndexBuffer(const Mesh &mesh) {
  mIndexBuffer = std::make_shared<Buffer>(sizeof(uint32_t) * mesh.indices.size(), VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
                                          VMA_MEMORY_USAGE_CPU_TO_GPU, mesh.indices.data());
}