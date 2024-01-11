#pragma once

#include "utils/incl/Glm.hpp"
#include <algorithm>
#include <cstdint>
#include <cstdlib> /* srand, rand */
#include <stack>
#include <vector>

/**
 * Geometry and material objects to be used on GPU. To minimize data size
 * integer links are used.
 */
namespace GpuModel {
enum MaterialType { LightSource, Lambertian, Metal, Glass };
enum ColorInputType { Normalized, EightBit };

struct Material {
  MaterialType type;
  alignas(16) glm::vec3 albedo{};

  Material(MaterialType t, glm::vec3 a, ColorInputType c = ColorInputType::Normalized) : type(t) {
    if (c == ColorInputType::Normalized) {
      albedo = a;
    } else {
      albedo = a / 255.F;
    }
  }
};

// a tri holds a single material index
struct Triangle {
  alignas(16) glm::vec3 v0;
  alignas(16) glm::vec3 v1;
  alignas(16) glm::vec3 v2;
  uint32_t materialIndex;
  uint32_t meshHash;
};

struct Sphere {
  alignas(16) glm::vec4 s;
  alignas(4) uint32_t materialIndex;
};

// Node in a non recursive BHV for use on GPU.
struct BvhNode {
  alignas(16) glm::vec3 min;
  alignas(16) glm::vec3 max;
  alignas(4) int leftNodeIndex  = -1;
  alignas(4) int rightNodeIndex = -1;
  alignas(4) int objectIndex    = -1;
};

// Model of light used for importance sampling.
struct Light {
  // Index in the array of triangles.
  uint32_t triangleIndex;
  // Area of the triangle;
  alignas(4) float area;
};

} // namespace GpuModel