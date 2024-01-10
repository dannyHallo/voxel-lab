#include "SvoScene.hpp"

#include "svo-ray-tracing/im-data/ImCoor.hpp"
#include "utils/logger/Logger.hpp"

#include <cassert>
#include <iomanip> // for advanced cout formatting
#include <iostream>

namespace {
// the best method for counting digit 1s in a 32 bit uint32_t in parallel
// https://graphics.stanford.edu/~seander/bithacks.html#CountBitsSetParallel
uint32_t bitCount(uint32_t v) {
  const std::vector<uint32_t> S{1U, 2U, 4U, 8U, 16U}; // Magic Binary Numbers
  const std::vector<uint32_t> B{0x55555555U, 0x33333333U, 0x0F0F0F0FU, 0x00FF00FFU, 0x0000FFFFU};

  uint32_t c = v - ((v >> 1) & B[0]);
  c          = ((c >> S[1]) & B[1]) + (c & B[1]);
  c          = ((c >> S[2]) + c) & B[2];
  c          = ((c >> S[3]) + c) & B[3];
  c          = ((c >> S[4]) + c) & B[4];
  return c;
}

void _printHexFormat(const std::vector<uint32_t> &vec) {
  for (size_t i = 0; i < vec.size(); ++i) {
    std::cout << "0x" << std::uppercase << std::setfill('0') << std::setw(8) << std::hex << vec[i]
              << "u";

    if (i != vec.size() - 1) {
      std::cout << ", ";
    }

    if ((i + 1) % 8 == 0) {
      std::cout << std::endl;
    }
  }
  std::cout << std::endl;
}

} // namespace

SvoScene::SvoScene(Logger *logger) : _logger(logger) { _run(); }

void SvoScene::_run() {
  _buildImageDatas();
  // _printImageDatas();
  _createBuffer();
  // _printBuffer();
  _logger->print("SvoScene::_run() done!");
}

void SvoScene::_buildImageDatas() {
  ImCoor3D const kBaseImageSize = {4, 4, 4};
  ImCoor3D const kRootImageSize = {1, 1, 1};

  // assert all components in the size should be a power of 2
  // https://stackoverflow.com/questions/1053582/how-does-this-bitwise-operation-check-for-a-power-of-2
  assert((kBaseImageSize.x & (kBaseImageSize.x - 1)) == 0);
  assert((kBaseImageSize.y & (kBaseImageSize.y - 1)) == 0);
  assert((kBaseImageSize.z & (kBaseImageSize.z - 1)) == 0);

  // create the base iamge data
  auto imageData = std::make_unique<ImageData>(kBaseImageSize);
  _imageDatas.push_back(std::move(imageData));

  int imIdx = 0;
  BaseLevelBuilder::build(_imageDatas[imIdx].get(), kBaseImageSize);

  while (_imageDatas[imIdx]->getImageSize() != kRootImageSize) {
    // std::cout << "imIdx: " << imIdx << std::endl;
    auto newImageData = std::make_unique<ImageData>(_imageDatas[imIdx]->getImageSize() / 2);
    _imageDatas.push_back(std::move(newImageData));
    UpperLevelBuilder::build(_imageDatas[imIdx].get(), _imageDatas[imIdx + 1].get());
    imIdx++;
  }
}

void SvoScene::_printImageDatas() {
  for (auto const &imageData : _imageDatas) {
    std::cout << "imageData->getImageSize(): " << imageData->getImageSize().x << " "
              << imageData->getImageSize().y << " " << imageData->getImageSize().z << std::endl;

    for (auto const &[coor, data] : imageData->getImageData()) {
      std::cout << "coor: " << coor.x << " " << coor.y << " " << coor.z << std::endl;
      std::cout << "data: " << std::hex << data << std::endl;
    }
  }
}

void SvoScene::_createBuffer() {
  int imIdx = static_cast<int>(_imageDatas.size() - 1); // start with the root image

  uint32_t accum = 1;

  std::vector<ImCoor3D> activeCoorsPrev{};

  // process the root node
  auto const &imageData = _imageDatas[imIdx];

  auto const coor      = ImCoor3D{0, 0, 0};
  uint32_t const &data = imageData->imageLoad(coor);
  if (data == 0U) {
    return;
  }
  activeCoorsPrev.push_back(coor);

  // change the data by filling the first 16 bits with the next bit offset
  uint32_t dataToUse = data | (accum << 16);
  _buffer.push_back(dataToUse);
  accum += bitCount(data & 0x0000FF00); // accum the bit offset

  // debug
  // std::cout << "coor: " << coor.x << " " << coor.y << " " << coor.z << std::endl;
  // std::cout << "data: " << std::hex << data << std::endl;

  while (--imIdx >= 0) {
    auto const &imageData = _imageDatas[imIdx];
    std::vector<ImCoor3D> thisTimeActiveCoors{};
    for (auto const activeCoor : activeCoorsPrev) {
      auto const remappedOrigin = activeCoor * 2;
      for (int x = 0; x < 2; x++) {
        for (int y = 0; y < 2; y++) {
          for (int z = 0; z < 2; z++) {
            auto const coor      = remappedOrigin + ImCoor3D{x, y, z};
            uint32_t const &data = imageData->imageLoad(coor);
            if (data == 0U) {
              continue;
            }
            thisTimeActiveCoors.push_back(coor);

            // change the data by filling the first 16 bits with the next bit offset
            uint32_t dataToUse = data | (accum << 16);
            _buffer.push_back(dataToUse);
            accum += bitCount(data & 0x0000FF00); // accum the bit offset

            // debug
            // std::cout << "coor: " << coor.x << " " << coor.y << " " << coor.z << std::endl;
            // std::cout << "data: " << std::hex << data << std::endl;
          }
        }
      }
    }
    activeCoorsPrev = thisTimeActiveCoors;
  }
}

void SvoScene::_printBuffer() {
  std::cout << "the buffer is: \n" << std::endl;
  _printHexFormat(_buffer);
}
