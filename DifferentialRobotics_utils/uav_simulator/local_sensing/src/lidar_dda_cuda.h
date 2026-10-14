#ifndef LIDAR_DDA_CUDA_H
#define LIDAR_DDA_CUDA_H

#include <cstdint>
#include <string>
#include <vector>

struct CudaDdaContext;

CudaDdaContext* cudaDdaCreate(int size_x, int size_y, int size_z,
                              float lower_x, float lower_y, float lower_z,
                              float resolution,
                              const std::vector<float>& sensor_directions,
                              std::string* error);

void cudaDdaDestroy(CudaDdaContext* context);

bool cudaDdaUploadOccupancy(CudaDdaContext* context,
                            const std::vector<uint8_t>& occupancy,
                            std::string* error);

bool cudaDdaCast(CudaDdaContext* context, const float origin[3],
                 const float rotation[9], float min_range, float max_range,
                 std::vector<float>* ranges, std::vector<uint8_t>* hits,
                 float* elapsed_ms, std::string* error);

#endif
