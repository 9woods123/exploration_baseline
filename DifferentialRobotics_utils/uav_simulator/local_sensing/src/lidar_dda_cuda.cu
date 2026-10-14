#include "lidar_dda_cuda.h"

#include <cuda_runtime.h>
#include <cmath>
#include <sstream>

struct CudaDdaContext {
  int size_x;
  int size_y;
  int size_z;
  int ray_count;
  float lower[3];
  float resolution;
  uint8_t* occupancy;
  float* sensor_directions;
  float* rotation;
  float* ranges;
  uint8_t* hits;
  cudaEvent_t start;
  cudaEvent_t stop;
};

namespace {

bool check(cudaError_t result, const char* operation, std::string* error) {
  if (result == cudaSuccess) return true;
  if (error) {
    std::ostringstream stream;
    stream << operation << ": " << cudaGetErrorString(result);
    *error = stream.str();
  }
  return false;
}

__device__ __forceinline__ bool inside(int x, int y, int z,
                                       const CudaDdaContext context) {
  return x >= 0 && x < context.size_x && y >= 0 && y < context.size_y &&
         z >= 0 && z < context.size_z;
}

__global__ void castDdaKernel(CudaDdaContext context, float3 origin,
                              float min_range,
                              float max_range) {
  const int ray = blockIdx.x * blockDim.x + threadIdx.x;
  if (ray >= context.ray_count) return;

  const float sx = context.sensor_directions[3 * ray];
  const float sy = context.sensor_directions[3 * ray + 1];
  const float sz = context.sensor_directions[3 * ray + 2];
  const float direction[3] = {
      context.rotation[0] * sx + context.rotation[1] * sy + context.rotation[2] * sz,
      context.rotation[3] * sx + context.rotation[4] * sy + context.rotation[5] * sz,
      context.rotation[6] * sx + context.rotation[7] * sy + context.rotation[8] * sz};
  const float position[3] = {origin.x, origin.y, origin.z};

  int voxel[3];
  int step[3];
  float t_max[3];
  float t_delta[3];
  for (int axis = 0; axis < 3; ++axis) {
    voxel[axis] = static_cast<int>(floorf(
        (position[axis] - context.lower[axis]) / context.resolution));
  }

  if (!inside(voxel[0], voxel[1], voxel[2], context)) {
    context.hits[ray] = 0;
    context.ranges[ray] = max_range;
    return;
  }

  for (int axis = 0; axis < 3; ++axis) {
    if (direction[axis] > 1e-8f) {
      step[axis] = 1;
      const float boundary = context.lower[axis] +
          static_cast<float>(voxel[axis] + 1) * context.resolution;
      t_max[axis] = (boundary - position[axis]) / direction[axis];
      t_delta[axis] = context.resolution / direction[axis];
    } else if (direction[axis] < -1e-8f) {
      step[axis] = -1;
      const float boundary = context.lower[axis] +
          static_cast<float>(voxel[axis]) * context.resolution;
      t_max[axis] = (boundary - position[axis]) / direction[axis];
      t_delta[axis] = -context.resolution / direction[axis];
    } else {
      step[axis] = 0;
      t_max[axis] = 1.0e30f;
      t_delta[axis] = 1.0e30f;
    }
  }

  float distance = 0.0f;
  while (distance <= max_range &&
         inside(voxel[0], voxel[1], voxel[2], context)) {
    const size_t address =
        (static_cast<size_t>(voxel[0]) * context.size_y + voxel[1]) *
            context.size_z + voxel[2];
    if (distance >= min_range && context.occupancy[address]) {
      context.hits[ray] = 1;
      context.ranges[ray] = fminf(distance + 1e-3f, max_range);
      return;
    }

    const float next = fminf(t_max[0], fminf(t_max[1], t_max[2]));
    if (!isfinite(next) || next > max_range) break;
    const float tie_epsilon = 1e-6f;
    for (int axis = 0; axis < 3; ++axis) {
      if (t_max[axis] <= next + tie_epsilon) {
        voxel[axis] += step[axis];
        t_max[axis] += t_delta[axis];
      }
    }
    distance = next;
  }

  context.hits[ray] = 0;
  context.ranges[ray] = max_range;
}

void release(CudaDdaContext* context) {
  if (!context) return;
  if (context->occupancy) cudaFree(context->occupancy);
  if (context->sensor_directions) cudaFree(context->sensor_directions);
  if (context->rotation) cudaFree(context->rotation);
  if (context->ranges) cudaFree(context->ranges);
  if (context->hits) cudaFree(context->hits);
  if (context->start) cudaEventDestroy(context->start);
  if (context->stop) cudaEventDestroy(context->stop);
  delete context;
}

}  // namespace

CudaDdaContext* cudaDdaCreate(int size_x, int size_y, int size_z,
                              float lower_x, float lower_y, float lower_z,
                              float resolution,
                              const std::vector<float>& sensor_directions,
                              std::string* error) {
  int device_count = 0;
  if (!check(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount", error) ||
      device_count == 0) {
    if (device_count == 0 && error) *error = "no CUDA device found";
    return nullptr;
  }

  CudaDdaContext* context = new CudaDdaContext();
  context->size_x = size_x;
  context->size_y = size_y;
  context->size_z = size_z;
  context->ray_count = static_cast<int>(sensor_directions.size() / 3);
  context->lower[0] = lower_x;
  context->lower[1] = lower_y;
  context->lower[2] = lower_z;
  context->resolution = resolution;
  context->occupancy = nullptr;
  context->sensor_directions = nullptr;
  context->rotation = nullptr;
  context->ranges = nullptr;
  context->hits = nullptr;
  context->start = nullptr;
  context->stop = nullptr;

  const size_t voxel_count =
      static_cast<size_t>(size_x) * size_y * size_z;
  if (!check(cudaMalloc(&context->occupancy, voxel_count * sizeof(uint8_t)),
             "cudaMalloc occupancy", error) ||
      !check(cudaMalloc(&context->sensor_directions,
                        sensor_directions.size() * sizeof(float)),
             "cudaMalloc directions", error) ||
      !check(cudaMalloc(&context->ranges,
                        context->ray_count * sizeof(float)),
             "cudaMalloc ranges", error) ||
      !check(cudaMalloc(&context->rotation, 9 * sizeof(float)),
             "cudaMalloc rotation", error) ||
      !check(cudaMalloc(&context->hits,
                        context->ray_count * sizeof(uint8_t)),
             "cudaMalloc hits", error) ||
      !check(cudaMemcpy(context->sensor_directions, sensor_directions.data(),
                        sensor_directions.size() * sizeof(float),
                        cudaMemcpyHostToDevice),
             "cudaMemcpy directions", error) ||
      !check(cudaEventCreate(&context->start), "cudaEventCreate start", error) ||
      !check(cudaEventCreate(&context->stop), "cudaEventCreate stop", error)) {
    release(context);
    return nullptr;
  }
  return context;
}

void cudaDdaDestroy(CudaDdaContext* context) { release(context); }

bool cudaDdaUploadOccupancy(CudaDdaContext* context,
                            const std::vector<uint8_t>& occupancy,
                            std::string* error) {
  const size_t expected = static_cast<size_t>(context->size_x) *
                          context->size_y * context->size_z;
  if (occupancy.size() != expected) {
    if (error) *error = "occupancy grid size mismatch";
    return false;
  }
  return check(cudaMemcpy(context->occupancy, occupancy.data(), expected,
                          cudaMemcpyHostToDevice),
               "cudaMemcpy occupancy", error);
}

bool cudaDdaCast(CudaDdaContext* context, const float origin[3],
                 const float rotation[9], float min_range, float max_range,
                 std::vector<float>* ranges, std::vector<uint8_t>* hits,
                 float* elapsed_ms, std::string* error) {
  if (!check(cudaMemcpy(context->rotation, rotation, 9 * sizeof(float),
                        cudaMemcpyHostToDevice),
             "cudaMemcpy rotation", error)) {
    return false;
  }

  cudaEventRecord(context->start);
  const int threads = 128;
  const int blocks = (context->ray_count + threads - 1) / threads;
  castDdaKernel<<<blocks, threads>>>(*context,
      make_float3(origin[0], origin[1], origin[2]), min_range, max_range);
  cudaEventRecord(context->stop);
  if (!check(cudaGetLastError(), "castDdaKernel launch", error) ||
      !check(cudaEventSynchronize(context->stop),
             "castDdaKernel execution", error)) {
    return false;
  }

  ranges->resize(context->ray_count);
  hits->resize(context->ray_count);
  const bool ok =
      check(cudaMemcpy(ranges->data(), context->ranges,
                       context->ray_count * sizeof(float),
                       cudaMemcpyDeviceToHost),
            "cudaMemcpy ranges", error) &&
      check(cudaMemcpy(hits->data(), context->hits,
                       context->ray_count * sizeof(uint8_t),
                       cudaMemcpyDeviceToHost),
            "cudaMemcpy hits", error);
  if (elapsed_ms) cudaEventElapsedTime(elapsed_ms, context->start, context->stop);
  return ok;
}
