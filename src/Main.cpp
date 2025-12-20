/*
 * BSD 2-Clause License
 *
 * Copyright (c) 2025, Christoph Neuhauser
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * * Redistributions of source code must retain the above copyright notice, this
 *   list of conditions and the following disclaimer.
 *
 * * Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#define _USE_MATH_DEFINES
#include <cmath>
#include <iostream>
#include <memory>
#include <utility>

#include <Math/Math.hpp>
#include <Math/half/half.hpp>
#include <Utils/AppSettings.hpp>
#include <Utils/File/Logfile.hpp>
#include <Graphics/Vulkan/Utils/Instance.hpp>
#include <Graphics/Vulkan/Utils/Device.hpp>
#include <Graphics/Vulkan/Buffers/Buffer.hpp>
#include <Graphics/Vulkan/Shader/ShaderManager.hpp>
#include <Graphics/Vulkan/Render/Renderer.hpp>
#include <Graphics/Vulkan/Render/CommandBuffer.hpp>
#include <Graphics/Vulkan/Render/ComputePipeline.hpp>
#include <Graphics/Vulkan/Render/Data.hpp>
#include <ImGui/Widgets/NumberFormatting.hpp>

#ifdef __linux__
#include <fstream>
#endif

#ifdef _WIN32
#include "WindowsUtils.hpp"
#endif

#define RES_TO_STR(r) case r: return #r

std::string getComponentTypeString(VkComponentTypeKHR compType) {
    switch (compType) {
        case VK_COMPONENT_TYPE_FLOAT16_KHR:
            return "float16";
        case VK_COMPONENT_TYPE_FLOAT32_KHR:
            return "float32";
        case VK_COMPONENT_TYPE_FLOAT64_KHR:
            return "float64";
        case VK_COMPONENT_TYPE_SINT8_KHR:
            return "sint8";
        case VK_COMPONENT_TYPE_SINT16_KHR:
            return "sint16";
        case VK_COMPONENT_TYPE_SINT32_KHR:
            return "sint32";
        case VK_COMPONENT_TYPE_SINT64_KHR:
            return "sint64";
        case VK_COMPONENT_TYPE_UINT8_KHR:
            return "uint8";
        case VK_COMPONENT_TYPE_UINT16_KHR:
            return "uint16";
        case VK_COMPONENT_TYPE_UINT32_KHR:
            return "uint32";
        case VK_COMPONENT_TYPE_UINT64_KHR:
            return "uint64";
        case VK_COMPONENT_TYPE_BFLOAT16_KHR:
            return "bloat16";
        case VK_COMPONENT_TYPE_SINT8_PACKED_NV:
            return "sint8_packed";
        case VK_COMPONENT_TYPE_UINT8_PACKED_NV:
            return "uint8_packed";
        case VK_COMPONENT_TYPE_FLOAT_E4M3_NV:
            return "float_e4m3";
        case VK_COMPONENT_TYPE_FLOAT_E5M2_NV:
            return "float_e5m2";
        default:
            return "UNKNOWN";
    }
}


namespace sgl {
// Override for nicer formating of bools.
inline std::string toString(bool boolVal) {
    return boolVal ? "true" : "false";
}

template <class T>
std::string toStringScientific(T obj) {
    std::ostringstream ostr;
    ostr << std::scientific;
    ostr << obj;
    return ostr.str();
}
}

template<typename... T>
void writeOut(T... args) {
    std::string text = (std::string() + ... + sgl::toString(std::move(args)));
    sgl::Logfile::get()->write(text, sgl::BLACK);
    std::cout << text << std::endl;
}

std::string uint8ArrayToHex(const uint8_t* arr, size_t numEntries) {
    std::string hexRep;
    for (size_t i = 0; i < numEntries; i++) {
        hexRep += sgl::toHexString(uint32_t(arr[i]));
    }
    return hexRep;
}

// https://stackoverflow.com/questions/54460727/is-there-a-standard-function-for-computing-units-in-the-last-place-in-c
template<typename T>
std::enable_if_t<std::is_floating_point_v<T>, T> ulp(T x) {
    if (x > 0) {
        return std::nexttoward(x, std::numeric_limits<T>::infinity()) - x;
    } else {
        return x - std::nexttoward(x, -std::numeric_limits<T>::infinity());
    }
}

// https://registry.khronos.org/vulkan/specs/latest/html/vkspec.html#spirvenv-op-prec
template<typename float_tpl>
void checkTrigonometricFunctionPrecision(
        sgl::vk::Device* device, const std::string& functionGlslName, std::function<float_tpl(float_tpl)> trigFn,
        const std::string& floatTypeGlslName, float_tpl minRangeFloat, float_tpl maxRangeFloat,
        bool minInclusive, bool maxInclusive) {
    //using uint_tpl = std::conditional_t<sizeof(float_tpl) == sizeof(uint32_t), uint32_t, uint16_t>;

    size_t batchBufferSize = std::min(size_t(1 << 30), size_t(device->getLimits().maxStorageBufferRange));
    size_t maxNumValuesCollected = sgl::ulceil(batchBufferSize, 4ull);
    batchBufferSize = batchBufferSize * 4ull;
    sgl::vk::BufferPtr stagingBuffer = std::make_shared<sgl::vk::Buffer>(
            device, batchBufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_CPU_ONLY);
    sgl::vk::BufferPtr inputBuffer = std::make_shared<sgl::vk::Buffer>(
            device, batchBufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            VMA_MEMORY_USAGE_GPU_ONLY);
    sgl::vk::BufferPtr outputBuffer = std::make_shared<sgl::vk::Buffer>(
            device, batchBufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            VMA_MEMORY_USAGE_GPU_ONLY);

    // Create renderer and command buffer.
    auto renderer = new sgl::vk::Renderer(device);
    sgl::vk::CommandPoolType commandPoolType;
    commandPoolType.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    auto commandBuffer = std::make_shared<sgl::vk::CommandBuffer>(device, commandPoolType);

    std::map<std::string, std::string> preprocessorDefines = {
            { "float_tpl", floatTypeGlslName },
            { "trig_fn", functionGlslName },
    };

    const char* SHADER_STRING_WRITE_IMAGE_COMPUTE = R"(
    #version 450 core
    layout(local_size_x = 512, local_size_y = 1, local_size_z = 1) in;
    layout(binding = 0) restrict readonly buffer SourceBuffer {
        float_tpl srcArray[];
    };
    layout(binding = 1) restrict writeonly buffer DestBuffer {
        float_tpl destArray[];
    };
    layout(push_constant) uniform PushConstants {
        uint NUM_VALUES;
    };
    void main() {
        uint idx = gl_GlobalInvocationID.x;
        if (idx < NUM_VALUES) {
            destArray[idx] = trig_fn(srcArray[idx]);
        }
    }
    )";
    auto* shaderManager = new sgl::vk::ShaderManagerVk(device);
    auto shaderStages = shaderManager->compileComputeShaderFromStringCached(
            floatTypeGlslName + "_" + functionGlslName + ".Compute", SHADER_STRING_WRITE_IMAGE_COMPUTE,
            preprocessorDefines);
    sgl::vk::ComputePipelineInfo computePipelineInfo(shaderStages);
    sgl::vk::ComputePipelinePtr computePipeline = std::make_shared<sgl::vk::ComputePipeline>(device, computePipelineInfo);
    auto computeData = std::make_shared<sgl::vk::ComputeData>(renderer, computePipeline);
    computeData->setStaticBuffer(inputBuffer, 0);
    computeData->setStaticBuffer(outputBuffer, 1);

    float_tpl maxAbsError = 0.0f;
    uint64_t maxUlpError = 0;

    if (!minInclusive) {
        minRangeFloat = std::nexttoward(minRangeFloat, std::numeric_limits<float_tpl>::infinity());
        minRangeFloat += 2.0f * std::numeric_limits<float_tpl>::epsilon();
    }
    if (!maxInclusive) {
        maxRangeFloat = std::nexttoward(maxRangeFloat, -std::numeric_limits<float_tpl>::infinity());
        maxRangeFloat -= 2.0f * std::numeric_limits<float_tpl>::epsilon();
    }

    std::vector<float> correctOutputValues;
    correctOutputValues.reserve(maxNumValuesCollected);
    auto* ptr = static_cast<float*>(stagingBuffer->mapMemory());
    for (float value = minRangeFloat; value <= maxRangeFloat; value = std::nextafter(value, std::numeric_limits<float>::max())) {
        ptr[correctOutputValues.size()] = value;
        correctOutputValues.push_back(trigFn(value));

        if (correctOutputValues.size() >= maxNumValuesCollected) {
            stagingBuffer->unmapMemory();

            renderer->pushCommandBuffer(commandBuffer);
            renderer->beginCommandBuffer();
            stagingBuffer->copyDataTo(inputBuffer, renderer->getVkCommandBuffer());
            renderer->insertBufferMemoryBarrier(
                    VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                    VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
                    inputBuffer);
            renderer->pushConstants(
                    computeData->getComputePipeline(), VK_SHADER_STAGE_COMPUTE_BIT,
                    0, uint32_t(correctOutputValues.size()));
            renderer->dispatch(computeData, sgl::uiceil(uint32_t(correctOutputValues.size()), 512u), 1, 1);
            renderer->insertBufferMemoryBarrier(
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                    VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT,
                    inputBuffer);
            outputBuffer->copyDataTo(stagingBuffer, renderer->getVkCommandBuffer());
            renderer->endCommandBuffer();
            renderer->submitToQueueImmediate();

            ptr = static_cast<float_tpl*>(stagingBuffer->mapMemory());
            for (size_t idx = 0; idx < correctOutputValues.size(); idx++) {
                float_tpl valVulkan = ptr[idx];
                float_tpl valCpu = correctOutputValues[idx];
                float_tpl diff = std::abs(valVulkan - valCpu);
                maxAbsError = std::max(maxAbsError, diff);
                auto ulpExact = ulp(valCpu);
                auto ulpDiff = std::ceil(std::abs(double(valCpu) - double(valVulkan)) / double(ulpExact));
                maxUlpError = std::max(maxUlpError, uint64_t(ulpDiff));
            }

            correctOutputValues.clear();
        }
    }
    stagingBuffer->unmapMemory();

    writeOut(
            "Maximum error for ", floatTypeGlslName, ", ", functionGlslName,
            ": abs ", sgl::toStringScientific(maxAbsError), ", ULP ", maxUlpError);

    computeData = {};
    delete renderer;
}

template<typename float_tpl>
void checkTrigonometricFunctionPrecision(
        sgl::vk::Device* device, const std::string& functionGlslName, std::function<float_tpl(float_tpl)> trigFn,
        const std::string& floatTypeGlslName, float_tpl minRangeFloat, float_tpl maxRangeFloat) {
    checkTrigonometricFunctionPrecision(
            device, functionGlslName, std::move(trigFn), floatTypeGlslName, minRangeFloat, maxRangeFloat, true, true);
}
template<typename float_tpl>
void checkTrigonometricFunctionPrecision(
        sgl::vk::Device* device, const std::string& functionGlslName, std::function<float_tpl(float_tpl)> trigFn,
        const std::string& floatTypeGlslName, float_tpl minRangeFloat, float_tpl maxRangeFloat, bool rangeInclusive) {
    checkTrigonometricFunctionPrecision(
            device, functionGlslName, std::move(trigFn), floatTypeGlslName, minRangeFloat, maxRangeFloat, rangeInclusive, rangeInclusive);
}

void checkVulkanDeviceFeatures(sgl::vk::Device* device) {
    sgl::Logfile::get()->write("<br>");
    writeOut(std::string() + "Device name: " + device->getDeviceName());
    if (device->getPhysicalDeviceProperties().apiVersion >= VK_API_VERSION_1_1) {
        writeOut("Device driver name: ", device->getDeviceDriverName());
        writeOut("Device driver info: ", device->getDeviceDriverInfo());
        writeOut("Device driver ID: ", device->getDeviceDriverId());
        writeOut("Device driver version: ", device->getDriverVersionString());
        writeOut("Device vendor ID: 0x", sgl::toHexString(device->getPhysicalDeviceProperties().vendorID));
        writeOut("Device ID: 0x", sgl::toHexString(device->getPhysicalDeviceProperties().deviceID));
        //writeOut("Device driver UUID: ", uint8ArrayToHex(device->getDeviceIDProperties().deviceUUID, VK_UUID_SIZE));
        //writeOut("Device driver LUID: ", uint8ArrayToHex(device->getDeviceIDProperties().deviceLUID, VK_UUID_SIZE));
    }

    writeOut("");
    writeOut("Default subgroup size: ", device->getPhysicalDeviceSubgroupProperties().subgroupSize);
    if (device->getPhysicalDeviceVulkan13Features().subgroupSizeControl
            && (device->getPhysicalDeviceVulkan13Properties().requiredSubgroupSizeStages & VK_SHADER_STAGE_COMPUTE_BIT) != 0) {
        writeOut("Min subgroup size: ", device->getPhysicalDeviceVulkan13Properties().minSubgroupSize);
        writeOut("Max subgroup size: ", device->getPhysicalDeviceVulkan13Properties().maxSubgroupSize);
    }

    writeOut("");
    writeOut("Max memory allocations: ", device->getLimits().maxMemoryAllocationCount);
    writeOut(
            "Max storage buffer range: ",
            sgl::getNiceMemoryStringDifference(device->getLimits().maxStorageBufferRange, 2, true));
    if (device->getPhysicalDeviceProperties().apiVersion >= VK_API_VERSION_1_1) {
        writeOut(
                "Max memory allocation size: ",
                sgl::getNiceMemoryStringDifference(device->getMaxMemoryAllocationSize(), 2, true));
    }
    writeOut("Supports shader 64-bit indexing: ", device->getShader64BitIndexingFeaturesEXT().shader64BitIndexing ? "Yes" : "No");
    writeOut("alignof(std::max_align_t): ", alignof(std::max_align_t));
    writeOut("Min imported host pointer alignment: ", device->getMinImportedHostPointerAlignment());

    /*
     * On Linux, dedicated NVIDIA GPUs seem to have (as of 2025-11-09) the following heaps:
     * - Heap 0: Full VRAM
     * - Heap 1: RAM
     * - Heap 2, only if ReBAR not supported: 246MiB
     *
     * Also, they seem to have at least 4 memory types with property flags != 0:
     * - MEMORY_PROPERTY_DEVICE_LOCAL_BIT: Heap 0
     * - MEMORY_PROPERTY_HOST_VISIBLE_BIT, MEMORY_PROPERTY_HOST_COHERENT_BIT: Heap 1
     * - MEMORY_PROPERTY_HOST_VISIBLE_BIT, MEMORY_PROPERTY_HOST_COHERENT_BIT, MEMORY_PROPERTY_HOST_CACHED_BIT: Heap 1
     * - MEMORY_PROPERTY_DEVICE_LOCAL_BIT, MEMORY_PROPERTY_HOST_VISIBLE_BIT, MEMORY_PROPERTY_HOST_COHERENT_BIT:
     *   - If ReBAR: Heap 0
     *   - If no ReBAR: Heap 2
     */
    std::vector<std::string> flagsStringMap = {
            "device local",  // VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
            "host visible",  // VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
            "host coherent", // VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
            "host cached"    // VK_MEMORY_PROPERTY_HOST_CACHED_BIT
    };
    const VkPhysicalDeviceMemoryProperties& deviceMemoryProperties = device->getMemoryProperties();
    for (uint32_t heapIdx = 0; heapIdx < deviceMemoryProperties.memoryHeapCount; heapIdx++) {
        VkMemoryPropertyFlagBits typeFlags{};
        for (uint32_t memoryTypeIdx = 0; memoryTypeIdx < deviceMemoryProperties.memoryTypeCount; memoryTypeIdx++) {
            if (deviceMemoryProperties.memoryTypes[memoryTypeIdx].heapIndex == heapIdx) {
                typeFlags = VkMemoryPropertyFlagBits(typeFlags | deviceMemoryProperties.memoryTypes[memoryTypeIdx].propertyFlags);
            }
        }
        std::string memoryHeapInfo;
        if (typeFlags != 0) {
            memoryHeapInfo = " (";
            typeFlags = VkMemoryPropertyFlagBits(typeFlags & 0xF);
            auto numEntries = int(sgl::popcount(uint32_t(typeFlags)));
            int entryIdx = 0;
            for (int i = 0; i < 4; i++) {
                auto flag = VkMemoryPropertyFlagBits(1 << i);
                if ((typeFlags & flag) != 0) {
                    memoryHeapInfo += flagsStringMap[i];
                    if (entryIdx != numEntries - 1) {
                        memoryHeapInfo += ", ";
                    }
                    entryIdx++;
                }
            }
            memoryHeapInfo += ")";
        }
        bool hasTypeDeviceLocal = (typeFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) != 0;
        bool isHeapDeviceLocal = (deviceMemoryProperties.memoryHeaps[heapIdx].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) != 0;
        if (hasTypeDeviceLocal != isHeapDeviceLocal) {
            sgl::Logfile::get()->writeError("Encountered memory heap with mismatching heap and type flags.");
        }
        writeOut(
                "Memory heap #", heapIdx, ": ",
                sgl::getNiceMemoryStringDifference(deviceMemoryProperties.memoryHeaps[heapIdx].size, 2, true),
                memoryHeapInfo);
    }

    writeOut("");
    writeOut("Shader float16 support: ", bool(device->getPhysicalDeviceVulkan12Features().shaderFloat16));
    writeOut("Shader bfloat16 support: ", bool(device->getPhysicalDeviceShaderBfloat16Features().shaderBFloat16Type));

    checkTrigonometricFunctionPrecision<float>(
            device, "sin", [](float value){ return std::sin(value); }, "float", float(-M_PI), float(M_PI));
    checkTrigonometricFunctionPrecision<float>(
            device, "cos", [](float value){ return std::cos(value); }, "float", float(-M_PI), float(M_PI));
    checkTrigonometricFunctionPrecision<float>(
            device, "tan", [](float value){ return std::tan(value); }, "float",
            float(-0.5 * M_PI), float(0.5 * M_PI), false);
    checkTrigonometricFunctionPrecision<float>(
            device, "atan", [](float value){ return std::atan(value); }, "float",
            std::numeric_limits<float>::lowest(), std::numeric_limits<float>::max());
    checkTrigonometricFunctionPrecision<float>(
            device, "asin", [](float value){ return std::asin(value); }, "float", -1.0f, 1.0f);
    checkTrigonometricFunctionPrecision<float>(
            device, "acos", [](float value){ return std::acos(value); }, "float", -1.0f, 1.0f);
    checkTrigonometricFunctionPrecision<float>(
            device, "exp", [](float value){ return std::exp(value); }, "float",
            0.5f, 2.0f);
    checkTrigonometricFunctionPrecision<float>(
            device, "exp2", [](float value){ return std::exp2(value); }, "float",
            0.5f, 2.0f);
    checkTrigonometricFunctionPrecision<float>(
            device, "log", [](float value){ return std::log(value); }, "float",
            0.0f, std::numeric_limits<float>::max(), false, true);
    checkTrigonometricFunctionPrecision<float>(
            device, "log2", [](float value){ return std::log2(value); }, "float",
            0.0f, std::numeric_limits<float>::max(), false, true);
    checkTrigonometricFunctionPrecision<float>(
            device, "sqrt", [](float value){ return std::sqrt(value); }, "float",
            0.0f, std::numeric_limits<float>::max(), false, true);
    checkTrigonometricFunctionPrecision<float>(
            device, "inversesqrt", [](float value){ return 1.0f / std::sqrt(value); }, "float",
            0.0f, std::numeric_limits<float>::max());
}

int main(int argc, char *argv[]) {
    for (int i = 1; i < argc; i++) {
        std::string command = argv[i];
        if (command == "--help" || command == "-h") {
            std::cout << "TestVulkanTrigPrecision: Queries Vulkan cooperative matrix support." << std::endl;
        } else {
            throw std::runtime_error("Invalid command line arguments.");
        }
    }

    sgl::Logfile::get()->createLogfile("Logfile.html", "TestVulkanTrigPrecision");
    sgl::Logfile::get()->write("\n<style>\n");
    sgl::Logfile::get()->write("table {\nborder-spacing: 10px 0;\n}\n");
    sgl::Logfile::get()->write("</style>\n");

    sgl::AppSettings::get()->setSaveSettings(false);
    sgl::AppSettings::get()->getSettings().addKeyValue("window-debugContext", false);

    auto* instance = new sgl::vk::Instance;
    instance->createInstance({}, false);

    std::vector<const char*> optionalDeviceExtensions;
    std::vector<const char*> requiredDeviceExtensions = {
            VK_EXT_SCALAR_BLOCK_LAYOUT_EXTENSION_NAME, VK_KHR_TIMELINE_SEMAPHORE_EXTENSION_NAME
    };
    sgl::vk::DeviceFeatures requestedDeviceFeatures{};
    requestedDeviceFeatures.optionalVulkan11Features.storageBuffer16BitAccess = VK_TRUE;
    requestedDeviceFeatures.optionalVulkan12Features.shaderFloat16 = VK_TRUE;

    std::vector<VkPhysicalDevice> physicalDevices = sgl::vk::enumeratePhysicalDevices(instance);
    std::vector<VkPhysicalDevice> suitablePhysicalDevices;
    for (auto& physicalDevice : physicalDevices) {
        if (sgl::vk::checkIsPhysicalDeviceSuitable(
                instance, physicalDevice, nullptr, requiredDeviceExtensions, requestedDeviceFeatures, true)) {
            suitablePhysicalDevices.push_back(physicalDevice);
        }
    }
    for (size_t i = 0; i < suitablePhysicalDevices.size(); i++) {
        if (i != 0) {
            std::cout << std::endl << "--------------------------------------------" << std::endl << std::endl;
        }
        sgl::Logfile::get()->write("<br><hr><br>\n");
        auto physicalDevice = suitablePhysicalDevices.at(i);
        auto* device = new sgl::vk::Device;
        device->createDeviceHeadlessFromPhysicalDevice(
                instance, physicalDevice, requiredDeviceExtensions,
                optionalDeviceExtensions, requestedDeviceFeatures, false);
        checkVulkanDeviceFeatures(device);
        delete device;
        if (i == suitablePhysicalDevices.size() - 1) {
            sgl::Logfile::get()->write("<br><hr>\n");
        }
    }

    delete instance;

#ifdef _WIN32
    pauseIfAppOwnsConsole();
#endif

    return 0;
}
