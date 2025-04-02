#include "glm/ext/quaternion_trigonometric.hpp"
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iterator>
#include <memory>
#include <optional>
#include <vector>
#include <array>
#include <chrono>

#include <stdlib.h>
#include <stdio.h>

// libraries - ignore all warnings
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Weverything"

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <vulkan/vulkan.h>
#include <vulkan/vulkan_core.h>
#include <vulkan/vk_enum_string_helper.h>

#include <GLFW/glfw3.h>

#define STB_IMAGE_IMPLEMENTATION
#include "third_party/stb_image.h"
#include "third_party/VkBootstrap.h"
#include "third_party/VkBootstrapDispatch.h"

#define VMA_IMPLEMENTATION
#define VMA_VULKAN_VERSION 100000000
#define VMA_STATIC_VULKAN_FUNCTIONS 0
#define VMA_DYNAMIC_VULKAN_FUNCTIONS 1
#include "third_party/vk_mem_alloc.h"

#pragma clang diagnostic pop

// binary resources
#include "resources/vertex.h"
#include "resources/fragment.h"
#include "resources/bricks.h"

#define LOG_ERROR(fmt, ...) fprintf(stderr, "[error] at %s line %d " fmt "\n", __FILE_NAME__, __LINE__, ##__VA_ARGS__)
#define LOG_INFO(fmt, ...) fprintf(stderr, "[info] at %s line %d " fmt "\n", __FILE_NAME__, __LINE__, ##__VA_ARGS__)

struct Buffer final {
private:
    VmaAllocator allocator_;
    VmaAllocation allocation_;
    VmaAllocationInfo alloc_info_;
    VkBuffer buffer_;

public:
    VmaAllocator allocator() const { return allocator_; }
    VmaAllocation allocation() const { return allocation_; }
    VkBuffer buffer() const { return buffer_; }
    const VkBuffer *addr_of() const { return &buffer_; }

    const VmaAllocationInfo &alloc_info() { return alloc_info_; }

    Buffer() : allocator_{VMA_NULL}, allocation_{VMA_NULL}, buffer_{VK_NULL_HANDLE} {};
    Buffer(VmaAllocator allocator, VkBuffer buffer, VmaAllocation allocation, const VmaAllocationInfo &alloc_info)
        : allocator_{allocator}, allocation_{allocation}, alloc_info_{alloc_info}, buffer_{buffer} {}

    Buffer(const Buffer &) = delete;
    Buffer &operator=(const Buffer &) = delete;

    Buffer(Buffer &&b) {
        allocator_ = b.allocator_;
        allocation_ = b.allocation_;
        alloc_info_ = b.alloc_info_;
        buffer_ = b.buffer_;

        b.allocator_ = VMA_NULL;
        b.allocation_ = VMA_NULL;
        b.buffer_ = VK_NULL_HANDLE;
    }

    Buffer &operator=(Buffer &&b) {
        if (this != &b) {
            destroy();
            allocator_ = b.allocator_;
            allocation_ = b.allocation_;
            alloc_info_ = b.alloc_info_;
            buffer_ = b.buffer_;

            b.allocator_ = VMA_NULL;
            b.allocation_ = VMA_NULL;
            b.buffer_ = VK_NULL_HANDLE;
        }

        return *this;
    }

    bool flush(VkDeviceSize offset = 0, VkDeviceSize size = VK_WHOLE_SIZE) const {
        VkResult res = vmaFlushAllocation(allocator_, allocation_, offset, size);
        if (VK_SUCCESS != res) {
            LOG_ERROR("VMA flush allocation failed: %s", string_VkResult(res));
            return false;
        }

        return true;
    }

    void destroy() {
        if (allocator_ != VMA_NULL && buffer_ != VK_NULL_HANDLE) {
            vmaDestroyBuffer(allocator_, buffer_, allocation_);
        }

        allocator_ = VMA_NULL;
        allocation_ = VMA_NULL;
        buffer_ = VK_NULL_HANDLE;
    }

    VkMemoryPropertyFlags mem_prop_flags() const {
        if (allocator_ == VMA_NULL && buffer_ == VK_NULL_HANDLE) {
            return 0;
        }

        VkMemoryPropertyFlags props;
        vmaGetAllocationMemoryProperties(allocator_, allocation_, &props);

        return props;
    }

    ~Buffer() { destroy(); }
};

struct Image final {
private:
    VmaAllocator allocator_;
    VmaAllocation allocation_;
    VmaAllocationInfo alloc_info_;
    VkImage image_;

public:
    struct View final {
    private:
        vkb::DispatchTable *dispatch_;
        VkImageView view_;

        View(vkb::DispatchTable &dispatch, VkImageView view) : dispatch_{&dispatch}, view_{view} {}
        friend struct Image;

        void destroy() {
            if (dispatch_ && view_ != VK_NULL_HANDLE) {
                dispatch_->destroyImageView(view_, nullptr);
            }
        }

    public:
        ~View() { destroy(); }
        VkImageView view() const { return view_; }

        View(const View &) = delete;
        View &operator=(const View &) = delete;

        View(View &&v) {
            view_ = v.view_;
            dispatch_ = v.dispatch_;

            v.view_ = VK_NULL_HANDLE;
            v.dispatch_ = nullptr;
        }

        View &operator=(View &&v) {
            if (this != &v) {
                destroy();
                view_ = v.view_;
                dispatch_ = v.dispatch_;

                v.view_ = VK_NULL_HANDLE;
                v.dispatch_ = nullptr;
            }

            return *this;
        }
    };

    VmaAllocator allocator() const { return allocator_; }
    VmaAllocation allocation() const { return allocation_; }
    VkImage image() const { return image_; }
    const VkImage *addr_of() const { return &image_; }

    const VmaAllocationInfo &alloc_info() { return alloc_info_; }

    Image() : allocator_{VMA_NULL}, allocation_{VMA_NULL}, image_{VK_NULL_HANDLE} {};
    Image(VmaAllocator allocator, VkImage image, VmaAllocation allocation, const VmaAllocationInfo &alloc_info)
        : allocator_{allocator}, allocation_{allocation}, alloc_info_{alloc_info}, image_{image} {}

    Image(const Image &) = delete;
    Image &operator=(const Image &) = delete;

    Image(Image &&i) {
        allocator_ = i.allocator_;
        allocation_ = i.allocation_;
        alloc_info_ = i.alloc_info_;
        image_ = i.image_;

        i.allocator_ = VMA_NULL;
        i.allocation_ = VMA_NULL;
        i.image_ = VK_NULL_HANDLE;
    }

    Image &operator=(Image &&i) {
        if (this != &i) {
            destroy();
            allocator_ = i.allocator_;
            allocation_ = i.allocation_;
            alloc_info_ = i.alloc_info_;
            image_ = i.image_;

            i.allocator_ = VMA_NULL;
            i.allocation_ = VMA_NULL;
            i.image_ = VK_NULL_HANDLE;
        }

        return *this;
    }

    std::optional<View> create_view(
        vkb::DispatchTable &dispatch, VkImageViewType type, VkFormat format, VkImageAspectFlags aspect_flags) {
        VkImageViewCreateInfo view_desc = {};
        view_desc.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        view_desc.viewType = type;
        view_desc.image = image_;
        view_desc.format = format;
        view_desc.subresourceRange.baseMipLevel = 0;
        view_desc.subresourceRange.levelCount = 1;
        view_desc.subresourceRange.baseArrayLayer = 0;
        view_desc.subresourceRange.layerCount = 1;
        view_desc.subresourceRange.aspectMask = aspect_flags;

        VkImageView image_view = VK_NULL_HANDLE;
        VkResult res = dispatch.createImageView(&view_desc, nullptr, &image_view);
        if (VK_SUCCESS != res) {
            LOG_ERROR("failed to create image view: %s", string_VkResult(res));
            return {};
        }

        return View{dispatch, image_view};
    }

    void destroy() {
        if (allocator_ != VMA_NULL && image_ != VK_NULL_HANDLE) {
            vmaDestroyImage(allocator_, image_, allocation_);
        }

        allocator_ = VMA_NULL;
        allocation_ = VMA_NULL;
        image_ = VK_NULL_HANDLE;
    }

    bool flush(VkDeviceSize offset = 0, VkDeviceSize size = VK_WHOLE_SIZE) const {
        VkResult res = vmaFlushAllocation(allocator_, allocation_, offset, size);
        if (VK_SUCCESS != res) {
            LOG_ERROR("VMA flush allocation failed: %s", string_VkResult(res));
            return false;
        }

        return true;
    }

    VkMemoryPropertyFlags mem_prop_flags() const {
        if (allocator_ == VMA_NULL && image_ == VK_NULL_HANDLE) {
            return 0;
        }

        VkMemoryPropertyFlags props;
        vmaGetAllocationMemoryProperties(allocator_, allocation_, &props);

        return props;
    }

    ~Image() { destroy(); }
};

struct ProgramState final {
private:
    vkb::Instance instance_;
    vkb::InstanceDispatchTable instance_dispatch_;
    VkSurfaceKHR surface_;
    vkb::PhysicalDevice phys_dev_;
    vkb::Device device_;
    vkb::DispatchTable dispatch_;
    vkb::Swapchain swapchain_;
    VkPhysicalDeviceProperties phys_dev_props_;

    // queues
    VkQueue graphics_queue_, present_queue_;

    // memory allocation
    VmaVulkanFunctions allocator_fns_;
    VmaAllocator allocator_;

    ProgramState()
        : surface_{VK_NULL_HANDLE}, allocator_{VMA_NULL}, graphics_queue_{VK_NULL_HANDLE},
          present_queue_{VK_NULL_HANDLE} {};
    ProgramState(const ProgramState &) = delete;
    ProgramState &operator=(const ProgramState) = delete;

public:
    VkQueue graphics_queue() const { return graphics_queue_; }
    VkQueue present_queue() const { return present_queue_; }

    VkSurfaceKHR surface() const { return surface_; }
    VmaAllocator allocator() const { return allocator_; }

    const vkb::Instance &instance() const { return instance_; }
    const vkb::InstanceDispatchTable &instance_dispatch() const { return instance_dispatch_; }
    const vkb::PhysicalDevice &phys_dev() const { return phys_dev_; }
    const vkb::Device &device() const { return device_; }
    const vkb::DispatchTable &dispatch() const { return dispatch_; }
    const vkb::Swapchain &swapchain() const { return swapchain_; }

    vkb::Instance &instance() { return instance_; }
    vkb::InstanceDispatchTable &instance_dispatch() { return instance_dispatch_; }
    vkb::PhysicalDevice &phys_dev() { return phys_dev_; }
    vkb::Device &device() { return device_; }
    vkb::DispatchTable &dispatch() { return dispatch_; }
    vkb::Swapchain &swapchain() { return swapchain_; }

    VkDeviceSize ubo_alignment() const { return phys_dev_props_.limits.minUniformBufferOffsetAlignment; }

    ~ProgramState() {
        LOG_INFO("freeing program state");

        vmaDestroyAllocator(allocator_);
        vkb::destroy_swapchain(swapchain_);
        vkb::destroy_device(device_);
        vkb::destroy_surface(instance_, surface_);
        vkb::destroy_instance(instance_);
    }

    bool init_swapchain() {
        vkb::SwapchainBuilder builder{device_};
        builder.set_old_swapchain(swapchain_);

        auto ret = builder.build();
        if (!ret) {
            LOG_ERROR("failed to create swap chain: %s", ret.error().message().c_str());
            return false;
        }

        vkb::destroy_swapchain(swapchain_);
        swapchain_ = ret.value();

        return true;
    }

    static VkSurfaceKHR make_surface_glfw(VkInstance instance, GLFWwindow *window) {
        VkSurfaceKHR surface;
        VkResult res = glfwCreateWindowSurface(instance, window, nullptr, &surface);

        if (VK_SUCCESS != res) {
            const char *error = nullptr;
            glfwGetError(&error);

            if (error) {
                LOG_ERROR("failed to create surface: %s", error);
            } else {
                LOG_ERROR("failed to create surface");
            }

            return VK_NULL_HANDLE;
        }

        return surface;
    }

    static std::unique_ptr<ProgramState> initialize(GLFWwindow *window) {
        std::unique_ptr<ProgramState> state{new ProgramState()};

        auto system_info_ret = vkb::SystemInfo::get_system_info();
        if (!system_info_ret) {
            LOG_ERROR("cannot fetch system info: %s", system_info_ret.error().message().c_str());
            return {};
        }

        auto system_info = system_info_ret.value();
        if (!system_info.is_extension_available("VK_KHR_display")) {
            LOG_ERROR("VK_KHR_display is not available");
            return {};
        }

        vkb::InstanceBuilder instance_builder;
        auto instance_ret = instance_builder.set_app_name("vulkan sample")
                                .request_validation_layers(true)
                                .set_engine_name("no engine")
                                .set_app_version(1, 0, 0)
                                .set_engine_version(1, 0, 0)
                                .enable_extension("VK_KHR_display")
                                .use_default_debug_messenger()
                                .set_debug_callback([](VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
                                                        VkDebugUtilsMessageTypeFlagsEXT messageType,
                                                        const VkDebugUtilsMessengerCallbackDataEXT *pCallbackData,
                                                        void *pUserData) -> VkBool32 {
            auto severity = vkb::to_string_message_severity(messageSeverity);
            auto type = vkb::to_string_message_type(messageType);

            LOG_ERROR("[%s: %s] %s\n", severity, type, pCallbackData->pMessage);
            return VK_FALSE;
        }).build();

        if (!instance_ret) {
            LOG_ERROR("failed to create instance: %s", instance_ret.error().message().c_str());
            return {};
        }

        state->instance_ = instance_ret.value();
        state->instance_dispatch_ = state->instance_.make_table();

        // create surface from window
        state->surface_ = make_surface_glfw(state->instance_, window);

        vkb::PhysicalDeviceSelector phys_dev_selector{state->instance_};
        auto devices_ret = phys_dev_selector.set_surface(state->surface_).select_devices();

        if (!devices_ret) {
            LOG_ERROR("device enumeration failed: %s", devices_ret.error().message().c_str());
            return {};
        }

        for (auto &device : devices_ret.value()) {
            LOG_INFO("detected vk device: %s", device.name.c_str());
        }

        state->phys_dev_ = devices_ret.value().front();
        LOG_INFO("selected vk device: %s", state->phys_dev_.name.c_str());

        vkb::DeviceBuilder device_builder{state->phys_dev_};
        auto device_ret = device_builder.build();

        if (!device_ret) {
            LOG_ERROR("failed to create device: %s", device_ret.error().message().c_str());
            return {};
        }

        state->device_ = device_ret.value();
        state->dispatch_ = state->device_.make_table();
        state->phys_dev_props_ = state->phys_dev_.properties;

        LOG_INFO("created vk device successfully");

        if (!state->init_swapchain()) {
            LOG_ERROR("failed to initialize swapchain");
            return {};
        }

        // fetch queues
        auto gq = state->device().get_queue(vkb::QueueType::graphics);
        auto pq = state->device().get_queue(vkb::QueueType::present);

        if (!gq.has_value()) {
            LOG_ERROR("no graphics queue: %s", gq.error().message().c_str());
            return {};
        }

        if (!pq.has_value()) {
            LOG_ERROR("no present queue: %s", pq.error().message().c_str());
            return {};
        }

        state->graphics_queue_ = gq.value();
        state->present_queue_ = pq.value();
        LOG_INFO("obtained graphics and present queue");

        // init vma
        state->allocator_fns_ = {};
        state->allocator_fns_.vkGetInstanceProcAddr = state->instance_.fp_vkGetInstanceProcAddr;
        state->allocator_fns_.vkGetDeviceProcAddr = state->instance_.fp_vkGetDeviceProcAddr;

        VmaAllocatorCreateInfo alloc_create_info = {};
        alloc_create_info.flags = 0;
        alloc_create_info.vulkanApiVersion = VK_API_VERSION_1_0;
        alloc_create_info.physicalDevice = state->phys_dev_;
        alloc_create_info.instance = state->instance_;
        alloc_create_info.device = state->device_;
        alloc_create_info.pVulkanFunctions = &state->allocator_fns_;

        VkResult res = vmaCreateAllocator(&alloc_create_info, &state->allocator_);
        if (VK_SUCCESS != res) {
            LOG_ERROR("failed to create allocator: %s", string_VkResult(res));
            return {};
        }

        LOG_INFO("created vk allocator successfully");
        return state;
    }
};

constexpr uint32_t kFramesInFlight = 2;

struct Vertex {
    glm::fvec3 position;
    glm::fvec3 normal;
    glm::fvec2 uv;
};

struct Geometry {
    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices;
};

struct Bitmap final {
private:
    uint32_t width_;
    uint32_t height_;
    std::vector<uint8_t> pixels_;

public:
    uint32_t width() const { return width_; }
    uint32_t height() const { return height_; }

    const std::vector<uint8_t> &pixels() const { return pixels_; }
    const uint8_t *raw_pixels() const { return pixels_.data(); }
    uint32_t size() const { return pixels_.size(); }

    uint8_t *raw_pixels() { return pixels_.data(); }

    Bitmap(uint32_t width, uint32_t height) : width_{width}, height_{height} { pixels_.resize(width_ * height_ * 4); }
    Bitmap(const Bitmap &) = default;
    Bitmap &operator=(const Bitmap &) = default;
};

struct cbPerFrame {
    glm::fmat4 view;
    glm::fmat4 proj;
};

struct cbPerObject {
    glm::fmat4 world;
};

struct MemoryHelper final {
private:
    ProgramState &state_;

    VkFence fence_complete_;
    VkCommandPool command_pool_;
    VkCommandBuffer command_buffer_;

    MemoryHelper(ProgramState &state) : state_{state} {}

public:
    ~MemoryHelper() {
        state_.dispatch().destroyFence(fence_complete_, nullptr);
        state_.dispatch().destroyCommandPool(command_pool_, nullptr);
    }

    MemoryHelper(const MemoryHelper &) = delete;
    MemoryHelper &operator=(const MemoryHelper &) = delete;

    std::optional<Image> create_image(
        VkFormat format, VkImageUsageFlags usage, VkImageType type, const VkExtent3D &extent) {
        VkResult res;
        VkImageCreateInfo create_info = {};

        create_info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        create_info.imageType = type;
        create_info.format = format;
        create_info.extent = extent;
        create_info.mipLevels = 1;
        create_info.arrayLayers = 1;
        create_info.samples = VK_SAMPLE_COUNT_1_BIT;
        create_info.tiling = VK_IMAGE_TILING_OPTIMAL;
        create_info.usage = usage;

        VmaAllocationCreateInfo alloc_desc = {};
        alloc_desc.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
        alloc_desc.flags = 0;
        alloc_desc.requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

        VkImage vk_image = VK_NULL_HANDLE;
        VmaAllocation allocation = VMA_NULL;
        VmaAllocationInfo alloc_info = {};

        res = vmaCreateImage(state_.allocator(), &create_info, &alloc_desc, &vk_image, &allocation, &alloc_info);
        if (VK_SUCCESS != res) {
            LOG_ERROR("cannot create image: %s", string_VkResult(res));
            return {};
        }

        return Image{state_.allocator(), vk_image, allocation, alloc_info};
    }

    std::optional<Image> create_image_rgba(
        VkImageUsageFlags usage, uint32_t width, uint32_t height, const void *pixels) const {
        VkResult res;
        VkDeviceSize image_size = width * height * 4;

        // create staging buffer for transfer
        auto staging_buffer = create_staging_buffer(image_size);
        if (!staging_buffer) {
            LOG_ERROR("failed to allocate staging buffer for transfer");
            return {};
        }

        // map the staging buffer and upload the raw pixels
        void *mapped_mem;
        res = vmaMapMemory(state_.allocator(), staging_buffer->allocation(), &mapped_mem);
        if (VK_SUCCESS != res) {
            LOG_ERROR("cannot map staging buffer: %s", string_VkResult(res));
            return {};
        }

        memcpy(mapped_mem, pixels, image_size);
        if (!staging_buffer->flush()) {
            LOG_ERROR("cannot flush staging buffer");
            return {};
        }

        VkImageCreateInfo create_info = {};
        create_info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        create_info.imageType = VK_IMAGE_TYPE_2D;
        create_info.format = VK_FORMAT_R8G8B8A8_SRGB;
        create_info.extent = VkExtent3D{width, height, 1};
        create_info.mipLevels = 1;
        create_info.arrayLayers = 1;
        create_info.samples = VK_SAMPLE_COUNT_1_BIT;
        create_info.tiling = VK_IMAGE_TILING_OPTIMAL;
        create_info.usage = usage | VK_IMAGE_USAGE_TRANSFER_DST_BIT;

        VmaAllocationCreateInfo alloc_desc = {};
        alloc_desc.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
        alloc_desc.flags = 0;
        alloc_desc.requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

        VkImage vk_image = VK_NULL_HANDLE;
        VmaAllocation allocation = VMA_NULL;
        VmaAllocationInfo alloc_info = {};

        res = vmaCreateImage(state_.allocator(), &create_info, &alloc_desc, &vk_image, &allocation, &alloc_info);
        if (VK_SUCCESS != res) {
            LOG_ERROR("cannot create image: %s", string_VkResult(res));
            return {};
        }

        Image image{state_.allocator(), vk_image, allocation, alloc_info};

        if (!run_on_transfer_queue([&](VkCommandBuffer command_buffer) {
            VkImageSubresourceRange range;
            range.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            range.baseMipLevel = 0;
            range.baseArrayLayer = 0;
            range.levelCount = 1;
            range.layerCount = 1;

            // move the image to layout optimal for transfer
            VkImageMemoryBarrier transfer_barrier = {};
            transfer_barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            transfer_barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            transfer_barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            transfer_barrier.image = image.image();
            transfer_barrier.subresourceRange = range;
            transfer_barrier.srcAccessMask = 0;
            transfer_barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

            state_.dispatch().cmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &transfer_barrier);

            // execute transfer from staging buffer to image
            VkBufferImageCopy image_copy = {};
            image_copy.bufferOffset = 0;
            image_copy.bufferRowLength = 0;
            image_copy.bufferImageHeight = 0;
            image_copy.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            image_copy.imageSubresource.mipLevel = 0;
            image_copy.imageSubresource.baseArrayLayer = 0;
            image_copy.imageSubresource.layerCount = 1;
            image_copy.imageExtent = VkExtent3D{width, height, 1};

            state_.dispatch().cmdCopyBufferToImage(command_buffer, staging_buffer->buffer(), image.image(),
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &image_copy);

            // move the image to layout optimal for gpu rendering
            VkImageMemoryBarrier optimal_barrier = {};
            optimal_barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            optimal_barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            optimal_barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            optimal_barrier.image = image.image();
            optimal_barrier.subresourceRange = range;

            // has to complete before its used in fragment shader
            state_.dispatch().cmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1, &optimal_barrier);
        })) {
            LOG_ERROR("failed to upload image data to the gpu");
            return {};
        }

        // remember to unmap before deallocation
        vmaUnmapMemory(state_.allocator(), staging_buffer->allocation());
        return image;
    }

    std::optional<Buffer> create_shared_buffer(const VkBufferUsageFlags usage, size_t byte_size) const {
        VkResult res;

        VkBufferCreateInfo create_info = {};
        create_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        create_info.size = byte_size;
        create_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        create_info.usage = usage;

        VmaAllocationCreateInfo alloc_create_info = {};
        alloc_create_info.usage = VMA_MEMORY_USAGE_AUTO;
        alloc_create_info.flags =
            VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;

        VkBuffer vk_buffer = VK_NULL_HANDLE;
        VmaAllocation allocation = VMA_NULL;
        VmaAllocationInfo alloc_info = {};

        res =
            vmaCreateBuffer(state_.allocator(), &create_info, &alloc_create_info, &vk_buffer, &allocation, &alloc_info);
        if (VK_SUCCESS != res) {
            LOG_ERROR("cannot create buffer: %s", string_VkResult(res));
            return {};
        }

        return Buffer{state_.allocator(), vk_buffer, allocation, alloc_info};
    }

    std::optional<Buffer> create_buffer(
        const VkBufferUsageFlags usage, const void *data, size_t byte_size, bool use_staging) const {
        VkResult res;

        VkBufferCreateInfo create_info = {};
        create_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        create_info.size = byte_size;
        create_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        if (use_staging) {
            create_info.usage = usage | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        } else {
            create_info.usage = usage;
        }

        // vma allocation info
        VmaAllocationCreateInfo alloc_create_info = {};
        alloc_create_info.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;

        if (use_staging) {
            // if we use staging buffer then we can have no sequential write since we will do copy using staging
            // this allows for usage of gram etc.
            alloc_create_info.flags = 0;
            alloc_create_info.requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
        } else {
            // if we don't use the staging buffer, then we need to have sequential access
            alloc_create_info.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;
        }

        VkBuffer vk_buffer = VK_NULL_HANDLE;
        VmaAllocation allocation = VMA_NULL;
        VmaAllocationInfo alloc_info = {};

        res =
            vmaCreateBuffer(state_.allocator(), &create_info, &alloc_create_info, &vk_buffer, &allocation, &alloc_info);
        if (VK_SUCCESS != res) {
            LOG_ERROR("cannot create buffer: %s", string_VkResult(res));
            return {};
        }

        Buffer buffer{state_.allocator(), vk_buffer, allocation, alloc_info};

        // check if can be mapped on host, not always use_staging = cannot be mapped
        auto mem_prop_flags = buffer.mem_prop_flags();

        // data upload
        if (mem_prop_flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) {
            // memory is host visible, we can memcpy into it

            void *mapped_mem;
            res = vmaMapMemory(state_.allocator(), buffer.allocation(), &mapped_mem);
            if (VK_SUCCESS != res) {
                LOG_ERROR("cannot map buffer: %s", string_VkResult(res));
                return {};
            }

            memcpy(mapped_mem, data, byte_size);
            if (!buffer.flush()) {
                LOG_ERROR("cannot flush buffer write");
                return {};
            }

            vmaUnmapMemory(state_.allocator(), buffer.allocation());

            res = vmaFlushAllocation(state_.allocator(), buffer.allocation(), 0, VK_WHOLE_SIZE);
            if (VK_SUCCESS != res) {
                LOG_ERROR("vmaFlushAllocation failure: %s", string_VkResult(res));
                return {};
            }
        } else {
            // prepare staging buffer for upload
            auto staging_buffer = create_staging_buffer(byte_size);
            if (!staging_buffer) {
                LOG_ERROR("failed to allocate staging buffer for transfer");
                return {};
            }

            void *mapped_mem;
            res = vmaMapMemory(state_.allocator(), staging_buffer->allocation(), &mapped_mem);
            if (VK_SUCCESS != res) {
                LOG_ERROR("cannot map staging buffer: %s", string_VkResult(res));
                return {};
            }

            memcpy(mapped_mem, data, byte_size);
            if (!staging_buffer->flush()) {
                LOG_ERROR("cannot flush staging buffer write");
                return {};
            }

            // transfer command
            res = vmaFlushAllocation(state_.allocator(), buffer.allocation(), 0, VK_WHOLE_SIZE);
            if (VK_SUCCESS != res) {
                LOG_ERROR("vmaFlushAllocation failure: %s", string_VkResult(res));
                return {};
            }

            copy_buffer(staging_buffer->buffer(), buffer.buffer(), byte_size);
            vmaUnmapMemory(state_.allocator(), staging_buffer->allocation());
        }

        return std::move(buffer);
    }

    template <typename F> bool run_on_transfer_queue(F runner) const {
        VkResult res;

        VkCommandBufferBeginInfo begin_info = {};
        begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        begin_info.pInheritanceInfo = 0;
        begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

        res = state_.dispatch().beginCommandBuffer(command_buffer_, &begin_info);
        if (VK_SUCCESS != res) {
            LOG_ERROR("begin upload command buffer failure: %s", string_VkResult(res));
            return false;
        }

        runner(command_buffer_);

        res = state_.dispatch().endCommandBuffer(command_buffer_);
        if (VK_SUCCESS != res) {
            LOG_ERROR("end upload command buffer failure: %s", string_VkResult(res));
            return false;
        }

        VkSubmitInfo submit_info = {};
        submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit_info.pCommandBuffers = &command_buffer_;
        submit_info.commandBufferCount = 1;

        res = state_.dispatch().queueSubmit(state_.graphics_queue(), 1, &submit_info, fence_complete_);
        if (VK_SUCCESS != res) {
            LOG_ERROR("submit upload command buffer failure: %s", string_VkResult(res));
            return false;
        }

        res = state_.dispatch().waitForFences(1, &fence_complete_, true, UINT64_MAX);
        if (VK_SUCCESS != res) {
            LOG_ERROR("wait complete fence failed: %s", string_VkResult(res));
            return false;
        }

        res = state_.dispatch().resetFences(1, &fence_complete_);
        if (VK_SUCCESS != res) {
            LOG_ERROR("reset complete fence failed: %s", string_VkResult(res));
            return false;
        }

        res = state_.dispatch().resetCommandPool(command_pool_, 0);
        if (VK_SUCCESS != res) {
            LOG_ERROR("reset upload command pool failed: %s", string_VkResult(res));
            return false;
        }

        return true;
    }

    bool copy_buffer(VkBuffer src, VkBuffer dst, VkDeviceSize size) const {
        return run_on_transfer_queue([&](VkCommandBuffer command_buffer) {
            VkBufferCopy copy_info = {};
            copy_info.srcOffset = 0;
            copy_info.dstOffset = 0;
            copy_info.size = size;

            state_.dispatch().cmdCopyBuffer(command_buffer, src, dst, 1, &copy_info);
        });
    }

    std::optional<Buffer> create_staging_buffer(VkDeviceSize size) const {
        // initialize staging buffer
        VkBufferCreateInfo staging_buffer_desc = {};
        staging_buffer_desc.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        staging_buffer_desc.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        staging_buffer_desc.size = size;
        staging_buffer_desc.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        VmaAllocationCreateInfo staging_alloc_desc = {};
        staging_alloc_desc.usage = VMA_MEMORY_USAGE_AUTO;
        staging_alloc_desc.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;

        VmaAllocation allocation = VMA_NULL;
        VkBuffer vk_buffer = VK_NULL_HANDLE;
        VmaAllocationInfo alloc_info = {};

        VkResult res = vmaCreateBuffer(
            state_.allocator(), &staging_buffer_desc, &staging_alloc_desc, &vk_buffer, &allocation, &alloc_info);
        if (res != VK_SUCCESS) {
            LOG_ERROR("failed to allocate staging buffer: %s", string_VkResult(res));
            return {};
        }

        return Buffer{state_.allocator(), vk_buffer, allocation, alloc_info};
    }

    static std::unique_ptr<MemoryHelper> initialize(ProgramState &state) {
        std::unique_ptr<MemoryHelper> memory{new MemoryHelper(state)};
        VkResult res;

        // initialize fence for completion
        VkFenceCreateInfo fence_desc = {};
        fence_desc.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;

        res = state.dispatch().createFence(&fence_desc, nullptr, &memory->fence_complete_);
        if (VK_SUCCESS != res) {
            LOG_ERROR("failed to create fence: %s", string_VkResult(res));
            return {};
        }

        // initialize command pool for the memory helper
        VkCommandPoolCreateInfo pool_desc = {};
        pool_desc.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        pool_desc.flags = 0;
        pool_desc.queueFamilyIndex = state.device().get_queue_index(vkb::QueueType::graphics).value();
        res = state.dispatch().createCommandPool(&pool_desc, nullptr, &memory->command_pool_);
        if (VK_SUCCESS != res) {
            LOG_ERROR("failed to create upload command pool: %s", string_VkResult(res));
            return {};
        }

        // since we have the staging buffer, we would also like to get the upload command buffer
        VkCommandBufferAllocateInfo cmd_alloc_info = {};
        cmd_alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        cmd_alloc_info.commandPool = memory->command_pool_;
        cmd_alloc_info.commandBufferCount = 1;
        cmd_alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;

        res = state.dispatch().allocateCommandBuffers(&cmd_alloc_info, &memory->command_buffer_);
        if (VK_SUCCESS != res) {
            LOG_ERROR("failed to create upload command buffer: %s", string_VkResult(res));
            return {};
        }

        return memory;
    }

    // dynamic ubo helper
    template <typename T> class DynamicUniformBuffer final {
    private:
        Buffer buffer_;

        VkDeviceSize aligned_size_;
        VkDeviceSize num_elements_;

        DynamicUniformBuffer(Buffer &&buffer, VkDeviceSize aligned_size, VkDeviceSize num_elements)
            : buffer_{std::move(buffer)}, aligned_size_{aligned_size}, num_elements_{num_elements} {}

        friend struct MemoryHelper;

    public:
        Buffer &buffer() { return buffer_; }
        const Buffer &buffer() const { return buffer_; }

        const VkDeviceSize num_elements() const { return num_elements_; }
        const VkDeviceSize aligned_size() const { return aligned_size_; }
        constexpr VkDeviceSize element_size() const { return sizeof(T); }

        ~DynamicUniformBuffer() = default;
        DynamicUniformBuffer(const DynamicUniformBuffer &) = delete;
        DynamicUniformBuffer &operator=(const DynamicUniformBuffer &) = delete;

        DynamicUniformBuffer(DynamicUniformBuffer &&b) {
            buffer_ = std::move(b.buffer_);
            aligned_size_ = b.aligned_size_;
            num_elements_ = b.num_elements_;

            b.aligned_size_ = 0;
            b.num_elements_ = 0;
        }

        DynamicUniformBuffer &operator=(DynamicUniformBuffer &&b) {
            if (this != &b) {
                buffer_ = std::move(b.buffer_);
                aligned_size_ = b.aligned_size_;
                num_elements_ = b.num_elements_;

                b.aligned_size_ = 0;
                b.num_elements_ = 0;
            }

            return *this;
        }

        size_t slot_offset(size_t slot) const { return slot * aligned_size_; }

        bool write_slot(size_t slot, const T &data, bool flush) {
            if (slot >= num_elements_) {
                return false;
            }

            VkDeviceSize offset = slot * aligned_size_;
            uint8_t *buffer_ptr = reinterpret_cast<uint8_t *>(buffer_.alloc_info().pMappedData);
            memcpy(buffer_ptr + offset, &data, element_size());

            if (flush) {
                if (!buffer_.flush(offset, aligned_size_)) {
                    LOG_ERROR("failed to flush dynamic ubo write");
                    return false;
                }
            }

            return true;
        }

        bool make_descriptor_info(VkDescriptorBufferInfo &info, size_t slot) const {
            if (slot >= num_elements_) {
                return false;
            }

            info = {};
            info.buffer = buffer_.buffer();
            info.offset = aligned_size_ * slot;
            info.range = element_size();
        }
    };

    template <typename T> std::optional<DynamicUniformBuffer<T>> init_dynamic_ubo(VkDeviceSize num_elements) {
        auto min_ubo_align = state_.ubo_alignment();
        auto cpu_size = sizeof(T);

        VkDeviceSize aligned_size =
            min_ubo_align > 0 ? (cpu_size + min_ubo_align - 1) & ~(min_ubo_align - 1) : cpu_size;

        VkDeviceSize buffer_size = aligned_size * num_elements;
        auto buffer = create_shared_buffer(VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, buffer_size);
        if (!buffer) {
            LOG_ERROR("failed to create backing buffer for dynamic ubo");
            return {};
        }

        return DynamicUniformBuffer<T>(std::move(buffer.value()), aligned_size, num_elements);
    }
};

struct SceneState final {
public:
    static constexpr size_t kMaxStaticMeshes = 128;
    static constexpr size_t kMaxObjects = 1024;
    static constexpr size_t kMaxMaterials = 256;

    template <typename T> struct Identifier {
    private:
        static constexpr uint32_t kInvalidId = UINT32_MAX;

        uint32_t id_;
        Identifier(uint32_t id) : id_{id} {}

        friend class SceneState;

    public:
        Identifier() : id_{kInvalidId} {}
        Identifier(const Identifier &) = default;
        Identifier &operator=(const Identifier &) = default;
        ~Identifier() = default;

        Identifier(Identifier &&i) {
            id_ = i.id_;
            i.id_ = kInvalidId;
        }

        Identifier &operator=(Identifier &&i) {
            if (this != &i) {
                id_ = i.id_;
                i.id_ = kInvalidId;
            }

            return *this;
        }

        bool valid() const { return id_ != kInvalidId; }

        operator bool() const { return valid(); }
        bool operator<(const Identifier<T> &other) const { return id_ < other.id_; }
        bool operator>(const Identifier<T> &other) const { return id_ > other.id_; }
        bool operator==(const Identifier<T> &other) const { return id_ == other.id_; }
    };

    struct Material final {
    public:
        using Id = Identifier<Material>;

    private:
        ProgramState *state_;
        Id id_;
        Image image_;
        Image::View image_view_;
        VkSampler sampler_;
        VkDescriptorSet descriptor_set_;

        Material(ProgramState &state, const Id &id, Image &&image, Image::View &&image_view, VkSampler sampler,
            VkDescriptorSet descriptor_set)
            : state_{&state}, id_{id}, image_{std::move(image)}, image_view_{std::move(image_view)}, sampler_{sampler},
              descriptor_set_{descriptor_set} {}

        friend struct SceneState;

    public:
        const Id &id() const { return id_; }
        Image &image() { return image_; }
        Image::View &image_view() { return image_view_; }
        VkSampler sampler() { return sampler_; }
        VkDescriptorSet descriptor_set() { return descriptor_set_; }
        const VkDescriptorSet *descriptor_set_addr() const { return &descriptor_set_; }

        ~Material() {
            if (state_ && sampler_ != VK_NULL_HANDLE) {
                state_->dispatch().destroySampler(sampler_, nullptr);
            }

            state_ = nullptr;
            sampler_ = VK_NULL_HANDLE;
        }

        Material(const Material &) = delete;
        Material &operator=(const Material &) = delete;

        Material(Material &&m) : image_view_{std::move(m.image_view_)} {
            state_ = m.state_;
            id_ = std::move(m.id_);
            image_ = std::move(m.image_);
            sampler_ = m.sampler_;
            descriptor_set_ = m.descriptor_set_;

            m.state_ = nullptr;
            m.sampler_ = VK_NULL_HANDLE;
            m.descriptor_set_ = VK_NULL_HANDLE;
        }

        Material &operator=(Material &&m) {
            if (this != &m) {
                state_ = m.state_;
                id_ = std::move(m.id_);
                image_ = std::move(m.image_);
                descriptor_set_ = m.descriptor_set_;
                descriptor_set_ = m.descriptor_set_;
                image_view_ = std::move(m.image_view_);

                m.state_ = nullptr;
                m.sampler_ = VK_NULL_HANDLE;
                m.descriptor_set_ = VK_NULL_HANDLE;
            }

            return *this;
        }
    };

    struct StaticMesh final {
    public:
        using Id = Identifier<StaticMesh>;

    private:
        Id id_;
        Buffer vertex_buffer_;
        Buffer index_buffer_;
        uint32_t num_vertices_;
        uint32_t num_indices_;

        StaticMesh(
            const Id &id, Buffer &&vertex_buffer, Buffer &&index_buffer, uint32_t num_vertices, uint32_t num_indices)
            : id_{id}, vertex_buffer_{std::move(vertex_buffer)}, index_buffer_{std::move(index_buffer)},
              num_vertices_{num_vertices}, num_indices_{num_indices} {}

        friend struct SceneState;

    public:
        const Id &id() const { return id_; }
        Buffer &vertex_buffer() { return vertex_buffer_; }
        Buffer &index_buffer() { return index_buffer_; }
        uint32_t num_vertices() const { return num_vertices_; }
        uint32_t num_indices() const { return num_indices_; }

        ~StaticMesh() = default;

        StaticMesh(const StaticMesh &) = delete;
        StaticMesh &operator=(const StaticMesh &) = delete;

        StaticMesh(StaticMesh &&m) {
            id_ = std::move(m.id_);
            vertex_buffer_ = std::move(m.vertex_buffer_);
            index_buffer_ = std::move(m.index_buffer_);
            num_vertices_ = m.num_vertices_;
            num_indices_ = m.num_indices_;

            m.num_vertices_ = 0;
            m.num_vertices_ = 0;
        }

        StaticMesh &operator=(StaticMesh &&m) {
            if (this != &m) {
                id_ = std::move(m.id_);
                vertex_buffer_ = std::move(m.vertex_buffer_);
                index_buffer_ = std::move(m.index_buffer_);
                num_vertices_ = m.num_vertices_;
                num_indices_ = m.num_indices_;

                m.num_vertices_ = 0;
                m.num_vertices_ = 0;
            }

            return *this;
        }

        void draw(vkb::DispatchTable &dispatch, VkCommandBuffer command_buffer) {
            VkDeviceSize buf_offset = 0;
            dispatch.cmdBindVertexBuffers(command_buffer, 0, 1, vertex_buffer_.addr_of(), &buf_offset);
            dispatch.cmdBindIndexBuffer(command_buffer, index_buffer_.buffer(), 0, VK_INDEX_TYPE_UINT32);
            dispatch.cmdDrawIndexed(command_buffer, num_indices_, 1, 0, 0, 0);
        }
    };

    struct SceneObject final {
    public:
        using Id = Identifier<SceneObject>;

    private:
        Id id_;
        glm::fvec3 translation_;
        glm::fvec3 scale_;
        glm::fquat rotation_;

        glm::fmat4x4 transform_;
        StaticMesh::Id mesh_id_;
        Material::Id material_id_;

        SceneObject(const Id &id)
            : id_{id}, translation_{0.0f, 0.0f, 0.0f}, scale_{1.0f, 1.0f, 1.0f}, rotation_{0.0f, 0.0f, 0.0f, 1.0f},
              transform_(1.0f), mesh_id_{} {}

        friend struct SceneState;

        void recalculate_transform() {
            transform_ = glm::translate(glm::mat4(1.0f), translation_) * glm::mat4_cast(rotation_) *
                         glm::scale(glm::mat4(1.0f), scale_);
        }

    public:
        SceneObject(const SceneObject &) = delete;
        SceneObject &operator=(const SceneObject &) = delete;

        SceneObject(SceneObject &&o) noexcept
            : id_{std::move(o.id_)}, translation_(std::move(o.translation_)), scale_(std::move(o.scale_)),
              rotation_(std::move(o.rotation_)), transform_(std::move(o.transform_)), mesh_id_(std::move(o.mesh_id_)) {}

        SceneObject &operator=(SceneObject &&o) noexcept {
            if (this != &o) {
                id_ = std::move(o.id_);
                translation_ = std::move(o.translation_);
                scale_ = std::move(o.scale_);
                rotation_ = std::move(o.rotation_);
                transform_ = std::move(o.transform_);
                mesh_id_ = std::move(o.mesh_id_);

                o.translation_ = {0.0f, 0.0f, 0.0f};
                o.scale_ = {1.0f, 1.0f, 1.0f};
                o.rotation_ = {0.0f, 0.0f, 0.0f, 1.0f};
                o.transform_ = glm::mat4(1.0f);
                o.mesh_id_ = {};
            }

            return *this;
        }

        const Id &id() const { return id_; }
        const glm::fvec3 &translation() const { return translation_; }
        const glm::fvec3 &scale() const { return scale_; }
        const glm::fquat &rotation() const { return rotation_; }
        const glm::fmat4x4 &transform() const { return transform_; }
        const StaticMesh::Id &mesh_id() const { return mesh_id_; }
        const Material::Id &material_id() const { return material_id_; }

        void set_translation(const glm::fvec3 &translation) {
            translation_ = translation;
            recalculate_transform();
        }

        void set_scale(const glm::fvec3 &scale) {
            scale_ = scale;
            recalculate_transform();
        }

        void set_rotation(const glm::fquat &rotation) {
            rotation_ = rotation;
            recalculate_transform();
        }

        void set_mesh_id(const StaticMesh::Id &mesh_id) { mesh_id_ = mesh_id; }
        void set_material_id(const Material::Id &material_id) { material_id_ = material_id; }
    };

    enum DescriptorSet { PerFrame, PerMaterial, PerObject, Count };
    struct FrameSubmitData final {
    private:
        ProgramState &state_;
        SceneState &scene_;

        VkCommandBuffer command_buffer_;
        VkSemaphore sem_image_avaliable_, sem_render_done_;
        VkFence fence_in_flight_;

        VkDescriptorSet per_frame_set_;
        Buffer per_frame_buffer_;

        FrameSubmitData(ProgramState &state, SceneState &scene)
            : state_{state}, scene_{scene}, command_buffer_{VK_NULL_HANDLE}, sem_image_avaliable_{VK_NULL_HANDLE},
              sem_render_done_{VK_NULL_HANDLE}, fence_in_flight_{VK_NULL_HANDLE}, per_frame_set_{VK_NULL_HANDLE} {}

    public:
        VkCommandBuffer command_buffer() { return command_buffer_; }
        Buffer &per_frame_buffer() { return per_frame_buffer_; }

        void update_per_frame(const cbPerFrame &data) {
            memcpy(per_frame_buffer_.alloc_info().pMappedData, &data, sizeof(cbPerFrame));

            if (!per_frame_buffer_.flush()) {
                LOG_ERROR("cannot flush per frame uniform buffer");
            }
        }

        FrameSubmitData(const FrameSubmitData &) = delete;
        FrameSubmitData &operator=(const FrameSubmitData &) = delete;

        FrameSubmitData(FrameSubmitData &&f) : state_{f.state_}, scene_{f.scene_} {
            command_buffer_ = f.command_buffer_;
            sem_image_avaliable_ = f.sem_image_avaliable_;
            sem_render_done_ = f.sem_render_done_;
            fence_in_flight_ = f.fence_in_flight_;
            per_frame_set_ = std::move(f.per_frame_set_);
            per_frame_buffer_ = std::move(f.per_frame_buffer_);

            f.command_buffer_ = VK_NULL_HANDLE;
            f.sem_image_avaliable_ = VK_NULL_HANDLE;
            f.sem_render_done_ = VK_NULL_HANDLE;
            f.fence_in_flight_ = VK_NULL_HANDLE;
            f.per_frame_set_ = VK_NULL_HANDLE;
        }

        ~FrameSubmitData() {
            state_.dispatch().destroySemaphore(sem_image_avaliable_, nullptr);
            state_.dispatch().destroySemaphore(sem_render_done_, nullptr);
            state_.dispatch().destroyFence(fence_in_flight_, nullptr);
        }

        friend struct SceneState;
    };

private:
    ProgramState &state_;
    std::unique_ptr<MemoryHelper> memory_;

    VkRenderPass render_pass_;
    VkPipelineLayout pipeline_layout_;
    VkPipeline graphics_pipeline_;
    VkCommandPool command_pool_;

    // object uniforms
    std::optional<MemoryHelper::DynamicUniformBuffer<cbPerObject>> object_uniforms_;

    // descriptor set layouts
    VkDescriptorPool descriptor_pool_;
    std::array<VkDescriptorSetLayout, DescriptorSet::Count> descriptor_layout_;

    VkDescriptorSet per_object_set_;

    // swapchain images
    std::vector<VkImage> swapchain_images_;
    std::vector<VkImageView> swapchain_views_;
    std::vector<VkFramebuffer> swapchain_fbs_;
    std::vector<FrameSubmitData> frame_data_;

    std::array<std::optional<SceneObject>, kMaxObjects> scene_objects_;
    std::array<std::optional<StaticMesh>, kMaxStaticMeshes> static_meshes_;
    std::array<std::optional<Material>, kMaxMaterials> materials_;

    Image depth_image_;
    std::optional<Image::View> depth_view_;

    // currently rendered frame out of frames in flight
    uint32_t current_frame_;

    SceneState(ProgramState &state)
        : state_{state}, render_pass_{VK_NULL_HANDLE}, pipeline_layout_{VK_NULL_HANDLE},
          graphics_pipeline_{VK_NULL_HANDLE}, command_pool_{VK_NULL_HANDLE}, descriptor_pool_{VK_NULL_HANDLE},
          current_frame_{0} {
        descriptor_layout_.fill(VK_NULL_HANDLE);
    }

    SceneState(const SceneState &) = delete;
    SceneState &operator=(const SceneState &) = delete;

    bool create_framebuffers() {
        // create the depth image
        auto depth_image = memory_->create_image(VK_FORMAT_D32_SFLOAT, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
            VK_IMAGE_TYPE_2D, VkExtent3D{state_.swapchain().extent.width, state_.swapchain().extent.height, 1});

        if (!depth_image) {
            LOG_ERROR("failed to initialize depth image");
            return false;
        }

        depth_image_ = std::move(depth_image.value());
        depth_view_ = depth_image_.create_view(
            state_.dispatch(), VK_IMAGE_VIEW_TYPE_2D, VK_FORMAT_D32_SFLOAT, VK_IMAGE_ASPECT_DEPTH_BIT);

        if (!depth_view_) {
            LOG_ERROR("failed to create depth view");
            return false;
        }

        swapchain_images_ = state_.swapchain().get_images().value();
        swapchain_views_ = state_.swapchain().get_image_views().value();

        VkResult res;

        swapchain_fbs_.resize(swapchain_views_.size());
        for (size_t i = 0; i < swapchain_views_.size(); ++i) {
            std::array<VkImageView, 2> attachments = {swapchain_views_[i], depth_view_->view()};

            VkFramebufferCreateInfo framebuffer_info = {};
            framebuffer_info.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
            framebuffer_info.renderPass = render_pass_;
            framebuffer_info.attachmentCount = static_cast<uint32_t>(attachments.size());
            framebuffer_info.pAttachments = attachments.data();
            framebuffer_info.width = state_.swapchain().extent.width;
            framebuffer_info.height = state_.swapchain().extent.height;
            framebuffer_info.layers = 1;

            res = state_.dispatch().createFramebuffer(&framebuffer_info, nullptr, &swapchain_fbs_[i]);

            if (VK_SUCCESS != res) {
                LOG_ERROR("failed to create swapchain fb: %s", string_VkResult(res));
                return false;
            }
        }

        return true;
    }

public:
    ~SceneState() {
        VkResult res = state_.dispatch().deviceWaitIdle();
        if (VK_SUCCESS != res) {
            LOG_ERROR("failed to wait device idle: %s", string_VkResult(res));
        }

        LOG_INFO("destroying the scene state");

        memory_.reset(); // manually release to prevent validation errors

        for (auto &fb : swapchain_fbs_) {
            state_.dispatch().destroyFramebuffer(fb, nullptr);
        }

        state_.swapchain().destroy_image_views(swapchain_views_);

        for (auto &layout : descriptor_layout_) {
            state_.dispatch().destroyDescriptorSetLayout(layout, nullptr);
        }

        state_.dispatch().destroyDescriptorPool(descriptor_pool_, nullptr);
        state_.dispatch().destroyCommandPool(command_pool_, nullptr);
        state_.dispatch().destroyRenderPass(render_pass_, nullptr);
        state_.dispatch().destroyPipelineLayout(pipeline_layout_, nullptr);
        state_.dispatch().destroyPipeline(graphics_pipeline_, nullptr);
    }

    MemoryHelper &memory() { return *memory_; }
    MemoryHelper::DynamicUniformBuffer<cbPerObject> &object_uniforms() { return *object_uniforms_; }

    template <typename F> void with_object(const SceneObject::Id &id, F f) const {
        if (id.valid()) {
            f(scene_objects_[id.id_]);
        }
    }

    template <typename F> void with_object(const SceneObject::Id &id, F f) {
        if (id.valid()) {
            f(*scene_objects_[id.id_]);
        }
    }

    template <typename F> void with_static_mesh(const StaticMesh::Id &id, F f) {
        if (id.valid()) {
            f(*static_meshes_[id.id_]);
        }
    }

    template <typename F> void with_material(const Material::Id &id, F f) {
        if (id.valid()) {
            f(*materials_[id.id_]);
        }
    }

    SceneObject::Id create_scene_object() {
        auto iter = std::find_if(scene_objects_.begin(), scene_objects_.end(), [&](const auto &slot) { return !slot; });
        if (iter == scene_objects_.end()) {
            LOG_ERROR("too many objects allocated, the limit is %lld", kMaxStaticMeshes);
            return {};
        }

        auto id = SceneObject::Id{static_cast<uint32_t>(std::distance(scene_objects_.begin(), iter))};
        iter->emplace(SceneObject(id));

        return id;
    }

    StaticMesh::Id create_static_mesh(const Geometry &geometry) {
        // find empty slot for this mesh
        auto iter = std::find_if(static_meshes_.begin(), static_meshes_.end(), [&](const auto &slot) { return !slot; });
        if (iter == static_meshes_.end()) {
            LOG_ERROR("too many meshes allocated, the limit is %lld", kMaxStaticMeshes);
            return {};
        }

        // create geometry and upload to a VRAM buffer
        auto vertex_buffer = memory_->create_buffer(VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, geometry.vertices.data(),
            sizeof(Vertex) * std::size(geometry.vertices), /* use staging buffer */ true);

        if (!vertex_buffer) {
            LOG_ERROR("failed to create a vertex buffer");
            return {};
        }

        if (vertex_buffer->mem_prop_flags() & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) {
            LOG_INFO("vertex buffer is host-mappable");
        } else {
            LOG_INFO("vertex buffer is device-local");
        }

        LOG_INFO("vertex buffer upload complete");

        auto index_buffer = memory_->create_buffer(VK_BUFFER_USAGE_INDEX_BUFFER_BIT, geometry.indices.data(),
            sizeof(uint32_t) * std::size(geometry.indices), /* use staging buffer */ true);

        if (!index_buffer) {
            LOG_ERROR("failed to create an index buffer");
            return {};
        }

        if (index_buffer->mem_prop_flags() & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) {
            LOG_INFO("index buffer is host-mappable");
        } else {
            LOG_INFO("index buffer is device-local");
        }

        LOG_INFO("index buffer upload complete");

        auto id = StaticMesh::Id{static_cast<uint32_t>(std::distance(static_meshes_.begin(), iter))};
        iter->emplace(std::move(StaticMesh(id, std::move(*vertex_buffer), std::move(*index_buffer),
            static_cast<uint32_t>(geometry.vertices.size()), static_cast<uint32_t>(geometry.indices.size()))));

        return id;
    }

    Material::Id create_material(const Bitmap &albedo_bitmap, VkFilter filter, VkSamplerAddressMode address_mode) {
        auto iter = std::find_if(materials_.begin(), materials_.end(), [&](const auto &slot) { return !slot; });
        if (iter == materials_.end()) {
            LOG_ERROR("too many materials allocated, the limit is %lld", kMaxStaticMeshes);
            return {};
        }

        auto image = memory_->create_image_rgba(
            VK_IMAGE_USAGE_SAMPLED_BIT, albedo_bitmap.width(), albedo_bitmap.height(), albedo_bitmap.raw_pixels());

        if (!image) {
            LOG_ERROR("failed to uplaod image to the gpu memory");
            return {};
        }

        auto image_view = image->create_view(
            state_.dispatch(), VK_IMAGE_VIEW_TYPE_2D, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_ASPECT_COLOR_BIT);

        if (!image_view) {
            LOG_ERROR("failed to create image view from uploaded image");
            return {};
        }

        // create sampler
        VkSamplerCreateInfo sampler_desc = {};
        sampler_desc.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        sampler_desc.magFilter = filter;
        sampler_desc.minFilter = filter;
        sampler_desc.addressModeU = address_mode;
        sampler_desc.addressModeV = address_mode;
        sampler_desc.addressModeW = address_mode;

        VkResult res;
        VkSampler sampler = VK_NULL_HANDLE;

        res = state_.dispatch().createSampler(&sampler_desc, nullptr, &sampler);
        if (VK_SUCCESS != res) {
            LOG_ERROR("failed to create sampler: %s", string_VkResult(res));
            return {};
        }

        VkDescriptorSet descriptor_set = VK_NULL_HANDLE;

        // create per material descriptor set
        VkDescriptorSetAllocateInfo set_alloc_info = {};
        set_alloc_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        set_alloc_info.descriptorPool = descriptor_pool_;
        set_alloc_info.descriptorSetCount = 1;
        set_alloc_info.pSetLayouts = &descriptor_layout_[DescriptorSet::PerMaterial];

        res = state_.dispatch().allocateDescriptorSets(&set_alloc_info, &descriptor_set);
        if (VK_SUCCESS != res) {
            LOG_ERROR("failed to allocate per material descriptor set: %s", string_VkResult(res));
            return {};
        }

        VkDescriptorImageInfo descriptor_image_info = {};
        descriptor_image_info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        descriptor_image_info.imageView = image_view->view();
        descriptor_image_info.sampler = sampler;

        VkWriteDescriptorSet per_material_write_set = {};
        per_material_write_set.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        per_material_write_set.dstBinding = 0;
        per_material_write_set.dstSet = descriptor_set;
        per_material_write_set.descriptorCount = 1;
        per_material_write_set.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        per_material_write_set.pImageInfo = &descriptor_image_info;

        state_.dispatch().updateDescriptorSets(1, &per_material_write_set, 0, nullptr);

        auto id = Material::Id{static_cast<uint32_t>(std::distance(materials_.begin(), iter))};
        iter->emplace(Material(state_, id, std::move(*image), std::move(*image_view), sampler, descriptor_set));

        return id;
    }

    bool rebuild_swapchain() {
        LOG_INFO("rebuilding swapchain");

        if (!swapchain_fbs_.empty()) {
            state_.dispatch().deviceWaitIdle();

            for (auto &fb : swapchain_fbs_) {
                state_.dispatch().destroyFramebuffer(fb, nullptr);
            }

            state_.swapchain().destroy_image_views(swapchain_views_);
            swapchain_images_.clear();
            swapchain_views_.clear();
            swapchain_fbs_.clear();
        }

        if (!state_.init_swapchain()) {
            LOG_ERROR("failed to initialize swapchain");
            return false;
        }

        if (!create_framebuffers()) {
            LOG_ERROR("failed to create swapchain framebuffers");
            return false;
        }

        return true;
    }

    template <typename F> bool draw_frame(F draw_commands) {
        auto &frame = frame_data_[current_frame_];
        VkResult res;

        res = state_.dispatch().waitForFences(1, &frame.fence_in_flight_, VK_TRUE, UINT64_MAX);
        if (VK_SUCCESS != res) {
            LOG_ERROR("wait for fences failed: %s", string_VkResult(res));
            return false;
        }

        uint32_t image_index;
        {
            res = state_.dispatch().acquireNextImageKHR(
                state_.swapchain(), UINT64_MAX, frame.sem_image_avaliable_, VK_NULL_HANDLE, &image_index);

            switch (res) {
            case VK_SUCCESS:
            case VK_SUBOPTIMAL_KHR:
                break;
            case VK_ERROR_OUT_OF_DATE_KHR:
                return rebuild_swapchain();
            default:
                LOG_ERROR("cannot acquire next swapchain image: %s", string_VkResult(res));
                return false;
            }
        }

        // only reset the fence if any work will be submitted
        res = state_.dispatch().resetFences(1, &frame.fence_in_flight_);
        if (VK_SUCCESS != res) {
            LOG_ERROR("failed to reset frames in flight fence: %s", string_VkResult(res));
            return false;
        }

        res = state_.dispatch().resetCommandBuffer(frame.command_buffer_, 0);
        if (VK_SUCCESS != res) {
            LOG_ERROR("failed to reset command buffer: %s", string_VkResult(res));
            return false;
        }

        // begin recording command buffer
        VkCommandBufferBeginInfo cmd_begin_desc = {};
        cmd_begin_desc.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        cmd_begin_desc.flags = 0;
        cmd_begin_desc.pInheritanceInfo = nullptr;

        res = state_.dispatch().beginCommandBuffer(frame.command_buffer_, &cmd_begin_desc);
        if (VK_SUCCESS != res) {
            LOG_ERROR("failed to begin command buffer: %s", string_VkResult(res));
            return false;
        }

        std::array<VkClearValue, 2> clear_values;
        clear_values[0].color = {{0.0f, 0.0f, 0.0f, 1.0f}};
        clear_values[1].depthStencil = {1.0f, 0};

        VkRenderPassBeginInfo render_begin_desc = {};
        render_begin_desc.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        render_begin_desc.renderPass = render_pass_;
        render_begin_desc.framebuffer = swapchain_fbs_[image_index];
        render_begin_desc.renderArea = VkRect2D{{0, 0}, state_.swapchain().extent};
        render_begin_desc.pClearValues = clear_values.data();
        render_begin_desc.clearValueCount = static_cast<uint32_t>(clear_values.size());

        state_.dispatch().cmdBeginRenderPass(frame.command_buffer_, &render_begin_desc, VK_SUBPASS_CONTENTS_INLINE);
        state_.dispatch().cmdBindPipeline(frame.command_buffer_, VK_PIPELINE_BIND_POINT_GRAPHICS, graphics_pipeline_);
        state_.dispatch().cmdBindDescriptorSets(frame.command_buffer_, VK_PIPELINE_BIND_POINT_GRAPHICS,
            pipeline_layout_, DescriptorSet::PerFrame, 1, &frame.per_frame_set_, 0, nullptr);

        // dynamic state
        VkViewport vp = {};
        vp.width = state_.swapchain().extent.width;
        vp.height = state_.swapchain().extent.height;
        vp.x = 0;
        vp.y = 0;
        vp.minDepth = 0.0f;
        vp.maxDepth = 1.0f;

        VkRect2D scissor{{0, 0}, state_.swapchain().extent};

        state_.dispatch().cmdSetViewport(frame.command_buffer_, 0, 1, &vp);
        state_.dispatch().cmdSetScissor(frame.command_buffer_, 0, 1, &scissor);

        res = draw_commands(frame);
        if (VK_SUCCESS != res) {
            LOG_ERROR("draw_commands returned %s", string_VkResult(res));
            return false;
        }

        // render scene objects
        cbPerObject object_data;

        std::array<const SceneObject *, kMaxObjects> render_queue;
        auto render_queue_end = render_queue.begin();
        for (const auto &object : scene_objects_) {
            // only render objects that are valid, have valid mesh and material
            if (object && object->mesh_id().valid() && object->material_id().valid()) {
                *render_queue_end = &object.value();
                render_queue_end++;
            }
        }

        // TODO: cache the order instead of recalculating each frame
        // sort by material
        std::sort(render_queue.begin(), render_queue_end, [&](const SceneObject *first, const SceneObject *second) {
            return first->material_id() < second->material_id();
        });

        Material::Id current_material;
        for (auto iter = render_queue.begin(); iter != render_queue_end; ++iter) {
            const auto &object = *iter;

            if (object->material_id() != current_material) {
                current_material = object->material_id();

                // bind material
                with_material(current_material, [&](const Material &mat) {
                    state_.dispatch().cmdBindDescriptorSets(frame.command_buffer_, VK_PIPELINE_BIND_POINT_GRAPHICS,
                        pipeline_layout_, DescriptorSet::PerMaterial, 1, mat.descriptor_set_addr(), 0, nullptr);
                });
            }

            // bind uniforms
            auto object_index = object->id_.id_;
            auto ubo_slot = kMaxObjects * current_frame_ + object_index;
            auto ubo_offset = uint32_t(object_uniforms_->slot_offset(ubo_slot));

            object_data.world = object->transform_;
            object_uniforms_->write_slot(ubo_slot, object_data, false);

            state_.dispatch().cmdBindDescriptorSets(frame.command_buffer_, VK_PIPELINE_BIND_POINT_GRAPHICS,
                pipeline_layout_, DescriptorSet::PerObject, 1, &per_object_set_, 1, &ubo_offset);
            with_static_mesh(
                object->mesh_id_, [&](StaticMesh &mesh) { mesh.draw(state_.dispatch(), frame.command_buffer()); });
        }

        // flush caches on uniforms before submitting the command buffer
        object_uniforms_->buffer().flush();

        state_.dispatch().cmdEndRenderPass(frame.command_buffer_);
        state_.dispatch().endCommandBuffer(frame.command_buffer_);

        // submitting the recorder buffer
        VkPipelineStageFlags wait_mask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;

        VkSubmitInfo submit_info = {};
        submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit_info.pWaitDstStageMask = &wait_mask;
        submit_info.pWaitSemaphores = &frame.sem_image_avaliable_;
        submit_info.waitSemaphoreCount = 1;
        submit_info.pCommandBuffers = &frame.command_buffer_;
        submit_info.commandBufferCount = 1;
        submit_info.pSignalSemaphores = &frame.sem_render_done_;
        submit_info.signalSemaphoreCount = 1;

        state_.dispatch().queueSubmit(state_.graphics_queue(), 1, &submit_info, frame.fence_in_flight_);

        VkPresentInfoKHR present_info = {};
        present_info.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
        present_info.pWaitSemaphores = &frame.sem_render_done_;
        present_info.waitSemaphoreCount = 1;
        present_info.pImageIndices = &image_index;
        present_info.pSwapchains = &state_.swapchain().swapchain;
        present_info.swapchainCount = 1;

        // present
        {
            res = state_.dispatch().queuePresentKHR(state_.present_queue(), &present_info);
            switch (res) {
            case VK_ERROR_OUT_OF_DATE_KHR:
            case VK_SUBOPTIMAL_KHR:
                if (!rebuild_swapchain()) {
                    return false;
                }

                break;

            case VK_SUCCESS:
                break;

            default:
                LOG_ERROR("cannot present swapchain image: %s", string_VkResult(res));
                return false;
            }
        }

        current_frame_ = current_frame_ % static_cast<uint32_t>(frame_data_.size());
        return true;
    }

    static VkShaderModule shader_from_bytecode(ProgramState &state, const void *buffer, size_t size) {
        VkResult res;

        VkShaderModuleCreateInfo create_info = {};
        create_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        create_info.codeSize = size;
        create_info.pCode = reinterpret_cast<const uint32_t *>(buffer);

        VkShaderModule module = VK_NULL_HANDLE;
        res = state.dispatch().createShaderModule(&create_info, nullptr, &module);
        if (VK_SUCCESS != res) {
            LOG_ERROR("failed to create shader module: %s", string_VkResult(res));
            return VK_NULL_HANDLE;
        }

        return module;
    }

    static bool create_render_pass(ProgramState &state, VkRenderPass *render_pass) {
        VkAttachmentDescription color_attachment = {};
        color_attachment.format = state.swapchain().image_format;
        color_attachment.samples = VK_SAMPLE_COUNT_1_BIT;
        color_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        color_attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        color_attachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        color_attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        color_attachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        color_attachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

        VkAttachmentDescription depth_attachment = {};
        depth_attachment.format = VK_FORMAT_D32_SFLOAT;
        depth_attachment.samples = VK_SAMPLE_COUNT_1_BIT;
        depth_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        depth_attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        depth_attachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        depth_attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        depth_attachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        depth_attachment.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

        VkAttachmentReference color_attachment_ref = {};
        color_attachment_ref.attachment = 0;
        color_attachment_ref.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

        VkAttachmentReference depth_attachment_ref = {};
        depth_attachment_ref.attachment = 1;
        depth_attachment_ref.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

        VkSubpassDescription subpass = {};
        subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        subpass.colorAttachmentCount = 1;
        subpass.pColorAttachments = &color_attachment_ref;
        subpass.pDepthStencilAttachment = &depth_attachment_ref;

        // ensure rendering does not start until image is available
        // this is actually not needed because drivers are required to automatically insert this
        VkSubpassDependency dependency = {};
        dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
        dependency.dstSubpass = 0;
        dependency.srcStageMask =
            VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
        dependency.srcAccessMask = 0;
        dependency.dstStageMask =
            VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
        dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
                                   VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

        std::array<VkAttachmentDescription, 2> attachments{color_attachment, depth_attachment};

        VkRenderPassCreateInfo render_pass_info = {};
        render_pass_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
        render_pass_info.attachmentCount = static_cast<uint32_t>(attachments.size());
        render_pass_info.pAttachments = attachments.data();
        render_pass_info.subpassCount = 1;
        render_pass_info.pSubpasses = &subpass;
        render_pass_info.dependencyCount = 1;
        render_pass_info.pDependencies = &dependency;

        VkResult res = state.dispatch().createRenderPass(&render_pass_info, nullptr, render_pass);
        if (VK_SUCCESS != res) {
            *render_pass = VK_NULL_HANDLE;
            LOG_ERROR("failed to create render pass: %s", string_VkResult(res));
            return false;
        }

        return true;
    }

    static bool create_descriptor_data(ProgramState &state, SceneState &scene) {
        // layout of the per-frame descriptor set
        std::array<VkDescriptorSetLayoutBinding, 1> per_frame_bindings = {};
        per_frame_bindings[0] = {};
        per_frame_bindings[0].binding = 0;
        per_frame_bindings[0].descriptorCount = 1;
        per_frame_bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        per_frame_bindings[0].stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

        VkDescriptorSetLayoutCreateInfo per_frame_set_desc = {};
        per_frame_set_desc.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        per_frame_set_desc.flags = 0;
        per_frame_set_desc.bindingCount = static_cast<uint32_t>(per_frame_bindings.size());
        per_frame_set_desc.pBindings = per_frame_bindings.data();

        VkResult res = state.dispatch().createDescriptorSetLayout(
            &per_frame_set_desc, nullptr, &scene.descriptor_layout_[DescriptorSet::PerFrame]);
        if (VK_SUCCESS != res) {
            LOG_ERROR("failed to create descriptor set layout: %s", string_VkResult(res));
            return false;
        }

        // layout of the per-material descriptor set
        std::array<VkDescriptorSetLayoutBinding, 1> per_material_bindings = {};
        per_material_bindings[0] = {};
        per_material_bindings[0].binding = 0;
        per_material_bindings[0].descriptorCount = 1;
        per_material_bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        per_material_bindings[0].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

        VkDescriptorSetLayoutCreateInfo per_material_set_desc = {};
        per_material_set_desc.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        per_material_set_desc.flags = 0;
        per_material_set_desc.bindingCount = static_cast<uint32_t>(per_material_bindings.size());
        per_material_set_desc.pBindings = per_material_bindings.data();

        res = state.dispatch().createDescriptorSetLayout(
            &per_material_set_desc, nullptr, &scene.descriptor_layout_[DescriptorSet::PerMaterial]);
        if (VK_SUCCESS != res) {
            LOG_ERROR("failed to create descriptor set layout: %s", string_VkResult(res));
            return false;
        }

        // layout of the per-object descriptor set
        std::array<VkDescriptorSetLayoutBinding, 1> per_object_bindings = {};
        per_object_bindings[0] = {};
        per_object_bindings[0].binding = 0;
        per_object_bindings[0].descriptorCount = 1;
        per_object_bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC;
        per_object_bindings[0].stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

        VkDescriptorSetLayoutCreateInfo per_object_set_desc = {};
        per_object_set_desc.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        per_object_set_desc.flags = 0;
        per_object_set_desc.bindingCount = static_cast<uint32_t>(per_object_bindings.size());
        per_object_set_desc.pBindings = per_object_bindings.data();

        res = state.dispatch().createDescriptorSetLayout(
            &per_object_set_desc, nullptr, &scene.descriptor_layout_[DescriptorSet::PerObject]);
        if (VK_SUCCESS != res) {
            LOG_ERROR("failed to create descriptor set layout: %s", string_VkResult(res));
            return false;
        }

        // allocate descriptor pool
        // clang-format off
        std::array<VkDescriptorPoolSize, 3> pool_sizes = {
            VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 10},
            VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 10},
            VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 10}
        };
        // clang-format on

        VkDescriptorPoolCreateInfo pool_desc = {};
        pool_desc.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        pool_desc.flags = 0;
        pool_desc.maxSets = 100;
        pool_desc.poolSizeCount = static_cast<uint32_t>(pool_sizes.size());
        pool_desc.pPoolSizes = pool_sizes.data();

        res = state.dispatch().createDescriptorPool(&pool_desc, nullptr, &scene.descriptor_pool_);
        if (VK_SUCCESS != res) {
            LOG_ERROR("failed to allocate descriptor pool: %s", string_VkResult(res));
            return false;
        }

        return true;
    }

    static bool create_pipeline_layout(ProgramState &state, SceneState &scene) {
        VkPipelineLayoutCreateInfo pipeline_layout_info = {};
        pipeline_layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipeline_layout_info.setLayoutCount = static_cast<uint32_t>(DescriptorSet::Count);
        pipeline_layout_info.pSetLayouts = scene.descriptor_layout_.data();
        pipeline_layout_info.pushConstantRangeCount = 0;

        VkResult res;
        res = state.dispatch().createPipelineLayout(&pipeline_layout_info, nullptr, &scene.pipeline_layout_);
        if (VK_SUCCESS != res) {
            scene.pipeline_layout_ = VK_NULL_HANDLE;
            LOG_ERROR("failed to pipeline layout: %s", string_VkResult(res));
            return false;
        }

        return true;
    }

    static bool create_graphics_pipeline(ProgramState &state, VkPipelineLayout layout, VkRenderPass render_pass,
        uint32_t subpass_index, VkPipeline *pipeline) {
        constexpr std::array<VkDynamicState, 2> kDynamicStates = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};

        // shader modules
        VkShaderModule vs_module = shader_from_bytecode(state, kVertex_spv.data(), kVertex_spv.size());
        if (vs_module == VK_NULL_HANDLE) {
            LOG_ERROR("fatal error when creating vertex shader module");
            return false;
        }

        VkShaderModule fs_module = shader_from_bytecode(state, kFragment_spv.data(), kFragment_spv.size());
        if (fs_module == VK_NULL_HANDLE) {
            LOG_ERROR("fatal error when creating fragment shader module");
            state.dispatch().destroyShaderModule(vs_module, nullptr);
            return false;
        }

        VkPipelineShaderStageCreateInfo vert_stage_info = {};
        vert_stage_info.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        vert_stage_info.stage = VK_SHADER_STAGE_VERTEX_BIT;
        vert_stage_info.module = vs_module;
        vert_stage_info.pName = "main";

        VkPipelineShaderStageCreateInfo frag_stage_info = {};
        frag_stage_info.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        frag_stage_info.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        frag_stage_info.module = fs_module;
        frag_stage_info.pName = "main";

        std::array<VkPipelineShaderStageCreateInfo, 2> shader_stages = {vert_stage_info, frag_stage_info};

        // set viewport and scissor as dynamic state
        VkPipelineDynamicStateCreateInfo dynamic_state_desc = {};
        dynamic_state_desc.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
        dynamic_state_desc.pDynamicStates = kDynamicStates.data();
        dynamic_state_desc.dynamicStateCount = static_cast<uint32_t>(kDynamicStates.size());

        // input state using the vertex struct
        constexpr std::array<VkVertexInputAttributeDescription, 3> kVertexAttribDesc = {
            VkVertexInputAttributeDescription{0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, position)},
            VkVertexInputAttributeDescription{1, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, normal)},
            VkVertexInputAttributeDescription{2, 0, VK_FORMAT_R32G32_SFLOAT, offsetof(Vertex, uv)},
        };

        VkVertexInputBindingDescription vertex_binding_desc = {};
        vertex_binding_desc.binding = 0;
        vertex_binding_desc.stride = sizeof(Vertex);
        vertex_binding_desc.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

        VkPipelineVertexInputStateCreateInfo input_state_desc = {};
        input_state_desc.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        input_state_desc.pVertexAttributeDescriptions = kVertexAttribDesc.data();
        input_state_desc.vertexAttributeDescriptionCount = static_cast<uint32_t>(kVertexAttribDesc.size());
        input_state_desc.pVertexBindingDescriptions = &vertex_binding_desc;
        input_state_desc.vertexBindingDescriptionCount = 1;

        // input assembly
        VkPipelineInputAssemblyStateCreateInfo assembly_desc = {};
        assembly_desc.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        assembly_desc.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        assembly_desc.primitiveRestartEnable = false;

        // dynamic state
        VkPipelineViewportStateCreateInfo viewport_desc = {};
        viewport_desc.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
        viewport_desc.viewportCount = 1;
        viewport_desc.scissorCount = 1;

        // rasterization
        VkPipelineRasterizationStateCreateInfo rasterizer_desc = {};
        rasterizer_desc.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        rasterizer_desc.depthClampEnable = false;
        rasterizer_desc.rasterizerDiscardEnable = false;
        rasterizer_desc.polygonMode = VK_POLYGON_MODE_FILL;
        rasterizer_desc.lineWidth = 1.0f;
        rasterizer_desc.cullMode = VK_CULL_MODE_NONE;
        rasterizer_desc.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
        rasterizer_desc.depthBiasEnable = false;
        rasterizer_desc.depthBiasConstantFactor = 0.0f;
        rasterizer_desc.depthBiasClamp = 0.0f;
        rasterizer_desc.depthBiasSlopeFactor = 0.0f;

        // no multisampling
        VkPipelineMultisampleStateCreateInfo multisample_desc = {};
        multisample_desc.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        multisample_desc.sampleShadingEnable = false;
        multisample_desc.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
        multisample_desc.minSampleShading = 1.0f;
        multisample_desc.pSampleMask = nullptr;
        multisample_desc.alphaToCoverageEnable = false;
        multisample_desc.alphaToOneEnable = false;

        // no blending
        VkPipelineColorBlendAttachmentState blend_att_desc = {};
        blend_att_desc.colorWriteMask =
            VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
        blend_att_desc.blendEnable = false;

        VkPipelineColorBlendStateCreateInfo blend_desc = {};
        blend_desc.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        blend_desc.logicOpEnable = false;
        blend_desc.logicOp = VK_LOGIC_OP_COPY;
        blend_desc.blendConstants[0] = 0.0f;
        blend_desc.blendConstants[1] = 0.0f;
        blend_desc.blendConstants[2] = 0.0f;
        blend_desc.blendConstants[3] = 0.0f;
        blend_desc.pAttachments = &blend_att_desc;
        blend_desc.attachmentCount = 1;

        VkPipelineDepthStencilStateCreateInfo depth_desc = {};
        depth_desc.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
        depth_desc.depthWriteEnable = VK_TRUE;
        depth_desc.depthTestEnable = VK_TRUE;
        depth_desc.depthCompareOp = VK_COMPARE_OP_LESS;
        depth_desc.depthBoundsTestEnable = VK_FALSE;
        depth_desc.stencilTestEnable = VK_FALSE;

        // finally, create the graphics pipeline
        VkGraphicsPipelineCreateInfo create_info = {};
        create_info.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        create_info.pStages = shader_stages.data();
        create_info.stageCount = static_cast<uint32_t>(shader_stages.size());
        create_info.pVertexInputState = &input_state_desc;
        create_info.pInputAssemblyState = &assembly_desc;
        create_info.pViewportState = &viewport_desc;
        create_info.pRasterizationState = &rasterizer_desc;
        create_info.pMultisampleState = &multisample_desc;
        create_info.pColorBlendState = &blend_desc;
        create_info.pDynamicState = &dynamic_state_desc;
        create_info.pDepthStencilState = &depth_desc;
        create_info.layout = layout;
        create_info.renderPass = render_pass;
        create_info.subpass = subpass_index;

        VkResult res = state.dispatch().createGraphicsPipelines(VK_NULL_HANDLE, 1, &create_info, nullptr, pipeline);

        // cleanup
        state.dispatch().destroyShaderModule(vs_module, nullptr);
        state.dispatch().destroyShaderModule(fs_module, nullptr);

        if (VK_SUCCESS != res) {
            *pipeline = VK_NULL_HANDLE;
            LOG_ERROR("failed to create pipeline: %s", string_VkResult(res));
            return false;
        }

        return true;
    }

    static bool create_command_pool(ProgramState &state, uint32_t family_index, VkCommandPool *command_pool) {
        VkCommandPoolCreateInfo create_info = {};
        create_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        create_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        create_info.queueFamilyIndex = family_index;

        VkResult res = state.dispatch().createCommandPool(&create_info, nullptr, command_pool);
        if (VK_SUCCESS != res) {
            *command_pool = VK_NULL_HANDLE;
            LOG_ERROR("failed to create command pool: %s", string_VkResult(res));

            return false;
        }

        return true;
    }

    static bool create_object_data(ProgramState &state, SceneState &scene, uint32_t frames_in_flight) {
        scene.object_uniforms_ = scene.memory_->init_dynamic_ubo<cbPerObject>(kMaxObjects * frames_in_flight);
        if (!scene.object_uniforms_) {
            LOG_ERROR("failed to allocate dynamic uniform buffer");
            return false;
        }

        // create per object descriptor set
        VkDescriptorSetAllocateInfo set_alloc_info = {};
        set_alloc_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        set_alloc_info.descriptorPool = scene.descriptor_pool_;
        set_alloc_info.descriptorSetCount = 1;
        set_alloc_info.pSetLayouts = &scene.descriptor_layout_[DescriptorSet::PerObject];

        VkResult res = state.dispatch().allocateDescriptorSets(&set_alloc_info, &scene.per_object_set_);
        if (VK_SUCCESS != res) {
            LOG_ERROR("failed to allocate per object descriptor set: %s", string_VkResult(res));
            return false;
        }

        VkDescriptorBufferInfo per_object_buffer_desc = {};
        per_object_buffer_desc.buffer = scene.object_uniforms_->buffer().buffer();
        per_object_buffer_desc.offset = 0;
        per_object_buffer_desc.range = sizeof(cbPerFrame);

        VkWriteDescriptorSet per_object_write_set = {};
        per_object_write_set.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        per_object_write_set.dstBinding = 0;
        per_object_write_set.dstSet = scene.per_object_set_;
        per_object_write_set.descriptorCount = 1;
        per_object_write_set.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC;
        per_object_write_set.pBufferInfo = &per_object_buffer_desc;

        state.dispatch().updateDescriptorSets(1, &per_object_write_set, 0, nullptr);
        return true;
    }

    static bool create_frame_data(ProgramState &state, SceneState &scene, uint32_t frames_in_flight) {
        VkResult res;

        std::vector<VkCommandBuffer> command_buffers;
        command_buffers.resize(frames_in_flight);
        std::fill(command_buffers.begin(), command_buffers.end(), VK_NULL_HANDLE);

        VkCommandBufferAllocateInfo buffer_info = {};
        buffer_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        buffer_info.commandPool = scene.command_pool_;
        buffer_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        buffer_info.commandBufferCount = frames_in_flight;

        res = state.dispatch().allocateCommandBuffers(&buffer_info, command_buffers.data());
        if (VK_SUCCESS != res) {
            LOG_ERROR("failed to create command buffers: %s", string_VkResult(res));
            return false;
        }

        for (uint32_t f = 0; f < frames_in_flight; ++f) {
            scene.frame_data_.push_back(FrameSubmitData(state, scene));
            auto &frame = scene.frame_data_.back();

            VkSemaphoreCreateInfo sem_create_info = {};
            sem_create_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

            VkFenceCreateInfo fence_create_info = {};
            fence_create_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
            fence_create_info.flags = VK_FENCE_CREATE_SIGNALED_BIT;

            frame.command_buffer_ = command_buffers[f];

            res = state.dispatch().createSemaphore(&sem_create_info, nullptr, &frame.sem_image_avaliable_);
            if (VK_SUCCESS != res) {
                LOG_ERROR("failed to create semaphore: %s", string_VkResult(res));
                return false;
            }

            res = state.dispatch().createSemaphore(&sem_create_info, nullptr, &frame.sem_render_done_);
            if (VK_SUCCESS != res) {
                LOG_ERROR("failed to create semaphore: %s", string_VkResult(res));
                return false;
            }

            res = state.dispatch().createFence(&fence_create_info, nullptr, &frame.fence_in_flight_);
            if (VK_SUCCESS != res) {
                LOG_ERROR("failed to create fence: %s", string_VkResult(res));
                return false;
            }

            // allocate per frame ubo data
            auto buffer = scene.memory_->create_shared_buffer(VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, sizeof(cbPerFrame));
            if (!buffer) {
                LOG_ERROR("failed allocating shared buffer");
                return false;
            }

            frame.per_frame_buffer_ = std::move(buffer.value());

            // allocate descriptor set
            VkDescriptorSetAllocateInfo set_alloc_info = {};
            set_alloc_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
            set_alloc_info.descriptorPool = scene.descriptor_pool_;
            set_alloc_info.descriptorSetCount = 1;
            set_alloc_info.pSetLayouts = &scene.descriptor_layout_[DescriptorSet::PerFrame];

            res = state.dispatch().allocateDescriptorSets(&set_alloc_info, &frame.per_frame_set_);
            if (VK_SUCCESS != res) {
                LOG_ERROR("failed to allocate per frame descriptor set: %s", string_VkResult(res));
                return false;
            }

            // point the descriptor set to the buffer
            VkDescriptorBufferInfo per_frame_buffer_desc = {};
            per_frame_buffer_desc.buffer = frame.per_frame_buffer_.buffer();
            per_frame_buffer_desc.offset = 0;
            per_frame_buffer_desc.range = sizeof(cbPerFrame);

            VkWriteDescriptorSet per_frame_write_set = {};
            per_frame_write_set.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            per_frame_write_set.dstBinding = 0;
            per_frame_write_set.dstSet = frame.per_frame_set_;
            per_frame_write_set.descriptorCount = 1;
            per_frame_write_set.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            per_frame_write_set.pBufferInfo = &per_frame_buffer_desc;

            state.dispatch().updateDescriptorSets(1, &per_frame_write_set, 0, nullptr);
        }

        return true;
    }

    static std::unique_ptr<SceneState> initialize(ProgramState &state) {
        std::unique_ptr<SceneState> scene{new SceneState(state)};

        auto memory = MemoryHelper::initialize(state);
        if (!memory) {
            LOG_ERROR("failed to initialize memory helper");
            return {};
        }

        scene->memory_ = std::move(memory);
        LOG_INFO("initialized memory helper");

        if (!create_render_pass(state, &scene->render_pass_)) {
            LOG_ERROR("failed to create render pass");
            return {};
        }

        LOG_INFO("created render pass");

        if (!scene->create_framebuffers()) {
            LOG_ERROR("failed to create swapchain framebuffers");
            return {};
        }

        LOG_INFO("created the swapchain framebuffers");

        if (!create_descriptor_data(state, *scene)) {
            LOG_ERROR("failed to initialize descriptor data");
            return {};
        }

        LOG_INFO("descriptor data initialized");

        if (!create_pipeline_layout(state, *scene)) {
            LOG_ERROR("failed to create pipeline layout");
            return {};
        }

        if (!create_graphics_pipeline(
                state, scene->pipeline_layout_, scene->render_pass_, 0, &scene->graphics_pipeline_)) {
            LOG_ERROR("failed to create pipeline");
            return {};
        }

        LOG_INFO("created graphics pipeline");

        if (!create_command_pool(
                state, state.device().get_queue_index(vkb::QueueType::graphics).value(), &scene->command_pool_)) {
            LOG_ERROR("failed to create command pool");
            return {};
        }

        LOG_INFO("created command pool");

        if (!create_object_data(state, *scene, kFramesInFlight)) {
            LOG_ERROR("failed to create object buffers");
            return {};
        }

        LOG_INFO("created per-object uniform buffer");

        if (!create_frame_data(state, *scene, kFramesInFlight)) {
            LOG_ERROR("failed to create frame submission data");
            return {};
        }

        LOG_INFO("created frame submission data");

        return scene;
    }
};

struct VulkanSample final {
private:
    using Clock = std::chrono::high_resolution_clock;

    ProgramState &state_;
    SceneState &scene_;

    Geometry cube_geometry_;

    SceneState::Material::Id material_;
    SceneState::StaticMesh::Id cube_mesh_;
    SceneState::SceneObject::Id cube_object_, test_object_;

    cbPerFrame per_frame_;
    Clock::time_point last_time_;
    float time_elapsed_;

    VulkanSample(ProgramState &state, SceneState &scene) : state_{state}, scene_{scene} {
        time_elapsed_ = 0.0f;
        last_time_ = Clock::now();
    }

public:
    VulkanSample(const VulkanSample &) = delete;
    ~VulkanSample() {
        LOG_INFO("destroying sample state");

        VkResult res = state_.dispatch().deviceWaitIdle();
        if (VK_SUCCESS != res) {
            LOG_ERROR("failed to wait device idle: %s", string_VkResult(res));
        }
    }

    VkResult frame(SceneState::FrameSubmitData &frame) {
        // calculate delta time
        constexpr double kNsToSeconds = 1e-9f;
        auto now = Clock::now();
        auto delta_time = static_cast<float>(
            static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(now - last_time_).count()) *
            kNsToSeconds);
        last_time_ = now;

        time_elapsed_ = time_elapsed_ + delta_time;

        // animate objects
        scene_.with_object(cube_object_, [&](SceneState::SceneObject &object) {
            object.set_rotation(glm::angleAxis(time_elapsed_ * +0.5f * glm::pi<float>(), glm::fvec3{0.0f, 1.0f, 0.0f}));
        });

        scene_.with_object(test_object_, [&](SceneState::SceneObject &object) {
            object.set_rotation(glm::angleAxis(time_elapsed_ * -1.0f * glm::pi<float>(), glm::fvec3{0.0f, 0.0f, 1.0f}));
        });

        // update camera
        float aspect =
            static_cast<float>(state_.swapchain().extent.width) / static_cast<float>(state_.swapchain().extent.height);

        per_frame_.view =
            glm::lookAt(glm::fvec3{5.0f, 5.0f, 5.0f}, glm::fvec3{0.0f, 0.0f, 0.0f}, glm::fvec3{0.0f, 1.0f, 0.0f});
        per_frame_.proj = glm::perspective(glm::pi<float>() * 0.25f, aspect, 0.5f, 50.0f);
        per_frame_.proj[1][1] *= -1.0f;

        frame.update_per_frame(per_frame_);

        return VK_SUCCESS;
    }

    static std::optional<Bitmap> load_png(const uint8_t *buffer, size_t size) {
        int width, height, components;
        int res = stbi_info_from_memory(buffer, size, &width, &height, &components);

        if (0 == res) {
            LOG_ERROR("cannot load png file: unsupported format");
            return {};
        }

        Bitmap bitmap{static_cast<uint32_t>(width), static_cast<uint32_t>(height)};
        auto img = stbi_load_from_memory(buffer, size, &width, &height, &components, 4);
        if (!img) {
            LOG_ERROR("failed to load png image");
            return {};
        }

        // since we passed 4 components as req, we can safely assume its fine now
        memcpy(bitmap.raw_pixels(), img, width * height * 4);

        stbi_image_free(img);
        return std::move(bitmap);
    }

    static Geometry cube_geometry() {
        using V = Vertex;
        // clang-format off
return Geometry{
    std::vector<V>{
        V{{  1.0f,  1.0f, -1.0f }, {  0.0f,  1.0f,  0.0f }, { 0.0f, 0.0f }},
        V{{ -1.0f,  1.0f, -1.0f }, {  0.0f,  1.0f,  0.0f }, { 1.0f, 0.0f }},
        V{{ -1.0f,  1.0f,  1.0f }, {  0.0f,  1.0f,  0.0f }, { 1.0f, 1.0f }},
        V{{  1.0f,  1.0f,  1.0f }, {  0.0f,  1.0f,  0.0f }, { 0.0f, 1.0f }},

        V{{  1.0f, -1.0f,  1.0f }, {  0.0f,  0.0f,  1.0f }, { 0.0f, 0.0f }},
        V{{  1.0f,  1.0f,  1.0f }, {  0.0f,  0.0f,  1.0f }, { 1.0f, 0.0f }},
        V{{ -1.0f,  1.0f,  1.0f }, {  0.0f,  0.0f,  1.0f }, { 1.0f, 1.0f }},
        V{{ -1.0f, -1.0f,  1.0f }, {  0.0f,  0.0f,  1.0f }, { 0.0f, 1.0f }},

        V{{ -1.0f, -1.0f,  1.0f }, { -1.0f,  0.0f,  0.0f }, { 0.0f, 0.0f }},
        V{{ -1.0f,  1.0f,  1.0f }, { -1.0f,  0.0f,  0.0f }, { 1.0f, 0.0f }},
        V{{ -1.0f,  1.0f, -1.0f }, { -1.0f,  0.0f,  0.0f }, { 1.0f, 1.0f }},
        V{{ -1.0f, -1.0f, -1.0f }, { -1.0f,  0.0f,  0.0f }, { 0.0f, 1.0f }},

        V{{ -1.0f, -1.0f, -1.0f }, {  0.0f, -1.0f,  0.0f }, { 0.0f, 0.0f }},
        V{{  1.0f, -1.0f, -1.0f }, {  0.0f, -1.0f,  0.0f }, { 1.0f, 0.0f }},
        V{{  1.0f, -1.0f,  1.0f }, {  0.0f, -1.0f,  0.0f }, { 1.0f, 1.0f }},
        V{{ -1.0f, -1.0f,  1.0f }, {  0.0f, -1.0f,  0.0f }, { 0.0f, 1.0f }},

        V{{  1.0f, -1.0f, -1.0f }, {  1.0f,  0.0f,  0.0f }, { 0.0f, 0.0f }},
        V{{  1.0f,  1.0f, -1.0f }, {  1.0f,  0.0f,  0.0f }, { 1.0f, 0.0f }},
        V{{  1.0f,  1.0f,  1.0f }, {  1.0f,  0.0f,  0.0f }, { 1.0f, 1.0f }},
        V{{  1.0f, -1.0f,  1.0f }, {  1.0f,  0.0f,  0.0f }, { 0.0f, 1.0f }},

        V{{ -1.0f, -1.0f, -1.0f }, {  0.0f,  0.0f, -1.0f }, { 0.0f, 0.0f }},
        V{{ -1.0f,  1.0f, -1.0f }, {  0.0f,  0.0f, -1.0f }, { 1.0f, 0.0f }},
        V{{  1.0f,  1.0f, -1.0f }, {  0.0f,  0.0f, -1.0f }, { 1.0f, 1.0f }},
        V{{  1.0f, -1.0f, -1.0f }, {  0.0f,  0.0f, -1.0f }, { 0.0f, 1.0f }}
    },
    std::vector<uint32_t>{
        0,  1,  2,  0,  2,  3, 
        4,  5,  6,  4,  6,  7, 
        8,  9,  10, 8,  10, 11, 
        12, 13, 14, 12, 14, 15, 
        16, 17, 18, 16, 18, 19, 
        20, 21, 22, 20, 22, 23
    }
};
        // clang-format on
    }

    static Geometry plane_geometry() {
        using V = Vertex;
        // clang-format off
return Geometry{
    std::vector<V>{
        V{{ -1.0f,  1.0f,  0.0f }, {  1.0f,  0.0f,  0.0f }, { 0.0f, 1.0f }},
        V{{  1.0f,  1.0f,  0.0f }, {  1.0f,  0.0f,  0.0f }, { 1.0f, 1.0f }},
        V{{  1.0f, -1.0f,  0.0f }, {  1.0f,  0.0f,  0.0f }, { 1.0f, 0.0f }},
        V{{ -1.0f,  1.0f,  0.0f }, {  1.0f,  0.0f,  0.0f }, { 0.0f, 1.0f }},
        V{{  1.0f, -1.0f,  0.0f }, {  1.0f,  0.0f,  0.0f }, { 1.0f, 0.0f }},
        V{{ -1.0f, -1.0f,  0.0f }, {  1.0f,  0.0f,  0.0f }, { 0.0f, 0.0f }}
    },
    std::vector<uint32_t>{
        0, 1, 2, 3, 4, 5
    }
};
        // clang-format on
    }

    static std::unique_ptr<VulkanSample> initialize(ProgramState &state, SceneState &scene) {
        std::unique_ptr<VulkanSample> sample{new VulkanSample(state, scene)};

        // load material
        auto bitmap = load_png(kBricks_png.data(), kBricks_png.size());
        if (!bitmap) {
            LOG_ERROR("failed to load png image bricks.png");
            return {};
        }

        auto material = scene.create_material(bitmap.value(), VK_FILTER_LINEAR, VK_SAMPLER_ADDRESS_MODE_REPEAT);
        if (!material.valid()) {
            LOG_ERROR("failed to create material for bricks.png");
            return {};
        }

        // create geometry and upload to a gram buffer
        auto geometry = cube_geometry();

        // upload meshes
        auto cube_mesh = scene.create_static_mesh(geometry);
        if (!cube_mesh.valid()) {
            LOG_ERROR("failed to upload cube mesh");
            return {};
        }

        auto cube_object = scene.create_scene_object();
        auto test_object = scene.create_scene_object();

        scene.with_object(cube_object, [&](SceneState::SceneObject &object) {
            object.set_translation(glm::fvec3{-2.5f, 0.0f, 0.0f});
            object.set_mesh_id(cube_mesh);
            object.set_material_id(material);
        });

        scene.with_object(test_object, [&](SceneState::SceneObject &object) {
            object.set_translation(glm::fvec3{+2.5f, 0.0f, 0.0f});
            object.set_mesh_id(cube_mesh);
            object.set_material_id(material);
        });

        sample->material_ = material;
        sample->cube_mesh_ = cube_mesh;
        sample->cube_object_ = cube_object;
        sample->test_object_ = test_object;

        return sample;
    }
};

int main(int argc, char **argv) {
    glfwInit();
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);

    LOG_INFO("using backend glfw");
    auto window = glfwCreateWindow(1366, 768, "minimal sample", nullptr, nullptr);

    // init vk
    auto program_state = ProgramState::initialize(window);

    if (!program_state) {
        LOG_ERROR("fatal initialization error, halting");
        return EXIT_FAILURE;
    }

    auto scene_state = SceneState::initialize(*program_state);

    if (!scene_state) {
        LOG_ERROR("fatal initialization error, halting");
        return EXIT_FAILURE;
    }

    auto sample = VulkanSample::initialize(*program_state, *scene_state);

    if (!sample) {
        LOG_ERROR("fatal initialization error, halting");
        return EXIT_FAILURE;
    }

    // event loop of the window
    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();

        if (!scene_state->draw_frame([&](SceneState::FrameSubmitData &frame) -> VkResult {
            // allow the sample to record its command queue
            return sample->frame(frame);
        })) {
            LOG_ERROR("a fatal error has occured while rendering a frame");
            return EXIT_FAILURE;
        }
    }

    // order of destruction is important here
    sample.reset();
    scene_state.reset();
    program_state.reset();

    glfwDestroyWindow(window);
    glfwTerminate();

    LOG_INFO("graceful program exit condition");
    return EXIT_SUCCESS;
}
