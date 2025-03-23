#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <optional>
#include <vector>
#include <array>

#include <stdlib.h>
#include <stdio.h>

// libraries - ignore all warnings
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Weverything"

#include <glm/glm.hpp>
#include <vulkan/vulkan.h>
#include <vulkan/vulkan_core.h>
#include <vulkan/vk_enum_string_helper.h>
#include <GLFW/glfw3.h>

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

#define LOG_ERROR(fmt, ...) fprintf(stderr, "[error] " fmt "\n", ##__VA_ARGS__)
#define LOG_INFO(fmt, ...) fprintf(stderr, "[info] " fmt "\n", ##__VA_ARGS__)

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
        destroy();
        allocator_ = b.allocator_;
        allocation_ = b.allocation_;
        alloc_info_ = b.alloc_info_;
        buffer_ = b.buffer_;

        b.allocator_ = VMA_NULL;
        b.allocation_ = VMA_NULL;
        b.buffer_ = VK_NULL_HANDLE;

        return *this;
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
        destroy();
        allocator_ = i.allocator_;
        allocation_ = i.allocation_;
        alloc_info_ = i.alloc_info_;
        image_ = i.image_;

        i.allocator_ = VMA_NULL;
        i.allocation_ = VMA_NULL;
        i.image_ = VK_NULL_HANDLE;

        return *this;
    }

    void destroy() {
        if (allocator_ != VMA_NULL && image_ != VK_NULL_HANDLE) {
            vmaDestroyImage(allocator_, image_, allocation_);
        }

        allocator_ = VMA_NULL;
        allocation_ = VMA_NULL;
        image_ = VK_NULL_HANDLE;
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

    // memory allocation
    VmaVulkanFunctions allocator_fns_;
    VmaAllocator allocator_;

    ProgramState() : surface_{VK_NULL_HANDLE}, allocator_{VMA_NULL} {};
    ProgramState(const ProgramState &) = delete;
    ProgramState &operator=(const ProgramState) = delete;

public:
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

        LOG_INFO("created vk device successfully");

        if (!state->init_swapchain()) {
            LOG_ERROR("failed to initialize swapchain");
            return {};
        }

        // init vma
        state->allocator_fns_ = {};
        state->allocator_fns_.vkGetInstanceProcAddr = state->instance_.fp_vkGetInstanceProcAddr;
        state->allocator_fns_.vkGetDeviceProcAddr = state->instance_.fp_vkGetDeviceProcAddr;

        VmaAllocatorCreateInfo alloc_create_info{};
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
constexpr uint32_t kStagingBufferSize = 64 * 1024;

struct Vertex {
    glm::fvec3 position;
    glm::fvec3 normal;
    glm::fvec2 uv;
};

struct Geometry {
    std::vector<Vertex> vertices;
    std::vector<uint16_t> indices;
};

struct SceneState final {
public:
    struct FrameSubmitData final {
    private:
        ProgramState &state_;
        SceneState &scene_;

        VkCommandBuffer command_buffer_;
        VkSemaphore sem_image_avaliable_, sem_render_done_;
        VkFence fence_in_flight_;

        FrameSubmitData(ProgramState &state, SceneState &scene)
            : state_{state}, scene_{scene}, command_buffer_{VK_NULL_HANDLE}, sem_image_avaliable_{VK_NULL_HANDLE},
              sem_render_done_{VK_NULL_HANDLE}, fence_in_flight_{VK_NULL_HANDLE} {}

    public:
        FrameSubmitData(const FrameSubmitData &) = delete;
        FrameSubmitData &operator=(const FrameSubmitData &) = delete;

        FrameSubmitData(FrameSubmitData &&f) : state_{f.state_}, scene_{f.scene_} {
            command_buffer_ = f.command_buffer_;
            sem_image_avaliable_ = f.sem_image_avaliable_;
            sem_render_done_ = f.sem_render_done_;
            fence_in_flight_ = f.fence_in_flight_;

            f.command_buffer_ = VK_NULL_HANDLE;
            f.sem_image_avaliable_ = VK_NULL_HANDLE;
            f.sem_render_done_ = VK_NULL_HANDLE;
            f.fence_in_flight_ = VK_NULL_HANDLE;
        }

        ~FrameSubmitData() {
            state_.dispatch().destroySemaphore(sem_image_avaliable_, nullptr);
            state_.dispatch().destroySemaphore(sem_render_done_, nullptr);
            state_.dispatch().destroyFence(fence_in_flight_, nullptr);
        }

        friend struct SceneState;
    };

    ProgramState &state_;
    VkQueue graphics_queue_, present_queue_;
    VkRenderPass render_pass_;
    VkPipelineLayout pipeline_layout_;
    VkPipeline graphics_pipeline_;
    VkCommandPool command_pool_;

    // resource uploading
    Buffer staging_buffer_;
    VkCommandBuffer upload_buffer_;

    std::vector<VkImage> swapchain_images_;
    std::vector<VkImageView> swapchain_views_;
    std::vector<VkFramebuffer> swapchain_fbs_;
    std::vector<FrameSubmitData> frame_data_;

    // currently rendered frame out of frames in flight
    uint32_t current_frame_;

private:
    SceneState(ProgramState &state)
        : state_{state}, graphics_queue_{VK_NULL_HANDLE}, present_queue_{VK_NULL_HANDLE}, render_pass_{VK_NULL_HANDLE},
          pipeline_layout_{VK_NULL_HANDLE}, graphics_pipeline_{VK_NULL_HANDLE}, command_pool_{VK_NULL_HANDLE},
          current_frame_{0} {}
    SceneState(const SceneState &) = delete;
    SceneState &operator=(const SceneState &) = delete;

    bool create_framebuffers() {
        swapchain_images_ = state_.swapchain().get_images().value();
        swapchain_views_ = state_.swapchain().get_image_views().value();

        VkResult res;

        swapchain_fbs_.resize(swapchain_views_.size());
        for (size_t i = 0; i < swapchain_views_.size(); ++i) {
            VkImageView attachments[] = {swapchain_views_[i]};

            VkFramebufferCreateInfo framebuffer_info = {};
            framebuffer_info.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
            framebuffer_info.renderPass = render_pass_;
            framebuffer_info.attachmentCount = 1;
            framebuffer_info.pAttachments = attachments;
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

        for (auto &fb : swapchain_fbs_) {
            state_.dispatch().destroyFramebuffer(fb, nullptr);
        }

        state_.swapchain().destroy_image_views(swapchain_views_);

        state_.dispatch().destroyCommandPool(command_pool_, nullptr);
        state_.dispatch().destroyRenderPass(render_pass_, nullptr);
        state_.dispatch().destroyPipelineLayout(pipeline_layout_, nullptr);
        state_.dispatch().destroyPipeline(graphics_pipeline_, nullptr);
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
        VkCommandBufferBeginInfo cmd_begin_desc{};
        cmd_begin_desc.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        cmd_begin_desc.flags = 0;
        cmd_begin_desc.pInheritanceInfo = nullptr;

        res = state_.dispatch().beginCommandBuffer(frame.command_buffer_, &cmd_begin_desc);
        if (VK_SUCCESS != res) {
            LOG_ERROR("failed to begin command buffer: %s", string_VkResult(res));
            return false;
        }

        VkClearValue clear_value;
        clear_value.color.float32[0] = 0.0f;
        clear_value.color.float32[1] = 0.0f;
        clear_value.color.float32[2] = 0.0f;
        clear_value.color.float32[3] = 1.0f;

        VkRenderPassBeginInfo render_begin_desc{};
        render_begin_desc.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        render_begin_desc.renderPass = render_pass_;
        render_begin_desc.framebuffer = swapchain_fbs_[image_index];
        render_begin_desc.renderArea = VkRect2D{{0, 0}, state_.swapchain().extent};
        render_begin_desc.pClearValues = &clear_value;
        render_begin_desc.clearValueCount = 1;

        state_.dispatch().cmdBeginRenderPass(frame.command_buffer_, &render_begin_desc, VK_SUBPASS_CONTENTS_INLINE);
        state_.dispatch().cmdBindPipeline(frame.command_buffer_, VK_PIPELINE_BIND_POINT_GRAPHICS, graphics_pipeline_);

        // dynamic state
        VkViewport vp{};
        vp.width = state_.swapchain().extent.width;
        vp.height = state_.swapchain().extent.height;
        vp.x = 0;
        vp.y = 0;
        vp.minDepth = 0.0f;
        vp.maxDepth = 1.0f;

        VkRect2D scissor{{0, 0}, state_.swapchain().extent};

        state_.dispatch().cmdSetViewport(frame.command_buffer_, 0, 1, &vp);
        state_.dispatch().cmdSetScissor(frame.command_buffer_, 0, 1, &scissor);

        res = draw_commands(frame.command_buffer_);
        if (VK_SUCCESS != res) {
            LOG_ERROR("draw_commands returned %s", string_VkResult(res));
            return false;
        }

        state_.dispatch().cmdEndRenderPass(frame.command_buffer_);
        state_.dispatch().endCommandBuffer(frame.command_buffer_);

        // submitting the recorder buffer
        VkPipelineStageFlags wait_mask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;

        VkSubmitInfo submit_info{};
        submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit_info.pWaitDstStageMask = &wait_mask;
        submit_info.pWaitSemaphores = &frame.sem_image_avaliable_;
        submit_info.waitSemaphoreCount = 1;
        submit_info.pCommandBuffers = &frame.command_buffer_;
        submit_info.commandBufferCount = 1;
        submit_info.pSignalSemaphores = &frame.sem_render_done_;
        submit_info.signalSemaphoreCount = 1;

        state_.dispatch().queueSubmit(graphics_queue_, 1, &submit_info, frame.fence_in_flight_);

        VkPresentInfoKHR present_info{};
        present_info.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
        present_info.pWaitSemaphores = &frame.sem_render_done_;
        present_info.waitSemaphoreCount = 1;
        present_info.pImageIndices = &image_index;
        present_info.pSwapchains = &state_.swapchain().swapchain;
        present_info.swapchainCount = 1;

        // present
        {
            res = state_.dispatch().queuePresentKHR(present_queue_, &present_info);
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

    std::optional<Buffer> create_buffer(
        const VkBufferUsageFlags usage, const void *data, size_t byte_size, bool use_staging) const {
        VkResult res;

        if (use_staging && kStagingBufferSize < byte_size) {
            LOG_ERROR("requested device-only memory but staging buffer too small for the resource");
            return {};
        }

        VkBufferCreateInfo create_info{};
        create_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        create_info.size = byte_size;
        create_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        if (use_staging) {
            create_info.usage = usage | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        } else {
            create_info.usage = usage;
        }

        // vma allocation info
        VmaAllocationCreateInfo alloc_create_info{};
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
        VmaAllocationInfo alloc_info{};

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
            vmaUnmapMemory(state_.allocator(), buffer.allocation());

            res = vmaFlushAllocation(state_.allocator(), buffer.allocation(), 0, VK_WHOLE_SIZE);
            if (VK_SUCCESS != res) {
                LOG_ERROR("vmaFlushAllocation failure: %s", string_VkResult(res));
                return {};
            }
        } else {
            void *mapped_mem;
            res = vmaMapMemory(state_.allocator(), staging_buffer_.allocation(), &mapped_mem);
            if (VK_SUCCESS != res) {
                LOG_ERROR("cannot map staging buffer: %s", string_VkResult(res));
                return {};
            }

            memcpy(mapped_mem, data, byte_size);
            vmaUnmapMemory(state_.allocator(), staging_buffer_.allocation());

            // transfer command
            res = vmaFlushAllocation(state_.allocator(), buffer.allocation(), 0, VK_WHOLE_SIZE);
            if (VK_SUCCESS != res) {
                LOG_ERROR("vmaFlushAllocation failure: %s", string_VkResult(res));
                return {};
            }

            copy_buffer(staging_buffer_.buffer(), buffer.buffer(), byte_size);
        }

        return std::move(buffer);
    }

    bool copy_buffer(VkBuffer src, VkBuffer dst, VkDeviceSize size) const {
        VkResult res;
        res = state_.dispatch().resetCommandBuffer(upload_buffer_, 0);
        if (VK_SUCCESS != res) {
            LOG_ERROR("reset upload command buffer failure: %s", string_VkResult(res));
            return false;
        }

        VkCommandBufferBeginInfo begin_info{};
        begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        begin_info.pInheritanceInfo = 0;
        begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

        res = state_.dispatch().beginCommandBuffer(upload_buffer_, &begin_info);
        if (VK_SUCCESS != res) {
            LOG_ERROR("begin upload command buffer failure: %s", string_VkResult(res));
            return false;
        }

        VkBufferCopy copy_info{};
        copy_info.srcOffset = 0;
        copy_info.dstOffset = 0;
        copy_info.size = size;

        state_.dispatch().cmdCopyBuffer(upload_buffer_, src, dst, 1, &copy_info);
        res = state_.dispatch().endCommandBuffer(upload_buffer_);
        if (VK_SUCCESS != res) {
            LOG_ERROR("end upload command buffer failure: %s", string_VkResult(res));
            return false;
        }

        VkSubmitInfo submit_info{};
        submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit_info.pCommandBuffers = &upload_buffer_;
        submit_info.commandBufferCount = 1;

        res = state_.dispatch().queueSubmit(graphics_queue_, 1, &submit_info, VK_NULL_HANDLE);
        if (VK_SUCCESS != res) {
            LOG_ERROR("submit upload command buffer failure: %s", string_VkResult(res));
            return false;
        }

        res = state_.dispatch().deviceWaitIdle();
        if (VK_SUCCESS != res) {
            LOG_ERROR("wait device idle failure: %s", string_VkResult(res));
            return false;
        }

        return true;
    }

    static VkShaderModule shader_from_bytecode(ProgramState &state, const void *buffer, size_t size) {
        VkResult res;

        VkShaderModuleCreateInfo create_info{};
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

        VkAttachmentReference color_attachment_ref = {};
        color_attachment_ref.attachment = 0;
        color_attachment_ref.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

        VkSubpassDescription subpass = {};
        subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        subpass.colorAttachmentCount = 1;
        subpass.pColorAttachments = &color_attachment_ref;

        // ensure rendering does not start until image is available
        // this is actually not needed because drivers are required to automatically insert this
        VkSubpassDependency dependency = {};
        dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
        dependency.dstSubpass = 0;
        dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        dependency.srcAccessMask = 0;
        dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

        VkRenderPassCreateInfo render_pass_info = {};
        render_pass_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
        render_pass_info.attachmentCount = 1;
        render_pass_info.pAttachments = &color_attachment;
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

    static bool create_pipeline_layout(ProgramState &state, VkPipelineLayout *layout) {
        VkPipelineLayoutCreateInfo pipeline_layout_info = {};
        pipeline_layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipeline_layout_info.setLayoutCount = 0;
        pipeline_layout_info.pushConstantRangeCount = 0;

        VkResult res;
        res = state.dispatch().createPipelineLayout(&pipeline_layout_info, nullptr, layout);
        if (VK_SUCCESS != res) {
            *layout = VK_NULL_HANDLE;
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
        VkPipelineDynamicStateCreateInfo dynamic_state_desc{};
        dynamic_state_desc.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
        dynamic_state_desc.pDynamicStates = kDynamicStates.data();
        dynamic_state_desc.dynamicStateCount = static_cast<uint32_t>(kDynamicStates.size());

        // input state using the vertex struct
        constexpr std::array<VkVertexInputAttributeDescription, 3> kVertexAttribDesc = {
            VkVertexInputAttributeDescription{0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, position)},
            VkVertexInputAttributeDescription{1, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, normal)},
            VkVertexInputAttributeDescription{2, 0, VK_FORMAT_R32G32_SFLOAT, offsetof(Vertex, uv)},
        };

        VkVertexInputBindingDescription vertex_binding_desc{};
        vertex_binding_desc.binding = 0;
        vertex_binding_desc.stride = sizeof(Vertex);
        vertex_binding_desc.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

        VkPipelineVertexInputStateCreateInfo input_state_desc{};
        input_state_desc.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        input_state_desc.pVertexAttributeDescriptions = kVertexAttribDesc.data();
        input_state_desc.vertexAttributeDescriptionCount = static_cast<uint32_t>(kVertexAttribDesc.size());
        input_state_desc.pVertexBindingDescriptions = &vertex_binding_desc;
        input_state_desc.vertexBindingDescriptionCount = 1;

        // input assembly
        VkPipelineInputAssemblyStateCreateInfo assembly_desc{};
        assembly_desc.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        assembly_desc.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        assembly_desc.primitiveRestartEnable = false;

        // dynamic state
        VkPipelineViewportStateCreateInfo viewport_desc{};
        viewport_desc.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
        viewport_desc.viewportCount = 1;
        viewport_desc.scissorCount = 1;

        // rasterization
        VkPipelineRasterizationStateCreateInfo rasterizer_desc{};
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
        VkPipelineMultisampleStateCreateInfo multisample_desc{};
        multisample_desc.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        multisample_desc.sampleShadingEnable = false;
        multisample_desc.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
        multisample_desc.minSampleShading = 1.0f;
        multisample_desc.pSampleMask = nullptr;
        multisample_desc.alphaToCoverageEnable = false;
        multisample_desc.alphaToOneEnable = false;

        // no blending
        VkPipelineColorBlendAttachmentState blend_att_desc{};
        blend_att_desc.colorWriteMask =
            VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
        blend_att_desc.blendEnable = false;

        VkPipelineColorBlendStateCreateInfo blend_desc{};
        blend_desc.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        blend_desc.logicOpEnable = false;
        blend_desc.logicOp = VK_LOGIC_OP_COPY;
        blend_desc.blendConstants[0] = 0.0f;
        blend_desc.blendConstants[1] = 0.0f;
        blend_desc.blendConstants[2] = 0.0f;
        blend_desc.blendConstants[3] = 0.0f;
        blend_desc.pAttachments = &blend_att_desc;
        blend_desc.attachmentCount = 1;

        // finally, create the graphics pipeline
        VkGraphicsPipelineCreateInfo create_info{};
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

    static bool create_staging_buffer(ProgramState &state, SceneState &scene) {
        // initialize staging buffer
        VkBufferCreateInfo staging_buffer_desc{};
        staging_buffer_desc.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        staging_buffer_desc.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        staging_buffer_desc.size = kStagingBufferSize;
        staging_buffer_desc.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        VmaAllocationCreateInfo staging_alloc_desc{};
        staging_alloc_desc.usage = VMA_MEMORY_USAGE_AUTO;
        staging_alloc_desc.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;

        VmaAllocation allocation = VMA_NULL;
        VkBuffer vk_buffer = VK_NULL_HANDLE;
        VmaAllocationInfo alloc_info{};

        VkResult res = vmaCreateBuffer(
            state.allocator(), &staging_buffer_desc, &staging_alloc_desc, &vk_buffer, &allocation, &alloc_info);
        if (res != VK_SUCCESS) {
            LOG_ERROR("failed to allocate staging buffer: %s", string_VkResult(res));
            return false;
        }

        scene.staging_buffer_ = Buffer{state.allocator(), vk_buffer, allocation, alloc_info};

        // since we have the staging buffer, we would also like to get the upload command buffer
        VkCommandBufferAllocateInfo cmd_alloc_info{};
        cmd_alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        cmd_alloc_info.commandPool = scene.command_pool_;
        cmd_alloc_info.commandBufferCount = 1;
        cmd_alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;

        res = state.dispatch().allocateCommandBuffers(&cmd_alloc_info, &scene.upload_buffer_);
        if (VK_SUCCESS != res) {
            LOG_ERROR("failed to create upload command buffer: %s", string_VkResult(res));
            return false;
        }

        return true;
    }

    static bool create_command_pool(ProgramState &state, uint32_t family_index, VkCommandPool *command_pool) {
        VkCommandPoolCreateInfo create_info{};
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

    static bool create_frame_data(ProgramState &state, SceneState &scene, uint32_t frames_in_flight) {
        VkResult res;

        std::vector<VkCommandBuffer> command_buffers;
        command_buffers.resize(frames_in_flight);
        std::fill(command_buffers.begin(), command_buffers.end(), VK_NULL_HANDLE);

        VkCommandBufferAllocateInfo buffer_info{};
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

            VkSemaphoreCreateInfo sem_create_info{};
            sem_create_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

            VkFenceCreateInfo fence_create_info{};
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
        }

        return true;
    }

    static std::unique_ptr<SceneState> initialize(ProgramState &state) {
        std::unique_ptr<SceneState> scene{new SceneState(state)};

        // fetch queues
        auto gq = state.device().get_queue(vkb::QueueType::graphics);
        auto pq = state.device().get_queue(vkb::QueueType::present);

        if (!gq.has_value()) {
            LOG_ERROR("no graphics queue: %s", gq.error().message().c_str());
            return {};
        }

        if (!pq.has_value()) {
            LOG_ERROR("no present queue: %s", pq.error().message().c_str());
            return {};
        }

        scene->graphics_queue_ = gq.value();
        scene->present_queue_ = pq.value();
        LOG_INFO("obtained graphics and present queue");

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

        if (!create_pipeline_layout(state, &scene->pipeline_layout_)) {
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

        if (!create_frame_data(state, *scene, kFramesInFlight)) {
            LOG_ERROR("failed to create frame submission data");
            return {};
        }

        LOG_INFO("created frame submission data");

        if (!create_staging_buffer(state, *scene)) {
            LOG_ERROR("failed to create staging buffer");
            return {};
        }

        LOG_INFO("created staging buffer");

        return scene;
    }
};

struct VulkanSample final {
private:
    VulkanSample(ProgramState &state, SceneState &scene) : state{state}, scene{scene} {};

public:
    ProgramState &state;
    SceneState &scene;

    Geometry geometry;
    Buffer vertex_buffer;
    Buffer index_buffer;

    VulkanSample(const VulkanSample &) = delete;
    ~VulkanSample() {
        LOG_INFO("destroying sample state");

        VkResult res = state.dispatch().deviceWaitIdle();
        if (VK_SUCCESS != res) {
            LOG_ERROR("failed to wait device idle: %s", string_VkResult(res));
        }
    }

    VkResult record_queue(VkCommandBuffer command_buffer) {
        VkDeviceSize buf_offset = 0;
        state.dispatch().cmdBindVertexBuffers(command_buffer, 0, 1, vertex_buffer.addr_of(), &buf_offset);
        state.dispatch().cmdBindIndexBuffer(command_buffer, index_buffer.buffer(), 0, VK_INDEX_TYPE_UINT16);
        state.dispatch().cmdDrawIndexed(command_buffer, geometry.indices.size(), 1, 0, 0, 0);

        return VK_SUCCESS;
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
    std::vector<uint16_t>{
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
    std::vector<uint16_t>{
        0, 1, 2, 3, 4, 5
    }
};
        // clang-format on
    }

    static std::unique_ptr<VulkanSample> initialize(ProgramState &state, SceneState &scene) {
        std::unique_ptr<VulkanSample> sample{new VulkanSample(state, scene)};

        // create geometry and upload to a gram buffer
        auto geometry = cube_geometry();
        auto vertex_buffer = scene.create_buffer(VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, geometry.vertices.data(),
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

        auto index_buffer = scene.create_buffer(VK_BUFFER_USAGE_INDEX_BUFFER_BIT, geometry.indices.data(),
            sizeof(uint16_t) * std::size(geometry.indices), /* use staging buffer */ true);

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

        sample->geometry = std::move(geometry);
        sample->vertex_buffer = std::move(vertex_buffer.value());
        sample->index_buffer = std::move(index_buffer.value());

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

        if (!scene_state->draw_frame([&](VkCommandBuffer buffer) -> VkResult {
            // allow the sample to record its command queue
            return sample->record_queue(buffer);
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
