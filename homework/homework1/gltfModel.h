#pragma once

#define TINYGLTF_NO_STB_IMAGE_WRITE
#include "tiny_gltf.h"
#include "vulkanexamplebase.h"

class VulkanglTFModel
{
public:
    // The class requires some Vulkan objects so it can create it's own resources
    vks::VulkanDevice* vulkanDevice;
    VkQueue copyQueue;

    // The vertex layout for the samples' model
    struct Vertex {
        glm::vec3 pos;
        glm::vec3 normal;
        glm::vec2 uv;
        glm::vec3 color;
        glm::vec4 jointIndices;
        glm::vec4 jointWeights;
    };

    // Single vertex buffer for all primitives
    struct {
        VkBuffer buffer;
        VkDeviceMemory memory;
    } vertices;

    // Single index buffer for all primitives
    struct {
        int count;
        VkBuffer buffer;
        VkDeviceMemory memory;
    } indices;

    // The following structures roughly represent the glTF scene structure
    // To keep things simple, they only contain those properties that are required for this sample
    struct Node;

    // A primitive contains the data for a single draw call
    struct Primitive {
        uint32_t firstIndex;
        uint32_t indexCount;
        int32_t materialIndex;
    };

    // Contains the node's (optional) geometry and can be made up of an arbitrary number of primitives
    struct Mesh {
        std::vector<Primitive> primitives;
    };

    // A node represents an object in the glTF scene graph
    struct Node {
        Node* parent;
        uint32_t            index;
        std::vector<Node*> children;
        Mesh mesh;
        glm::vec3           translation{};
        glm::vec3           scale{1.0f};
        glm::quat           rotation{};
        int32_t             skin = -1;
        glm::mat4 matrix;
        glm::mat4 getLocalMatrix()
        {
            return glm::translate(glm::mat4(1.0f), translation) * glm::mat4(rotation) * glm::scale(glm::mat4(1.0f), scale) * matrix;
        }
        ~Node() {
            for (auto& child : children) {
                delete child;
            }
        }
    };

    // A glTF material stores information in e.g. the texture that is attached to it and colors
    struct Material {
        glm::vec4 baseColorFactor = glm::vec4(1.0f);
        uint32_t baseColorTextureIndex;
    };

    // Contains the texture for a single glTF image
    // Images may be reused by texture objects and are as such separated
    struct Image {
        vks::Texture2D texture;
        // We also store (and create) a descriptor set that's used to access this texture from the fragment shader
        VkDescriptorSet descriptorSet;
    };

    // A glTF texture stores a reference to the image and a sampler
    // In this sample, we are only interested in the image
    struct Texture {
        int32_t imageIndex;
    };

    struct Skin
    {
        std::string            name;
        Node *                 skeletonRoot = nullptr;
        std::vector<glm::mat4> inverseBindMatrices;
        std::vector<Node *>    joints;
        vks::Buffer            ssbo;
        VkDescriptorSet        descriptorSet;
    };

    struct AnimationSampler
    {
        std::string            interpolation;
        std::vector<float>     inputs;
        std::vector<glm::vec4> outputsVec4;
    };

    struct AnimationChannel
    {
        std::string path;
        Node *      node;
        uint32_t    samplerIndex;
    };

    struct Animation
    {
        std::string                   name;
        std::vector<AnimationSampler> samplers;
        std::vector<AnimationChannel> channels;
        float                         start       = std::numeric_limits<float>::max();
        float                         end         = std::numeric_limits<float>::min();
        float                         currentTime = 0.0f;
    };

    /*
        Model data
    */
    std::vector<Image> images;
    std::vector<Texture> textures;
    std::vector<Material> materials;
    std::vector<Node*> nodes;
    std::vector<Skin>      skins;
    std::vector<Animation> animations;

    uint32_t activeAnimation = 0;

    ~VulkanglTFModel()
    {
        for (auto node : nodes) {
            delete node;
        }
        // Release all Vulkan resources allocated for the model
        vkDestroyBuffer(vulkanDevice->logicalDevice, vertices.buffer, nullptr);
        vkFreeMemory(vulkanDevice->logicalDevice, vertices.memory, nullptr);
        vkDestroyBuffer(vulkanDevice->logicalDevice, indices.buffer, nullptr);
        vkFreeMemory(vulkanDevice->logicalDevice, indices.memory, nullptr);
        for (Image image : images) {
            vkDestroyImageView(vulkanDevice->logicalDevice, image.texture.view, nullptr);
            vkDestroyImage(vulkanDevice->logicalDevice, image.texture.image, nullptr);
            vkDestroySampler(vulkanDevice->logicalDevice, image.texture.sampler, nullptr);
            vkFreeMemory(vulkanDevice->logicalDevice, image.texture.deviceMemory, nullptr);
        }
    }

    void loadImages(tinygltf::Model& input);
    void loadTextures(tinygltf::Model& input);
    void loadMaterials(tinygltf::Model& input);
    void loadNode(const tinygltf::Node &inputNode, const tinygltf::Model &input, VulkanglTFModel::Node *parent, uint32_t nodeIndex, std::vector<uint32_t> &indexBuffer, std::vector<VulkanglTFModel::Vertex> &vertexBuffer);
    VulkanglTFModel::Node *findNode(Node *parent, uint32_t index);
    VulkanglTFModel::Node *nodeFromIndex(uint32_t index);
    void loadSkins(tinygltf::Model& input);
    void loadAnimations(tinygltf::Model &input);
    glm::mat4 getNodeMatrix(VulkanglTFModel::Node *node);
    void updateJoints(VulkanglTFModel::Node *node);
    void updateAnimation(float deltaTime);
    void drawNode(VkCommandBuffer commandBuffer, VkPipelineLayout pipelineLayout, VulkanglTFModel::Node* node);
    void draw(VkCommandBuffer commandBuffer, VkPipelineLayout pipelineLayout);

};