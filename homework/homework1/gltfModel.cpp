#include "gltfModel.h"

void VulkanglTFModel::loadImages(tinygltf::Model &input) {
    // Images can be stored inside the glTF (which is the case for the sample model), so instead of directly
    // loading them from disk, we fetch them from the glTF loader and upload the buffers
    images.resize(input.images.size());
    for (size_t i = 0; i < input.images.size(); i++) {
        tinygltf::Image& glTFImage = input.images[i];
        // Get the image data from the glTF loader
        unsigned char* buffer = nullptr;
        VkDeviceSize bufferSize = 0;
        bool deleteBuffer = false;
        // We convert RGB-only images to RGBA, as most devices don't support RGB-formats in Vulkan
        if (glTFImage.component == 3) {
            bufferSize = glTFImage.width * glTFImage.height * 4;
            buffer = new unsigned char[bufferSize];
            unsigned char* rgba = buffer;
            unsigned char* rgb = &glTFImage.image[0];
            for (size_t i = 0; i < glTFImage.width * glTFImage.height; ++i) {
                memcpy(rgba, rgb, sizeof(unsigned char) * 3);
                rgba += 4;
                rgb += 3;
            }
            deleteBuffer = true;
        }
        else {
            buffer = &glTFImage.image[0];
            bufferSize = glTFImage.image.size();
        }
        // Load texture from image buffer
        images[i].texture.fromBuffer(buffer, bufferSize, VK_FORMAT_R8G8B8A8_UNORM, glTFImage.width, glTFImage.height, vulkanDevice, copyQueue);
        if (deleteBuffer) {
            delete[] buffer;
        }
    }
}

void VulkanglTFModel::loadTextures(tinygltf::Model &input) {
    textures.resize(input.textures.size());
    for (size_t i = 0; i < input.textures.size(); i++) {
        textures[i].imageIndex = input.textures[i].source;
    }
}

void VulkanglTFModel::loadMaterials(tinygltf::Model &input) {
    materials.resize(input.materials.size());
    for (size_t i = 0; i < input.materials.size(); i++) {
        // We only read the most basic properties required for our sample
        tinygltf::Material glTFMaterial = input.materials[i];
        // Get the base color factor
        if (glTFMaterial.values.find("baseColorFactor") != glTFMaterial.values.end()) {
            materials[i].baseColorFactor = glm::make_vec4(glTFMaterial.values["baseColorFactor"].ColorFactor().data());
        }
        // Get base color texture index
        if (glTFMaterial.values.find("baseColorTexture") != glTFMaterial.values.end()) {
            materials[i].baseColorTextureIndex = glTFMaterial.values["baseColorTexture"].TextureIndex();
        }
    }
}

void
VulkanglTFModel::loadNode(const tinygltf::Node &inputNode, const tinygltf::Model &input, VulkanglTFModel::Node *parent,
                          uint32_t nodeIndex, std::vector<uint32_t> &indexBuffer,
                          std::vector<VulkanglTFModel::Vertex> &vertexBuffer) {
    VulkanglTFModel::Node *node = new VulkanglTFModel::Node{};
    node->parent                = parent;
    node->matrix                = glm::mat4(1.0f);
    node->index                 = nodeIndex;
    node->skin                  = inputNode.skin;

    // Get the local node matrix
    // It's either made up from translation, rotation, scale or a 4x4 matrix
    if (inputNode.translation.size() == 3)
    {
        node->translation = glm::make_vec3(inputNode.translation.data());
    }
    if (inputNode.rotation.size() == 4)
    {
        glm::quat q    = glm::make_quat(inputNode.rotation.data());
        node->rotation = glm::mat4(q);
    }
    if (inputNode.scale.size() == 3)
    {
        node->scale = glm::make_vec3(inputNode.scale.data());
    }
    if (inputNode.matrix.size() == 16)
    {
        node->matrix = glm::make_mat4x4(inputNode.matrix.data());
    };

    // Load node's children
    if (inputNode.children.size() > 0)
    {
        for (size_t i = 0; i < inputNode.children.size(); i++)
        {
            loadNode(input.nodes[inputNode.children[i]], input, node, inputNode.children[i], indexBuffer, vertexBuffer);
        }
    }

    // If the node contains mesh data, we load vertices and indices from the buffers
    // In glTF this is done via accessors and buffer views
    if (inputNode.mesh > -1)
    {
        const tinygltf::Mesh mesh = input.meshes[inputNode.mesh];
        // Iterate through all primitives of this node's mesh
        for (size_t i = 0; i < mesh.primitives.size(); i++)
        {
            const tinygltf::Primitive &glTFPrimitive = mesh.primitives[i];
            uint32_t                   firstIndex    = static_cast<uint32_t>(indexBuffer.size());
            uint32_t                   vertexStart   = static_cast<uint32_t>(vertexBuffer.size());
            uint32_t                   indexCount    = 0;
            bool                       hasSkin       = false;
            // Vertices
            {
                const float *   positionBuffer     = nullptr;
                const float *   normalsBuffer      = nullptr;
                const float *   texCoordsBuffer    = nullptr;
                const uint16_t *jointIndicesBuffer = nullptr;
                const float *   jointWeightsBuffer = nullptr;
                size_t          vertexCount        = 0;

                // Get buffer data for vertex normals
                if (glTFPrimitive.attributes.find("POSITION") != glTFPrimitive.attributes.end())
                {
                    const tinygltf::Accessor &  accessor = input.accessors[glTFPrimitive.attributes.find("POSITION")->second];
                    const tinygltf::BufferView &view     = input.bufferViews[accessor.bufferView];
                    positionBuffer                       = reinterpret_cast<const float *>(&(input.buffers[view.buffer].data[accessor.byteOffset + view.byteOffset]));
                    vertexCount                          = accessor.count;
                }
                // Get buffer data for vertex normals
                if (glTFPrimitive.attributes.find("NORMAL") != glTFPrimitive.attributes.end())
                {
                    const tinygltf::Accessor &  accessor = input.accessors[glTFPrimitive.attributes.find("NORMAL")->second];
                    const tinygltf::BufferView &view     = input.bufferViews[accessor.bufferView];
                    normalsBuffer                        = reinterpret_cast<const float *>(&(input.buffers[view.buffer].data[accessor.byteOffset + view.byteOffset]));
                }
                // Get buffer data for vertex texture coordinates
                // glTF supports multiple sets, we only load the first one
                if (glTFPrimitive.attributes.find("TEXCOORD_0") != glTFPrimitive.attributes.end())
                {
                    const tinygltf::Accessor &  accessor = input.accessors[glTFPrimitive.attributes.find("TEXCOORD_0")->second];
                    const tinygltf::BufferView &view     = input.bufferViews[accessor.bufferView];
                    texCoordsBuffer                      = reinterpret_cast<const float *>(&(input.buffers[view.buffer].data[accessor.byteOffset + view.byteOffset]));
                }

                // POI: Get buffer data required for vertex skinning
                // Get vertex joint indices
                if (glTFPrimitive.attributes.find("JOINTS_0") != glTFPrimitive.attributes.end())
                {
                    const tinygltf::Accessor &  accessor = input.accessors[glTFPrimitive.attributes.find("JOINTS_0")->second];
                    const tinygltf::BufferView &view     = input.bufferViews[accessor.bufferView];
                    jointIndicesBuffer                   = reinterpret_cast<const uint16_t *>(&(input.buffers[view.buffer].data[accessor.byteOffset + view.byteOffset]));
                }
                // Get vertex joint weights
                if (glTFPrimitive.attributes.find("WEIGHTS_0") != glTFPrimitive.attributes.end())
                {
                    const tinygltf::Accessor &  accessor = input.accessors[glTFPrimitive.attributes.find("WEIGHTS_0")->second];
                    const tinygltf::BufferView &view     = input.bufferViews[accessor.bufferView];
                    jointWeightsBuffer                   = reinterpret_cast<const float *>(&(input.buffers[view.buffer].data[accessor.byteOffset + view.byteOffset]));
                }

                hasSkin = (jointIndicesBuffer && jointWeightsBuffer);

                // Append data to model's vertex buffer
                for (size_t v = 0; v < vertexCount; v++)
                {
                    Vertex vert{};
                    vert.pos          = glm::vec4(glm::make_vec3(&positionBuffer[v * 3]), 1.0f);
                    vert.normal       = glm::normalize(glm::vec3(normalsBuffer ? glm::make_vec3(&normalsBuffer[v * 3]) : glm::vec3(0.0f)));
                    vert.uv           = texCoordsBuffer ? glm::make_vec2(&texCoordsBuffer[v * 2]) : glm::vec3(0.0f);
                    vert.color        = glm::vec3(1.0f);
                    vert.jointIndices = hasSkin ? glm::vec4(glm::make_vec4(&jointIndicesBuffer[v * 4])) : glm::vec4(0.0f);
                    vert.jointWeights = hasSkin ? glm::make_vec4(&jointWeightsBuffer[v * 4]) : glm::vec4(0.0f);
                    vertexBuffer.push_back(vert);
                }
            }
            // Indices
            {
                const tinygltf::Accessor &  accessor   = input.accessors[glTFPrimitive.indices];
                const tinygltf::BufferView &bufferView = input.bufferViews[accessor.bufferView];
                const tinygltf::Buffer &    buffer     = input.buffers[bufferView.buffer];

                indexCount += static_cast<uint32_t>(accessor.count);

                // glTF supports different component types of indices
                switch (accessor.componentType)
                {
                    case TINYGLTF_PARAMETER_TYPE_UNSIGNED_INT: {
                        const uint32_t* buf = reinterpret_cast<const uint32_t*>(&buffer.data[accessor.byteOffset + bufferView.byteOffset]);
                        for (size_t index = 0; index < accessor.count; index++)
                        {
                            indexBuffer.push_back(buf[index] + vertexStart);
                        }
                        break;
                    }
                    case TINYGLTF_PARAMETER_TYPE_UNSIGNED_SHORT: {
                        const uint16_t* buf = reinterpret_cast<const uint16_t*>(&buffer.data[accessor.byteOffset + bufferView.byteOffset]);
                        for (size_t index = 0; index < accessor.count; index++)
                        {
                            indexBuffer.push_back(buf[index] + vertexStart);
                        }
                        break;
                    }
                    case TINYGLTF_PARAMETER_TYPE_UNSIGNED_BYTE: {
                        const uint8_t* buf = reinterpret_cast<const uint8_t*>(&buffer.data[accessor.byteOffset + bufferView.byteOffset]);
                        for (size_t index = 0; index < accessor.count; index++)
                        {
                            indexBuffer.push_back(buf[index] + vertexStart);
                        }
                        break;
                    }
                    default:
                        std::cerr << "Index component type " << accessor.componentType << " not supported!" << std::endl;
                        return;
                }
            }
            Primitive primitive{};
            primitive.firstIndex    = firstIndex;
            primitive.indexCount    = indexCount;
            primitive.materialIndex = glTFPrimitive.material;
            node->mesh.primitives.push_back(primitive);
        }
    }

    if (parent)
    {
        parent->children.push_back(node);
    }
    else
    {
        nodes.push_back(node);
    }
}

VulkanglTFModel::Node *VulkanglTFModel::findNode(VulkanglTFModel::Node *parent, uint32_t index) {
    Node *nodeFound = nullptr;
    if (parent->index == index)
    {
        return parent;
    }
    for (auto &child : parent->children)
    {
        nodeFound = findNode(child, index);
        if (nodeFound)
        {
            break;
        }
    }
    return nodeFound;
}

VulkanglTFModel::Node *VulkanglTFModel::nodeFromIndex(uint32_t index) {
    Node *nodeFound = nullptr;
    for (auto &node : nodes)
    {
        nodeFound = findNode(node, index);
        if (nodeFound)
        {
            break;
        }
    }
    return nodeFound;
}

void VulkanglTFModel::loadSkins(tinygltf::Model &input) {
    skins.resize(input.skins.size());

    for (size_t i = 0; i < input.skins.size(); i++)
    {
        tinygltf::Skin glTFSkin = input.skins[i];

        skins[i].name = glTFSkin.name;
        // Find the root node of the skeleton
        skins[i].skeletonRoot = nodeFromIndex(glTFSkin.skeleton);

        // Find joint nodes
        for (int jointIndex : glTFSkin.joints)
        {
            Node *node = nodeFromIndex(jointIndex);
            if (node)
            {
                skins[i].joints.push_back(node);
            }
        }

        // Get the inverse bind matrices from the buffer associated to this skin
        if (glTFSkin.inverseBindMatrices > -1)
        {
            const tinygltf::Accessor &  accessor   = input.accessors[glTFSkin.inverseBindMatrices];
            const tinygltf::BufferView &bufferView = input.bufferViews[accessor.bufferView];
            const tinygltf::Buffer &    buffer     = input.buffers[bufferView.buffer];
            skins[i].inverseBindMatrices.resize(accessor.count);
            memcpy(skins[i].inverseBindMatrices.data(), &buffer.data[accessor.byteOffset + bufferView.byteOffset], accessor.count * sizeof(glm::mat4));

            // Store inverse bind matrices for this skin in a shader storage buffer object
            // To keep this sample simple, we create a host visible shader storage buffer
            VK_CHECK_RESULT(vulkanDevice->createBuffer(
                    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                    &skins[i].ssbo,
                    sizeof(glm::mat4) * skins[i].inverseBindMatrices.size(),
                    skins[i].inverseBindMatrices.data()));
            VK_CHECK_RESULT(skins[i].ssbo.map());
        }
    }
}

void VulkanglTFModel::loadAnimations(tinygltf::Model &input) {
    animations.resize(input.animations.size());

    for (size_t i = 0; i < input.animations.size(); i++)
    {
        tinygltf::Animation glTFAnimation = input.animations[i];
        animations[i].name                = glTFAnimation.name;

        // Samplers
        animations[i].samplers.resize(glTFAnimation.samplers.size());
        for (size_t j = 0; j < glTFAnimation.samplers.size(); j++)
        {
            tinygltf::AnimationSampler glTFSampler = glTFAnimation.samplers[j];
            AnimationSampler &         dstSampler  = animations[i].samplers[j];
            dstSampler.interpolation               = glTFSampler.interpolation;

            // Read sampler keyframe input time values
            {
                const tinygltf::Accessor &  accessor   = input.accessors[glTFSampler.input];
                const tinygltf::BufferView &bufferView = input.bufferViews[accessor.bufferView];
                const tinygltf::Buffer &    buffer     = input.buffers[bufferView.buffer];
                const void *                dataPtr    = &buffer.data[accessor.byteOffset + bufferView.byteOffset];
                const float *               buf        = static_cast<const float *>(dataPtr);
                for (size_t index = 0; index < accessor.count; index++)
                {
                    dstSampler.inputs.push_back(buf[index]);
                }
                // Adjust animation's start and end times
                for (auto input : animations[i].samplers[j].inputs)
                {
                    if (input < animations[i].start)
                    {
                        animations[i].start = input;
                    };
                    if (input > animations[i].end)
                    {
                        animations[i].end = input;
                    }
                }
            }

            // Read sampler keyframe output translate/rotate/scale values
            {
                const tinygltf::Accessor &  accessor   = input.accessors[glTFSampler.output];
                const tinygltf::BufferView &bufferView = input.bufferViews[accessor.bufferView];
                const tinygltf::Buffer &    buffer     = input.buffers[bufferView.buffer];
                const void *                dataPtr    = &buffer.data[accessor.byteOffset + bufferView.byteOffset];
                switch (accessor.type)
                {
                    case TINYGLTF_TYPE_VEC3: {
                        const glm::vec3 *buf = static_cast<const glm::vec3 *>(dataPtr);
                        for (size_t index = 0; index < accessor.count; index++)
                        {
                            dstSampler.outputsVec4.push_back(glm::vec4(buf[index], 0.0f));
                        }
                        break;
                    }
                    case TINYGLTF_TYPE_VEC4: {
                        const glm::vec4 *buf = static_cast<const glm::vec4 *>(dataPtr);
                        for (size_t index = 0; index < accessor.count; index++)
                        {
                            dstSampler.outputsVec4.push_back(buf[index]);
                        }
                        break;
                    }
                    default: {
                        std::cout << "unknown type" << std::endl;
                        break;
                    }
                }
            }
        }

        // Channels
        animations[i].channels.resize(glTFAnimation.channels.size());
        for (size_t j = 0; j < glTFAnimation.channels.size(); j++)
        {
            tinygltf::AnimationChannel glTFChannel = glTFAnimation.channels[j];
            AnimationChannel &         dstChannel  = animations[i].channels[j];
            dstChannel.path                        = glTFChannel.target_path;
            dstChannel.samplerIndex                = glTFChannel.sampler;
            dstChannel.node                        = nodeFromIndex(glTFChannel.target_node);
        }
    }
}

glm::mat4 VulkanglTFModel::getNodeMatrix(VulkanglTFModel::Node *node) {
    glm::mat4              nodeMatrix    = node->getLocalMatrix();
    VulkanglTFModel::Node *currentParent = node->parent;
    while (currentParent)
    {
        nodeMatrix    = currentParent->getLocalMatrix() * nodeMatrix;
        currentParent = currentParent->parent;
    }
    return nodeMatrix;
}

void VulkanglTFModel::updateJoints(VulkanglTFModel::Node *node) {
    if (node->skin > -1)
    {
        // Update the joint matrices
        glm::mat4              inverseTransform = glm::inverse(getNodeMatrix(node));
        Skin                   skin             = skins[node->skin];
        size_t                 numJoints        = (uint32_t) skin.joints.size();
        std::vector<glm::mat4> jointMatrices(numJoints);
        for (size_t i = 0; i < numJoints; i++)
        {
            jointMatrices[i] = getNodeMatrix(skin.joints[i]) * skin.inverseBindMatrices[i];
            jointMatrices[i] = inverseTransform * jointMatrices[i];
        }
        // Update ssbo
        skin.ssbo.copyTo(jointMatrices.data(), jointMatrices.size() * sizeof(glm::mat4));
    }

    for (auto &child : node->children)
    {
        updateJoints(child);
    }
}

void VulkanglTFModel::updateAnimation(float deltaTime) {
    if (activeAnimation > static_cast<uint32_t>(animations.size()) - 1)
    {
        std::cout << "No animation with index " << activeAnimation << std::endl;
        return;
    }
    Animation &animation = animations[activeAnimation];
    animation.currentTime += deltaTime;
    if (animation.currentTime > animation.end)
    {
        animation.currentTime -= animation.end;
    }

    for (auto &channel : animation.channels)
    {
        AnimationSampler &sampler = animation.samplers[channel.samplerIndex];
        for (size_t i = 0; i < sampler.inputs.size() - 1; i++)
        {
            if (sampler.interpolation != "LINEAR")
            {
                std::cout << "This sample only supports linear interpolations\n";
                continue;
            }

            // Get the input keyframe values for the current time stamp
            if ((animation.currentTime >= sampler.inputs[i]) && (animation.currentTime <= sampler.inputs[i + 1]))
            {
                float a = (animation.currentTime - sampler.inputs[i]) / (sampler.inputs[i + 1] - sampler.inputs[i]);
                if (channel.path == "translation")
                {
                    channel.node->translation = glm::mix(sampler.outputsVec4[i], sampler.outputsVec4[i + 1], a);
                }
                if (channel.path == "rotation")
                {
                    glm::quat q1;
                    q1.x = sampler.outputsVec4[i].x;
                    q1.y = sampler.outputsVec4[i].y;
                    q1.z = sampler.outputsVec4[i].z;
                    q1.w = sampler.outputsVec4[i].w;

                    glm::quat q2;
                    q2.x = sampler.outputsVec4[i + 1].x;
                    q2.y = sampler.outputsVec4[i + 1].y;
                    q2.z = sampler.outputsVec4[i + 1].z;
                    q2.w = sampler.outputsVec4[i + 1].w;

                    channel.node->rotation = glm::normalize(glm::slerp(q1, q2, a));
                }
                if (channel.path == "scale")
                {
                    channel.node->scale = glm::mix(sampler.outputsVec4[i], sampler.outputsVec4[i + 1], a);
                }
            }
        }
    }
    for (auto &node : nodes)
    {
        updateJoints(node);
    }
}

void
VulkanglTFModel::drawNode(VkCommandBuffer commandBuffer, VkPipelineLayout pipelineLayout, VulkanglTFModel::Node *node) {
    if (node->mesh.primitives.size() > 0) {
        // Pass the node's matrix via push constants
        // Traverse the node hierarchy to the top-most parent to get the final matrix of the current node
        glm::mat4 nodeMatrix = node->matrix;
        VulkanglTFModel::Node* currentParent = node->parent;
        while (currentParent) {
            nodeMatrix = currentParent->matrix * nodeMatrix;
            currentParent = currentParent->parent;
        }
        // Pass the final matrix to the vertex shader using push constants
        vkCmdPushConstants(commandBuffer, pipelineLayout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(glm::mat4), &nodeMatrix);
        if(node->skin != -1)
        {
            vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 1, 1,
                                    &skins[node->skin].descriptorSet, 0, nullptr);
        }
        for (VulkanglTFModel::Primitive& primitive : node->mesh.primitives) {
            if (primitive.indexCount > 0) {
                // Get the texture index for this primitive
                VulkanglTFModel::Texture texture = textures[materials[primitive.materialIndex].baseColorTextureIndex];
                // Bind the descriptor for the current primitive's texture
                vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 1, 1, &images[texture.imageIndex].descriptorSet, 0, nullptr);
                vkCmdDrawIndexed(commandBuffer, primitive.indexCount, 1, primitive.firstIndex, 0, 0);
            }
        }
    }
    for (auto& child : node->children) {
        drawNode(commandBuffer, pipelineLayout, child);
    }
}

void VulkanglTFModel::draw(VkCommandBuffer commandBuffer, VkPipelineLayout pipelineLayout) {
    // All vertices and indices are stored in single buffers, so we only need to bind once
    VkDeviceSize offsets[1] = { 0 };
    vkCmdBindVertexBuffers(commandBuffer, 0, 1, &vertices.buffer, offsets);
    vkCmdBindIndexBuffer(commandBuffer, indices.buffer, 0, VK_INDEX_TYPE_UINT32);
    // Render all nodes at top-level
    for (auto& node : nodes) {
        drawNode(commandBuffer, pipelineLayout, node);
    }
}
