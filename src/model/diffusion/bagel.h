#ifndef __SD_MODEL_DIFFUSION_BAGEL_H__
#define __SD_MODEL_DIFFUSION_BAGEL_H__

#include "model/diffusion/dit.hpp"
#include "model/diffusion/model.hpp"
#include "runtime/denoiser.hpp"
#include "conditioning/conditioner.hpp"

#include <array>
#include <cstring>
#include <stdexcept>

namespace Bagel {

struct ExternalKVConditioner : public Conditioner {
    SDCondition get_learned_condition(int, const ConditionerParams&) override {
        LOG_ERROR("BAGEL requires imported KV conditioning");
        return {};
    }
    void get_param_tensors(std::map<std::string, ggml_tensor*>&) override {}
    void set_flash_attention_enabled(bool) override {}
};

struct BagelFlowDenoiser : public DiscreteFlowDenoiser {
    float sigma_to_t(float sigma) override { return sigma; }

    std::vector<float> get_sigmas(uint32_t steps, int, scheduler_t, SDVersion, const char* = nullptr) override {
        std::vector<float> sigmas(steps + 1, 0.f);
        for (uint32_t i = 0; i < steps; ++i) {
            sigmas[i] = time_snr_shift(shift, 1.f - float(i) / float(steps));
        }
        return sigmas;
    }
};

inline sd::Tensor<float> guide(float timestep, const sd::Tensor<float>& conditional,
                               const sd::Tensor<float>& without_text, const sd::Tensor<float>& without_image,
                               float text_scale, float image_scale) {
    if (timestep <= 0.4f || timestep > 1.f || text_scale <= 1.f) {
        return conditional;
    }
    if (without_text.empty()) {
        return {};
    }
    auto result = without_text + (conditional - without_text) * text_scale;
    if (image_scale > 1.f) {
        if (without_image.empty()) {
            return {};
        }
        result = without_image + (result - without_image) * image_scale;
    }
    double original_norm = 0, guided_norm = 0;
    for (size_t i = 0; i < conditional.values().size(); ++i) {
        original_norm += double(conditional.values()[i]) * conditional.values()[i];
        guided_norm += double(result.values()[i]) * result.values()[i];
    }
    const auto scale = std::min(1.0, std::sqrt(original_norm) / (std::sqrt(guided_norm) + 1e-8));
    return result * float(scale);
}

// The two boundary tokens use the understanding expert in every denoising layer.
// They are stored first here; all image-group tokens have the same RoPE position
// and attend bidirectionally, so this permutation does not change attention.
struct Model : public GGMLBlock {
    static constexpr int hidden_size = 3584;
    static constexpr int head_dim = 128;
    static constexpr int heads = 28;
    static constexpr int kv_heads = 4;
    static constexpr int layers = 28;

    std::map<std::string, ggml_tensor*> understanding;

    static bool generation_weight(const std::string& name) {
        return (starts_with(name, "language_model.") && name.find("_moe_gen") != std::string::npos) ||
               starts_with(name, "vae2llm.") || starts_with(name, "llm2vae.") ||
               starts_with(name, "time_embedder.") || starts_with(name, "latent_pos_embed.");
    }

    void init_params(ggml_context* ctx, const String2TensorStorage& storage,
                     const std::string prefix) override {
        constexpr const char* understanding_prefix = "text_encoders.llm.";
        for (const auto& entry : storage) {
            std::string name;
            bool understanding_weight = false;
            if (starts_with(entry.first, understanding_prefix)) {
                name = entry.first.substr(std::strlen(understanding_prefix));
                understanding_weight = true;
            } else if (starts_with(entry.first, prefix)) {
                name = entry.first.substr(prefix.size());
            } else {
                continue;
            }
            if (understanding_weight && !starts_with(name, "blk.") && name != "token_embd.weight") {
                continue;
            }
            if (!understanding_weight && !generation_weight(name)) {
                continue;
            }
            const auto& tensor = entry.second;
            const auto type = tensor.n_dims == 1 ? GGML_TYPE_F32 : get_type(entry.first, storage, tensor.type);
            const auto key = understanding_weight ? "understanding." + name : name;
            params[key] = ggml_new_tensor(ctx, type, tensor.n_dims, tensor.ne);
            if (understanding_weight) {
                understanding[name] = params[key];
            }
        }
    }

    void get_param_tensors(std::map<std::string, ggml_tensor*>& tensors,
                           const std::string& prefix = "") {
        for (const auto& [name, tensor] : params) {
            const bool is_understanding = starts_with(name, "understanding.");
            const auto local_name = is_understanding ? name.substr(std::strlen("understanding.")) : name;
            const auto storage_name = is_understanding ? "text_encoders.llm." + local_name : local_name;
            const auto full_name = prefix.empty() ? storage_name : prefix + "." + storage_name;
            tensors[full_name] = tensor;
            ggml_set_name(tensor, full_name.c_str());
        }
    }

    ggml_tensor* generated(const std::string& name) const {
        const auto it = params.find(name);
        GGML_ASSERT(it != params.end());
        return it->second;
    }

    void validate() const {
        auto require = [&](const std::string& name, int64_t rows, int64_t columns = 1) {
            const auto it = params.find(name);
            if (it == params.end() || it->second->ne[0] != rows || it->second->ne[1] != columns ||
                it->second->ne[2] != 1 || it->second->ne[3] != 1) {
                throw std::invalid_argument("Missing or incompatible BAGEL tensor: " + name);
            }
        };
        require("vae2llm.weight", 64, hidden_size);
        require("vae2llm.bias", hidden_size);
        require("llm2vae.weight", hidden_size, 64);
        require("llm2vae.bias", 64);
        require("latent_pos_embed.pos_embed", hidden_size, 4096);
        require("time_embedder.mlp.0.weight", 256, hidden_size);
        require("time_embedder.mlp.0.bias", hidden_size);
        require("time_embedder.mlp.2.weight", hidden_size, hidden_size);
        require("time_embedder.mlp.2.bias", hidden_size);
        require("language_model.model.norm_moe_gen.weight", hidden_size);
        for (int layer = 0; layer < layers; ++layer) {
            const auto p = "language_model.model.layers." + std::to_string(layer) + ".";
            require(p + "input_layernorm_moe_gen.weight", hidden_size);
            require(p + "post_attention_layernorm_moe_gen.weight", hidden_size);
            for (const auto proj : {"q", "k", "v"}) {
                const int width = std::string(proj) == "q" ? hidden_size : head_dim * kv_heads;
                require(p + "self_attn." + proj + "_proj_moe_gen.weight", hidden_size, width);
                require(p + "self_attn." + proj + "_proj_moe_gen.bias", width);
            }
            require(p + "self_attn.o_proj_moe_gen.weight", hidden_size, hidden_size);
            require(p + "self_attn.q_norm_moe_gen.weight", head_dim);
            require(p + "self_attn.k_norm_moe_gen.weight", head_dim);
            require(p + "mlp_moe_gen.gate_proj.weight", hidden_size, 18944);
            require(p + "mlp_moe_gen.up_proj.weight", hidden_size, 18944);
            require(p + "mlp_moe_gen.down_proj.weight", 18944, hidden_size);
        }
    }

    ggml_tensor* understood(const std::string& name) const {
        const auto it = understanding.find(name);
        GGML_ASSERT(it != understanding.end() && it->second != nullptr);
        return it->second;
    }

    static ggml_tensor* linear(ggml_context* ctx, ggml_tensor* x,
                               ggml_tensor* weight, ggml_tensor* bias = nullptr) {
        auto result = ggml_mul_mat(ctx, weight, x);
        return bias ? ggml_add(ctx, result, bias) : result;
    }

    static ggml_tensor* norm(ggml_context* ctx, ggml_tensor* x, ggml_tensor* weight) {
        return ggml_mul(ctx, ggml_rms_norm(ctx, x, 1e-6f), weight);
    }

    static ggml_tensor* slice(ggml_context* ctx, ggml_tensor* x, int dim, int64_t first, int64_t last) {
        return ggml_cont(ctx, ggml_ext_slice(ctx, x, dim, first, last));
    }

    ggml_tensor* mixed_norm(ggml_context* ctx, ggml_tensor* x, int dim,
                            const std::string& text_name, const std::string& image_name) const {
        auto text = norm(ctx, slice(ctx, x, dim, 0, 2), understood(text_name));
        auto image = norm(ctx, slice(ctx, x, dim, 2, x->ne[dim]), generated(image_name));
        return ggml_concat(ctx, text, image, dim);
    }

    ggml_tensor* mixed_linear(ggml_context* ctx, ggml_tensor* x,
                              const std::string& text_name, const std::string& image_name,
                              bool bias = false) const {
        auto text = linear(ctx, slice(ctx, x, 1, 0, 2), understood(text_name + ".weight"),
                           bias ? understood(text_name + ".bias") : nullptr);
        auto image = linear(ctx, slice(ctx, x, 1, 2, x->ne[1]), generated(image_name + ".weight"),
                            bias ? generated(image_name + ".bias") : nullptr);
        return ggml_concat(ctx, text, image, 1);
    }

    static ggml_tensor* rope(ggml_context* ctx, ggml_tensor* x, ggml_tensor* positions) {
        return ggml_rope_ext(ctx, x, positions, nullptr, head_dim, GGML_ROPE_TYPE_NEOX,
                             32768, 1000000.f, 1.f, 0.f, 1.f, 32.f, 1.f);
    }

    ggml_tensor* attention(GGMLRunnerContext* run, ggml_tensor* x, ggml_tensor* positions,
                           int layer, ggml_tensor* prefix_k, ggml_tensor* prefix_v,
                           const std::string& export_prefix) const {
        auto ctx = run->ggml_ctx;
        const auto text = "blk." + std::to_string(layer) + ".";
        const auto image = "language_model.model.layers." + std::to_string(layer) + ".self_attn.";
        const auto length = x->ne[1];
        auto q = mixed_linear(ctx, x, text + "attn_q", image + "q_proj_moe_gen", true);
        auto k = mixed_linear(ctx, x, text + "attn_k", image + "k_proj_moe_gen", true);
        auto v = mixed_linear(ctx, x, text + "attn_v", image + "v_proj_moe_gen", true);
        q = ggml_reshape_3d(ctx, q, head_dim, heads, length);
        k = ggml_reshape_3d(ctx, k, head_dim, kv_heads, length);
        v = ggml_reshape_3d(ctx, v, head_dim, kv_heads, length);
        q = mixed_norm(ctx, q, 2, text + "attn_q_norm.weight", image + "q_norm_moe_gen.weight");
        k = mixed_norm(ctx, k, 2, text + "attn_k_norm.weight", image + "k_norm_moe_gen.weight");
        q = rope(ctx, q, positions);
        k = rope(ctx, k, positions);
        GGML_ASSERT((prefix_k == nullptr) == (prefix_v == nullptr));
        if (prefix_k) {
            k = ggml_concat(ctx, prefix_k, k, 2);
            v = ggml_concat(ctx, prefix_v, v, 2);
        }
        if (!export_prefix.empty()) {
            run->persist_cache_tensor(export_prefix + std::to_string(layer) + ".k", k);
            run->persist_cache_tensor(export_prefix + std::to_string(layer) + ".v", v);
        }
        q = ggml_cont(ctx, ggml_permute(ctx, q, 0, 2, 1, 3));
        k = ggml_cont(ctx, ggml_permute(ctx, k, 0, 2, 1, 3));
        auto result = ggml_ext_attention_ext(ctx, run->backend, q, k, v, heads,
                                             nullptr, true, run->flash_attn_enabled);
        result = ggml_reshape_2d(ctx, result, hidden_size, length);
        return mixed_linear(ctx, result, text + "attn_output", image + "o_proj_moe_gen");
    }

    ggml_tensor* feed_forward(ggml_context* ctx, ggml_tensor* x, int layer) const {
        const auto text = "blk." + std::to_string(layer) + ".";
        const auto image = "language_model.model.layers." + std::to_string(layer) + ".mlp_moe_gen.";
        auto gate = mixed_linear(ctx, x, text + "ffn_gate", image + "gate_proj");
        auto up = mixed_linear(ctx, x, text + "ffn_up", image + "up_proj");
        auto activation = ggml_mul(ctx, ggml_silu(ctx, gate), up);
        return mixed_linear(ctx, activation, text + "ffn_down", image + "down_proj");
    }

    ggml_tensor* forward(GGMLRunnerContext* run, ggml_tensor* latent, ggml_tensor* timestep,
                         ggml_tensor* spatial_ids, ggml_tensor* rope_ids, ggml_tensor* boundary_embeddings,
                         const std::vector<ggml_tensor*>& prefix_keys,
                         const std::vector<ggml_tensor*>& prefix_values,
                         const std::string& export_prefix = "") const {
        auto ctx = run->ggml_ctx;
        GGML_ASSERT(latent->ne[2] == 16 && latent->ne[3] == 1);
        GGML_ASSERT(latent->ne[0] % 2 == 0 && latent->ne[1] % 2 == 0);
        GGML_ASSERT(prefix_keys.size() == layers && prefix_values.size() == layers);
        auto patches = DiT::patchify(ctx, latent, 2, 2, false);
        patches = ggml_reshape_2d(ctx, patches, 64, patches->ne[1]);
        auto image = linear(ctx, patches, generated("vae2llm.weight"), generated("vae2llm.bias"));
        auto time = ggml_ext_timestep_embedding(ctx, timestep, 256, 10000.f, 1.f);
        time = linear(ctx, time, generated("time_embedder.mlp.0.weight"), generated("time_embedder.mlp.0.bias"));
        time = linear(ctx, ggml_silu(ctx, time), generated("time_embedder.mlp.2.weight"), generated("time_embedder.mlp.2.bias"));
        image = ggml_add(ctx, image, time);
        image = ggml_add(ctx, image, ggml_get_rows(ctx, generated("latent_pos_embed.pos_embed"), spatial_ids));
        auto x = ggml_concat(ctx, boundary_embeddings, image, 1);
        for (int layer = 0; layer < layers; ++layer) {
            const auto text = "blk." + std::to_string(layer) + ".";
            const auto gen = "language_model.model.layers." + std::to_string(layer) + ".";
            auto normalized = mixed_norm(ctx, x, 1, text + "attn_norm.weight", gen + "input_layernorm_moe_gen.weight");
            x = ggml_add(ctx, x, attention(run, normalized, rope_ids, layer, prefix_keys[layer], prefix_values[layer], export_prefix));
            normalized = mixed_norm(ctx, x, 1, text + "ffn_norm.weight", gen + "post_attention_layernorm_moe_gen.weight");
            x = ggml_add(ctx, x, feed_forward(ctx, normalized, layer));
        }
        if (!export_prefix.empty()) return x;
        x = norm(ctx, slice(ctx, x, 1, 2, x->ne[1]), generated("language_model.model.norm_moe_gen.weight"));
        x = linear(ctx, x, generated("llm2vae.weight"), generated("llm2vae.bias"));
        return DiT::unpatchify(ctx, x, latent->ne[1] / 2, latent->ne[0] / 2, 2, 2, false);
    }
};

struct Runner : public DiffusionModelRunner {
    Model model;
    std::array<int32_t, 2> boundary_tokens{151652, 151653};
    std::array<int32_t, 3> next_positions{};
    std::array<std::vector<int32_t>, 3> prefix_tokens, prefix_positions;
    std::vector<ggml_tensor*> exported_keys, exported_values;
    std::array<size_t, 3> prefix_lengths{};
    std::array<bool, 3> prefix_ready{};
    int active_slot = -1;
    std::vector<int32_t> spatial_positions;
    std::vector<int32_t> image_positions;

    Runner(ggml_backend_t backend, const String2TensorStorage& storage,
            std::shared_ptr<RunnerWeightManager> manager)
        : DiffusionModelRunner(backend, "", manager) {
        model.init(params_ctx, storage);
        model.validate();
    }

    std::string get_desc() override { return "BAGEL-7B-MoT"; }

    void get_param_tensors(std::map<std::string, ggml_tensor*>& tensors,
                           const std::string& prefix) override {
        model.get_param_tensors(tensors, prefix);
    }

    bool supports_kv_prefix() const override { return true; }

    static std::string cache_name(int slot, int layer) {
        return "bagel.prefix." + std::to_string(slot) + "." + std::to_string(layer);
    }

    bool set_kv_prefix_slot(int threads, int slot, const sd_kv_prefix_t& data) override {
        if (slot < 0 || slot > 2 || data.token_count > 32768) {
            return false;
        }
        if (!data.token_count) {
            prefix_tokens[slot].clear();
            prefix_positions[slot].clear();
            prefix_lengths[slot] = 0;
            next_positions[slot] = 0;
            prefix_ready[slot] = true;
            return true;
        }
        if (data.layer_count != Model::layers || !data.keys || !data.values || !data.token_ids) {
            return false;
        }
        if (data.positions) {
            for (size_t i = 0; i < data.token_count; ++i) {
                if (data.positions[i] < 0 || (i && data.positions[i] < data.positions[i - 1])) return false;
            }
        }
        for (size_t layer = 0; layer < data.layer_count; ++layer) {
            for (auto tensor : {data.keys[layer], data.values[layer]}) {
                if (!tensor || !tensor->buffer || !ggml_is_contiguous(tensor) ||
                    tensor->ne[0] != Model::head_dim || tensor->ne[1] != Model::kv_heads ||
                    tensor->ne[2] != int64_t(data.token_count) || tensor->ne[3] != 1 ||
                    (tensor->type != GGML_TYPE_F32 && tensor->type != GGML_TYPE_F16 && tensor->type != GGML_TYPE_BF16)) {
                    return false;
                }
            }
        }
        int64_t next = data.positions ? int64_t(data.positions[data.token_count - 1]) + 1 : int64_t(data.token_count);
        if (next < 0 || next > 32768) {
            return false;
        }
        auto graph = [&]() {
            auto gf = new_graph_custom(16384);
            for (int layer = 0; layer < Model::layers; ++layer) {
                auto key = ggml_cast(compute_ctx, to_backend(data.keys[layer]), GGML_TYPE_F32);
                auto value = ggml_cast(compute_ctx, to_backend(data.values[layer]), GGML_TYPE_F32);
                ggml_set_output(key);
                ggml_set_output(value);
                cache(cache_name(slot, layer) + ".k", key);
                cache(cache_name(slot, layer) + ".v", value);
                ggml_build_forward_expand(gf, key);
                ggml_build_forward_expand(gf, value);
            }
            return gf;
        };
        prefix_ready[slot] = false;
        if (!GGMLRunner::compute<float>(graph, threads, false, true, true, true).has_value()) {
            return false;
        }
        prefix_lengths[slot] = data.token_count;
        prefix_tokens[slot].assign(data.token_ids, data.token_ids + data.token_count);
        prefix_positions[slot].resize(data.token_count);
        for (size_t i = 0; i < data.token_count; ++i) {
            prefix_positions[slot][i] = data.positions ? data.positions[i] : int32_t(i);
        }
        next_positions[slot] = next;
        prefix_ready[slot] = true;
        return true;
    }

    ggml_cgraph* build_graph(const DiffusionParams& input, int slot, bool export_prefix) {
        auto gf = new_graph_custom(32768);
        auto x = make_input(*input.x);
        auto t = make_input(*input.timesteps);
        const int64_t width = x->ne[0] / 2, height = x->ne[1] / 2;
        spatial_positions.resize(width * height);
        for (int64_t i = 0; i < width * height; ++i) {
            spatial_positions[i] = int32_t((i / width) * 64 + i % width);
        }
        image_positions.assign(width * height + 2, next_positions[slot]);
        auto spatial = ggml_new_tensor_1d(compute_ctx, GGML_TYPE_I32, spatial_positions.size());
        auto positions = ggml_new_tensor_1d(compute_ctx, GGML_TYPE_I32, image_positions.size());
        auto boundary_ids = ggml_new_tensor_1d(compute_ctx, GGML_TYPE_I32, boundary_tokens.size());
        set_backend_tensor_data(spatial, spatial_positions.data());
        set_backend_tensor_data(positions, image_positions.data());
        set_backend_tensor_data(boundary_ids, boundary_tokens.data());
        auto boundary_embeddings = ggml_get_rows(compute_ctx, model.understood("token_embd.weight"), boundary_ids);
        std::vector<ggml_tensor*> keys(Model::layers, nullptr), values(Model::layers, nullptr);
        if (prefix_lengths[slot]) {
            for (int layer = 0; layer < Model::layers; ++layer) {
                keys[layer] = get_cache_tensor_by_name(cache_name(slot, layer) + ".k");
                values[layer] = get_cache_tensor_by_name(cache_name(slot, layer) + ".v");
                GGML_ASSERT(keys[layer] && values[layer]);
            }
        }
        auto run = get_context();
        auto output = model.forward(&run, x, t, spatial, positions, boundary_embeddings, keys, values,
                                    export_prefix ? "bagel.prefix." + std::to_string(slot) + "." : "");
        ggml_build_forward_expand(gf, output);
        return gf;
    }

    bool supports_image_prefix(int width, int height) const override {
        return width >= 16 && height >= 16 && width <= 1024 && height <= 1024 && width % 16 == 0 && height % 16 == 0;
    }

    bool encode_image_prefix(int threads, int slot, const sd::Tensor<float>& latent, sd_kv_prefix_t& output) override {
        if (slot < 0 || slot > 2 || !prefix_ready[slot] || latent.dim() != 4 || latent.shape()[2] != 16 ||
            latent.shape()[3] != 1 || latent.shape()[0] < 2 || latent.shape()[1] < 2 ||
            latent.shape()[0] > 128 || latent.shape()[1] > 128 || latent.shape()[0] % 2 || latent.shape()[1] % 2) return false;
        const size_t added = latent.shape()[0] * latent.shape()[1] / 4 + 2;
        if (prefix_lengths[slot] + added > 32768) return false;
        sd::Tensor<float> timestep({1}, {0.f});
        DiffusionParams input;
        input.x = &latent;
        input.timesteps = &timestep;
        auto graph = [&]() { return build_graph(input, slot, true); };
        prefix_ready[slot] = false;
        if (!GGMLRunner::compute<float>(graph, threads, false, true, true, true).has_value()) return false;
        prefix_ready[slot] = true;
        auto& tokens = prefix_tokens[slot];
        const auto start = tokens.size();
        tokens.resize(start + added, -1);
        tokens[start] = 151652;
        tokens[start + 1] = 151653;
        prefix_positions[slot].insert(prefix_positions[slot].end(), added, next_positions[slot]++);
        prefix_lengths[slot] += added;
        exported_keys.resize(Model::layers);
        exported_values.resize(Model::layers);
        for (int layer = 0; layer < Model::layers; ++layer) {
            exported_keys[layer] = get_cache_tensor_by_name(cache_name(slot, layer) + ".k");
            exported_values[layer] = get_cache_tensor_by_name(cache_name(slot, layer) + ".v");
        }
        output = {tokens.data(), tokens.size(), exported_keys.data(), exported_values.data(),
                  exported_keys.size(), prefix_positions[slot].data()};
        return true;
    }

    void set_active_slot(int slot) { active_slot = slot; }

    sd::Tensor<float> compute(int threads, const DiffusionParams& input) override {
        const int slot = active_slot;
        if (slot < 0 || slot > 2 || !prefix_ready[slot] || !input.x || !input.timesteps) {
            LOG_ERROR("BAGEL requires an initialized prefix slot and latent inputs");
            return {};
        }
        const auto& shape = input.x->shape();
        if (input.x->dim() != 4 || shape[2] != 16 || shape[3] != 1 ||
            shape[0] < 2 || shape[1] < 2 || shape[0] > 128 || shape[1] > 128 || shape[0] % 2 || shape[1] % 2) {
            LOG_ERROR("BAGEL requires one 16-channel latent with even dimensions up to 128");
            return {};
        }
        auto graph = [&]() { return build_graph(input, slot, false); };
        return restore_trailing_singleton_dims(
            GGMLRunner::compute<float>(graph, threads, false, false, false), input.x->dim());
    }
};

}  // namespace Bagel

#endif  // __SD_MODEL_DIFFUSION_BAGEL_H__
