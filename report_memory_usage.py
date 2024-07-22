# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

"""Computes theoretical memory footprint for model training."""


import math

NUM_BYTES_IN_MEGABYTE = 1024 * 1024


def compute_weight_and_optimizer_memory(args, verbose=False):
    # Attention projection size.
    query_projection_size = args.kv_channels * args.num_attention_heads
    query_projection_to_hidden_size_ratio = query_projection_size / args.hidden_size
    # Group Query Attention.
    args.num_query_groups = args.num_attention_heads
    # MoE.
    gated_linear_multiplier = 3 / 2 if args.swiglu else 1
    num_sparse_layers = args.num_layers - 1
    num_dense_layers = 1

    # Number of parameters in transformer layers.
    num_parameters_in_transformer_dense_layers = (
        2
        * num_dense_layers
        * args.hidden_size
        * args.hidden_size
        * (
            # Attention.
            (
                (1 + (args.num_query_groups / args.num_attention_heads))
                * query_projection_to_hidden_size_ratio
            )
            # MLP.
            + ((args.ffn_hidden_size / args.hidden_size) * gated_linear_multiplier)
            # Transformer layernorms.
            + (2 / args.hidden_size)
            # Final layernorm.
            + (1 / (num_dense_layers * args.hidden_size))
        )
    )
    num_parameters_in_transformer_sparse_layers = (
        2
        * num_sparse_layers
        * args.hidden_size
        * args.hidden_size
        * (
            # Attention.
            (
                (1 + (args.num_query_groups / args.num_attention_heads))
                * query_projection_to_hidden_size_ratio
            )
            # MLP.
            + ((args.ffn_hidden_size / args.hidden_size) * (args.num_routed_experts + args.num_shared_experts) * gated_linear_multiplier)
            # Transformer layernorms.
            + (2 / args.hidden_size)
            # Final layernorm.
            + (1 / (num_sparse_layers * args.hidden_size))
        )
    )

    embedding_size = args.hidden_size * args.padded_vocab_size
    if args.untie_embeddings_and_output_weights:
        num_parameters_in_embedding_layers = 2 * embedding_size
    else:
        num_parameters_in_embedding_layers = embedding_size
    num_parameters_in_transformer_layers = num_parameters_in_transformer_dense_layers + num_parameters_in_transformer_sparse_layers
    num_total_parameters = num_parameters_in_transformer_layers + num_parameters_in_embedding_layers
    if verbose:
        print(
            f" > Number of parameters in transformer layers in billions:"
            f"{num_parameters_in_transformer_layers / 10**9: .2f} ({args.num_layers} layers)"
        )
        print(
            f"  > Number of parameters in transformer dense layers in billions:"
            f"{num_parameters_in_transformer_dense_layers / 10**9: .2f} ({num_dense_layers} layers)"
        )
        print(
            f"  > Number of parameters in transformer sparse layers in billions:"
            f"{num_parameters_in_transformer_sparse_layers / 10**9: .2f} ({num_sparse_layers} layers)"
        )
        print(
            f" > Number of parameters in embedding layers in billions: "
            f"{num_parameters_in_embedding_layers / 10**9:.2f}"
        )
        print(f"> Total number of parameters in billions: {num_total_parameters / 10**9:.2f}")

    # Most loaded model shard has (1/pp_size transformer layers + 1 embedding layer) / tp_size.
    num_parameters_on_most_loaded_model_shard = (
        (num_parameters_in_transformer_layers / args.pipeline_model_parallel_size) + embedding_size
    ) / args.tensor_model_parallel_size
    if args.zero_stage >= 1:
        num_parameters_on_most_loaded_model_shard = num_parameters_on_most_loaded_model_shard // args.data_parallel_size
    if args.untie_embeddings_and_output_weights and args.pipeline_model_parallel_size == 1:
        num_parameters_on_most_loaded_model_shard += (
            embedding_size / args.tensor_model_parallel_size
        )
    if verbose:
        print(
            f"  > Number of parameters in most loaded shard in billions: "
            f"{num_parameters_on_most_loaded_model_shard / 10**9:.4f}"
        )

    if args.pipeline_model_parallel_size > 1:
        # Other shards just have (1/pp_size transformer layers) / tp_size.
        num_parameters_on_other_model_shards = num_parameters_in_transformer_layers / (
            args.pipeline_model_parallel_size * args.tensor_model_parallel_size
        )
        if verbose:
            print(
                f"  > Number of parameters in other shards in billions: "
                f"{num_parameters_on_other_model_shards / 10**9:.4f}"
            )

    num_bytes_per_parameter = (
        18 if not args.use_distributed_optimizer else 6 + (12 / args.data_parallel_size)
    )
    weight_and_optimizer_memory = (
        num_parameters_on_most_loaded_model_shard * num_bytes_per_parameter
    )

    return weight_and_optimizer_memory


def compute_activation_memory(args, num_microbatches, verbose=False):
    # Using formula in Table 2 of https://arxiv.org/pdf/2205.05198.pdf.
    # We are trying to compute the maximum activation footprint, so all calculations in this
    # function are for the first pipeline stage.

    # TODO: This function needs to take into account query_projection_size potentially being
    # different from hidden_size.

    # Memory footprint from transformer layer (self-attention and MLP).
    sbh = args.seq_length * args.micro_batch_size * args.hidden_size
    if args.tensor_model_parallel_size > 1:
        ht = args.hidden_size * args.tensor_model_parallel_size
        if args.sequence_parallel:
            # tensor parallel + sequence parallel + selective activation recomputation
            if args.recompute_granularity == 'selective':
                activation_memory = sbh / args.tensor_model_parallel_size * (
                    18 +
                    (4 * args.ffn_hidden_size / args.hidden_size)
                )
            # tensor parallel + sequence parallel
            else:
                activation_memory = sbh / args.tensor_model_parallel_size * (
                    18 +
                    (4 * args.ffn_hidden_size / args.hidden_size) +
                    (5 * args.num_attention_heads * args.seq_length / args.hidden_size)
                )
        # tensor parallel + selective activation recomputatation
        elif args.recompute_granularity == 'selective':
            activation_memory = sbh * (
                10 +
                (4 * args.ffn_hidden_size / ht) +
                (8 / args.tensor_model_parallel_size)
            )
        # tensor parallel
        else:
            activation_memory = sbh * (
                10 +
                (4 * args.ffn_hidden_size / ht) +
                (8 / args.tensor_model_parallel_size) + 
                (5 * args.num_attention_heads * args.seq_length / ht)
            )
    # full activation recomputation
    elif args.recompute_granularity == 'full':
        activation_memory = 2 * sbh
    # no parallelism or recomputation.
    else:
        activation_memory = sbh * (
            18 +
            (4 * args.ffn_hidden_size / args.hidden_size) + 
            (5 * args.num_attention_heads * args.seq_length / args.hidden_size)
        )

    if verbose:
        print(
            f" > Activation memory footprint per transformer layer: "
            f"{activation_memory / NUM_BYTES_IN_MEGABYTE / args.tensor_model_parallel_size:.1f} MB"
        )
    activation_memory *= args.num_layers

    # Now add activation memory required for input embeddings, last LayerNorm and output layer.

    # Input to embedding (pp_size microbatches in flight).
    activation_memory += (
        8 * args.seq_length * args.micro_batch_size * args.pipeline_model_parallel_size
    )
    # Dropout in embedding layer (pp_size microbatches in flight).
    activation_memory += (
        args.seq_length
        * args.micro_batch_size
        * args.hidden_size
        * args.pipeline_model_parallel_size
    )

    # Multiply by interleaved PP memory factor.
    if args.virtual_pipeline_model_parallel_size is not None:
        interleaved_schedule_memory_penalty = 1 + (
            (args.pipeline_model_parallel_size - 1)
            / (args.pipeline_model_parallel_size * args.virtual_pipeline_model_parallel_size)
        )
        in_flight_microbatches = math.ceil(
            interleaved_schedule_memory_penalty * args.pipeline_model_parallel_size
        )
        if verbose:
            print(
                f"Memory penalty from interleaved schedule: {interleaved_schedule_memory_penalty:.2f}"
            )
            print(f"Number of in-flight microbatches: {in_flight_microbatches}")
        activation_memory *= interleaved_schedule_memory_penalty

    # If using non-interleaved schedule, number of microbatches in pipeline can be less than pp_size,
    # so discount accordingly.
    if args.virtual_pipeline_model_parallel_size is None and args.pipeline_model_parallel_size > 1:
        if num_microbatches is not None:
            activation_memory *= min(1, num_microbatches / args.pipeline_model_parallel_size)
            in_flight_microbatches = min(num_microbatches, args.pipeline_model_parallel_size)
        else:
            in_flight_microbatches = args.pipeline_model_parallel_size
        if verbose:
            print(f"Number of in-flight microbatches: {in_flight_microbatches}")

    if args.pipeline_model_parallel_size == 1:
        # Inputs to output layer and CE loss.
        activation_memory += (
            args.seq_length
            * args.micro_batch_size
            * args.hidden_size
            * 4
            * (1 + (args.padded_vocab_size / args.hidden_size))
        )

    return activation_memory


def report_theoretical_memory_usage(args, num_microbatches=None, verbose=False):
    print("Theoretical memory usage estimation:")

    weight_and_optimizer_memory = (
        compute_weight_and_optimizer_memory(args, verbose=verbose) / NUM_BYTES_IN_MEGABYTE
    )

    activation_memory = (
        compute_activation_memory(args, num_microbatches=num_microbatches, verbose=verbose)
        / NUM_BYTES_IN_MEGABYTE
    )
    total_memory = weight_and_optimizer_memory + activation_memory

    print(
        f"> Theoretical memory footprints: weight and optimizer={weight_and_optimizer_memory:.2f} MB, "
        f"activation={activation_memory:.2f} MB, total={total_memory:.2f} MB\n"
    )