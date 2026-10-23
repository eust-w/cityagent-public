"""BF16 Kev loader for a shared GPU; only the embedding lookup stays on CPU.

No quantization, training, upstream edits, or shared-service configuration changes.
This is a capacity workaround, not the official exclusive-GPU speed configuration.
"""
import time


def load_graph_gpu(checkpoint):
    """Bounded all-GPU graph experiment; fail rather than claim a fallback.

    CPU embedding hooks cannot be captured. Full weights must fit, along with
    smaller graph buffers. These are isolated Kev-process constants only.
    """
    import torch
    from kev.checkpoint import LoadOptions
    import kev.cuda_graphs as cg

    free, total = torch.cuda.mem_get_info()
    reserve = 768 * 1024**2
    cap = min(int(total * .90), free - reserve)
    if cap < 8 * 1024**3:
        raise RuntimeError('CUDA Graph probe needs full weights and buffers; insufficient free memory')
    torch.cuda.set_per_process_memory_fraction(cap / total)
    started = time.monotonic()
    tokenizer, model = checkpoint.load('cpu', LoadOptions(dtype=torch.bfloat16,
        attn='sdpa', merge=True, cuda_graphs=False, fused=False))
    model.lm.to('cuda'); model.head.to('cuda'); model.device = 'cuda'
    limits = ({name:getattr(cg,name) for name in ('GRAPH_STATES','BANK_WIDTH','GRAPH_STATE',
        'GRAPH_ROWS','GRAPH_ROW','GRAPH_TOKENS','GRAPHS_KEPT')} if free >= 16*1024**3 else
        dict(GRAPH_STATES=4, BANK_WIDTH=1024, GRAPH_STATE=512,
        GRAPH_ROWS=8, GRAPH_ROW=1024, GRAPH_TOKENS=4096, GRAPHS_KEPT=8))
    for name, value in limits.items(): setattr(cg, name, value)
    model.graphs = cg.CudaGraphs(model.lm, model.pad_id)
    torch.cuda.synchronize()
    return tokenizer, model, {'mode':'bf16_full_gpu_bounded_cuda_graphs',
        'load_seconds':time.monotonic()-started, 'free_before_bytes':free,
        'cuda_cap_bytes':cap, 'reserve_bytes':reserve, 'graph_limits':limits,
        'cuda_graphs_requested':True,'fused':False,
        'gpu_parameter_bytes':sum(p.numel()*p.element_size() for p in model.parameters()
            if p.device.type=='cuda')}


def load_shared_gpu(checkpoint):
    import torch
    from kev.checkpoint import LoadOptions

    free, total = torch.cuda.mem_get_info()
    cap = min(int(total * .26), free - 1024**3)
    if cap < 7 * 1024**3:
        raise RuntimeError('Insufficient free memory; existing services must remain untouched')
    torch.cuda.set_per_process_memory_fraction(cap / total)
    started = time.monotonic()
    tokenizer, model = checkpoint.load('cpu', LoadOptions(
        dtype=torch.bfloat16, attn='sdpa', merge=True,
        cuda_graphs=False, fused=False))
    embedding = model.lm.embed_tokens
    assert isinstance(embedding, torch.nn.Embedding)
    assert next(embedding.parameters()).device.type == 'cpu'
    cpu_bytes = sum(p.numel() * p.element_size() for p in embedding.parameters())

    # Move the entire decoder, including non-persistent rotary buffers, then
    # restore the original CPU embedding. Never materialize that large embedding
    # on the shared GPU, including temporarily during loading.
    model.lm.embed_tokens = torch.nn.Identity()
    model.lm.to('cuda')
    model.lm.embed_tokens = embedding
    model.head.to('cuda')
    model.device = 'cuda'

    def ids_to_cpu(module, args):
        return tuple(v.to('cpu') if isinstance(v, torch.Tensor) else v for v in args)

    def vectors_to_cuda(module, args, result):
        return result.to('cuda')

    embedding.register_forward_pre_hook(ids_to_cpu)
    embedding.register_forward_hook(vectors_to_cuda)
    # Check exactly preserved lookup values without a decoder call.
    sample = torch.tensor([[0, 1, 2]], device='cuda', dtype=torch.long)
    actual = embedding(sample)
    expected = torch.nn.functional.embedding(sample.cpu(), embedding.weight).to('cuda')
    assert torch.equal(actual, expected), 'Embedding transfer changed values'
    del actual, expected, sample
    torch.cuda.synchronize()
    return tokenizer, model, {
        'mode': 'bf16_cpu_embedding_gpu_decoder',
        'load_seconds': time.monotonic() - started,
        'cpu_embedding_bytes': cpu_bytes,
        'gpu_parameter_bytes': sum(p.numel() * p.element_size() for p in model.parameters()
                                   if p.device.type == 'cuda'),
        'cuda_cap_bytes': cap, 'free_before_bytes': free,
        'embedding_transfer_exact': True,
        'cuda_graphs': False, 'fused': False,
        'note': 'Shared GPU capacity validation, not an exclusive-GPU speed benchmark',
    }
