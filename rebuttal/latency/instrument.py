"""Process-local CUDA-event instrumentation of the unchanged production source.

Only the diagnostic run installs these wrappers. No per-stage synchronization,
no copied implementation, and no modification to production files.
"""
import ast
from collections import defaultdict
from contextlib import contextmanager
import functools
import inspect
import textwrap

import torch
from picokv.picokv_standalone import online, prepare
from picokv.picokv_standalone import transformers as adapter


def union_ms(intervals):
    total = 0.0
    end = None
    for a, b in sorted(intervals):
        if end is None or a > end:
            total += b-a
            end = b
        elif b > end:
            total += b-end
            end = b
    return total


class Recorder:
    def __init__(self):
        self.records = []
        self.layer = -1
        self.side = 'shared'
        self.layers = []
        self.anchor = torch.cuda.Event(enable_timing=True)
        self.anchor.record()
        self.patches = []

    @contextmanager
    def phase(self, name):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        stream = torch.cuda.current_stream()
        start.record(stream)
        try:
            yield
        finally:
            end.record(stream)
            self.records.append((self.layer, self.side, name, start, end, stream.cuda_stream))

    def patch(self, obj, name, value):
        self.patches.append((obj, name, getattr(obj, name)))
        setattr(obj, name, value)

    def install(self):
        measured_quantize = instrument_quantize(self)
        def quantize(*args, **kwargs):
            # Production prepare dispatches value first on its side stream.
            previous = self.side
            self.side = 'value' if self.quantize_calls == 0 else 'key'
            self.quantize_calls += 1
            assert kwargs.get('encoders') is None and kwargs.get('decoders') is None
            try:
                return measured_quantize(*args, **kwargs)
            finally:
                self.side = previous
        self.patch(prepare, 'quantize_block32_online', quantize)
        original_prepare = adapter.prepare_kv_cache
        @functools.wraps(original_prepare)
        def prepare_call(*args, **kwargs):
            self.layer += 1
            self.quantize_calls = 0
            self.side = 'shared'
            assert all(kwargs.get(k) is None for k in ['key_encoders', 'value_encoders', 'key_decoders', 'value_decoders'])
            with self.phase('complete_cache_prepare'):
                result = original_prepare(*args, **kwargs)
            assert self.quantize_calls == 2
            self.layers.append({'layer': self.layer, 'input_shape': list(args[0].shape),
                                'global_len': result.global_len, 'dense_len': result.dense_len})
            return result
        self.patch(adapter, 'prepare_kv_cache', prepare_call)
        for name, category in [
            ('pack_key_transposed_cuda', 'packing'),
            ('pack_value_token_cuda', 'packing'),
            ('_select_eas_and_kept_indices', 'eas_selection'),
            ('gather_pack_from_packed_cuda', 'gather_compact_packed_cache'),
        ]:
            original = getattr(prepare, name)
            def wrap(*args, _original=original, _category=category, **kwargs):
                with self.phase(_category):
                    return _original(*args, **kwargs)
            self.patch(prepare, name, wrap)
        return self

    def restore(self):
        for obj, name, value in reversed(self.patches):
            setattr(obj, name, value)
        self.patches.clear()

    def summarize(self):
        torch.cuda.synchronize()
        rows = []
        for layer, side, category, start, end, stream in self.records:
            rows.append({'layer': layer, 'side': side, 'category': category,
                         'start_ms': self.anchor.elapsed_time(start),
                         'end_ms': self.anchor.elapsed_time(end),
                         'elapsed_ms': start.elapsed_time(end), 'stream': stream})
        stages = defaultdict(list)
        per_layer = defaultdict(lambda: defaultdict(float))
        per_side = defaultdict(lambda: defaultdict(float))
        for r in rows:
            stages[r['category']].append((r['start_ms'], r['end_ms']))
            per_layer[str(r['layer'])][r['category']] += r['elapsed_ms']
            per_side[r['side']][r['category']] += r['elapsed_ms']
        return {
            'scope': 'CUDA-event elapsed intervals; diagnostic run only; K/V streams overlap; stage sums are NOT an additive wall-time breakdown',
            'complete_cache_prepare_ms_all_layers': sum(per_layer[k]['complete_cache_prepare'] for k in per_layer),
            'stage_sum_across_streams_ms': {k: sum(b-a for a, b in v) for k, v in stages.items()},
            'stage_union_across_streams_ms': {k: union_ms(v) for k, v in stages.items()},
            'per_side_stage_sum_ms': dict(per_side),
            'per_layer_stage_sum_ms': dict(per_layer),
            'layer_shapes': self.layers, 'intervals': rows,
        }


def _wrap(nodes, category):
    call = ast.Call(func=ast.Name(id='_latency_phase', ctx=ast.Load()),
                    args=[ast.Constant(category)], keywords=[])
    return ast.With(items=[ast.withitem(context_expr=call)], body=nodes)


def instrument_quantize(recorder):
    """Group original AST statements; only add event context managers.

    Encoder initialization/search/QR + ridge decoder fitting are projection
    fitting; sign encoding, residual updates, code stores and final error norm
    are GPC. Nested helper operations remain inside their enclosing interval.
    Fails closed if expected production statements change.
    """
    original = online.quantize_block32_online
    tree = ast.parse(textwrap.dedent(inspect.getsource(original)))
    fn = tree.body[0]
    source = lambda n: ast.unparse(n)
    def target(n):
        if isinstance(n, ast.Assign):
            return source(n.targets[0])
        return ''
    expected = {'generator', 'init_vectors', 'residual', 'codes', 'decoder', 'used_encoders', 'token_error'}
    assert expected <= {target(n) for n in fn.body}, 'production online structure changed'
    new = []
    seen = defaultdict(int)
    for n in fn.body:
        name = target(n)
        if isinstance(n, ast.For) and source(n.target) == 'group_idx':
            assert len(n.body) == 13, 'production GPC loop changed; re-audit instrumentation'
            # start,end,encoder search,encoder store,residual cast,code GEMM,
            # binary sign,decoder solve,decoder cast,residual update,sampled
            # residual update,code store,decoder store.
            assert target(n.body[4]) == 'residual_m'
            assert target(n.body[5]) == 'group_codes_m'
            assert '_solve_decoder_from_matrices' in source(n.body[7])
            n.body = n.body[:2] + [_wrap(n.body[2:4], 'projection_fitting'),
                                 _wrap(n.body[4:7], 'gpc_encoding_residual'),
                                 _wrap(n.body[7:8], 'projection_fitting'),
                                 _wrap(n.body[8:], 'gpc_encoding_residual')]
            new.append(n)
            seen['group'] += 1
        elif name in {'generator', 'init_vectors'} or (isinstance(n, ast.Expr) and 'generator.manual_seed' in source(n)):
            new.append(_wrap([n], 'projection_fitting'))
        elif isinstance(n, ast.If) and source(n.test) == 'fixed_encoders is None':
            new.append(_wrap([n], 'projection_fitting'))
            seen['init'] += 1
        elif name == 'token_error':
            new.append(_wrap([n], 'gpc_encoding_residual'))
        else:
            new.append(n)
    assert dict(seen) == {'init': 1, 'group': 1}
    fn.body = new
    namespace = dict(original.__globals__)
    namespace['_latency_phase'] = recorder.phase
    exec(compile(ast.fix_missing_locations(tree), inspect.getsourcefile(original), 'exec'), namespace)
    return namespace[original.__name__]
