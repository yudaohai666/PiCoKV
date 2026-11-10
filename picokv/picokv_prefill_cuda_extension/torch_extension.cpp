#include <torch/extension.h>
#include <ATen/ATen.h>
#include <torch/csrc/autograd/python_variable.h>

#include <algorithm>

#include "cuda_kernel.h"

std::vector<at::Tensor> picokv_prefill_learn_projection_fn(
    at::Tensor full_states,
    at::Tensor init_vectors,
    int64_t glob_start,
    int64_t glob_len,
    int64_t zerorope,
    bool apply_zerorope) {
  return picokv_prefill_learn_projection(
      full_states,
      init_vectors,
      glob_start,
      glob_len,
      zerorope,
      apply_zerorope);
}


std::vector<at::Tensor> picokv_prefill_quantize_fn(
    at::Tensor full_states,
    at::Tensor projection,
    int64_t glob_start,
    int64_t glob_len,
    int64_t zerorope,
    bool apply_zerorope) {
  return picokv_prefill_quantize(
      full_states,
      projection,
      glob_start,
      glob_len,
      zerorope,
      apply_zerorope);
}


std::vector<at::Tensor> picokv_prefill_quantize_pack_fn(
    at::Tensor full_states,
    at::Tensor projection,
    int64_t glob_start,
    int64_t glob_len,
    int64_t zerorope,
    bool apply_zerorope,
    bool pack_key_layout) {
  return picokv_prefill_quantize_pack(
      full_states,
      projection,
      glob_start,
      glob_len,
      zerorope,
      apply_zerorope,
      pack_key_layout);
}

std::vector<at::Tensor> picokv_prefill_blockk_fast_fn(
    at::Tensor full_states,
    at::Tensor init_vectors,
    int64_t glob_start,
    int64_t glob_len,
    int64_t zerorope,
    bool apply_zerorope,
    int64_t group_bits,
    int64_t power_iters,
    int64_t solve_mode) {
  return picokv_prefill_blockk_fast(
      full_states,
      init_vectors,
      glob_start,
      glob_len,
      zerorope,
      apply_zerorope,
      group_bits,
      power_iters,
      solve_mode);
}

std::vector<at::Tensor> picokv_prefill_blockk_fixed_fn(
    at::Tensor full_states,
    at::Tensor encoders,
    at::Tensor decoders,
    int64_t glob_start,
    int64_t glob_len,
    int64_t zerorope,
    bool apply_zerorope,
    bool pack_key_layout) {
  return picokv_prefill_blockk_fixed(
      full_states,
      encoders,
      decoders,
      glob_start,
      glob_len,
      zerorope,
      apply_zerorope,
      pack_key_layout);
}

std::vector<at::Tensor> picokv_prefill_blockk_fixed_variant_fn(
    at::Tensor full_states,
    at::Tensor encoders,
    at::Tensor decoders,
    int64_t glob_start,
    int64_t glob_len,
    int64_t zerorope,
    bool apply_zerorope,
    bool pack_key_layout,
    int64_t band_nb) {
  return picokv_prefill_blockk_fixed_variant(
      full_states.contiguous(),
      encoders.contiguous(),
      decoders.contiguous(),
      glob_start,
      glob_len,
      zerorope,
      apply_zerorope,
      pack_key_layout,
      band_nb);
}

std::vector<at::Tensor> picokv_prefill_select_eas_fn(
    at::Tensor key_err,
    at::Tensor value_err,
    int64_t eas_size) {
  return picokv_prefill_select_eas(key_err, value_err, eas_size);
}

at::Tensor picokv_prefill_pack_key_transposed_fn(at::Tensor codes) {
  return picokv_prefill_pack_key_transposed(codes);
}

at::Tensor picokv_prefill_pack_value_token_fn(at::Tensor codes) {
  return picokv_prefill_pack_value_token(codes);
}

std::vector<at::Tensor> picokv_prefill_gather_pack_fn(
    at::Tensor full_key,
    at::Tensor full_value,
    at::Tensor key_codes,
    at::Tensor value_codes,
    at::Tensor eas_indices,
    at::Tensor kept_indices,
    int64_t sink_size,
    int64_t local_size) {
  return picokv_prefill_gather_pack(
      full_key,
      full_value,
      key_codes,
      value_codes,
      eas_indices,
      kept_indices,
      sink_size,
      local_size);
}


std::vector<at::Tensor> picokv_prefill_gather_pack_from_packed_fn(
    at::Tensor full_key,
    at::Tensor full_value,
    at::Tensor packed_key_all,
    at::Tensor packed_value_token,
    at::Tensor eas_indices,
    at::Tensor kept_indices,
    int64_t sink_size,
    int64_t local_size) {
  return picokv_prefill_gather_pack_from_packed(
      full_key,
      full_value,
      packed_key_all,
      packed_value_token,
      eas_indices,
      kept_indices,
      sink_size,
      local_size);
}

std::vector<at::Tensor> picokv_prefill_gather_pack_from_packed_out_fn(
    at::Tensor full_key,
    at::Tensor full_value,
    at::Tensor packed_key_all,
    at::Tensor packed_value_token,
    at::Tensor eas_indices,
    at::Tensor kept_indices,
    int64_t sink_size,
    int64_t local_size,
    at::Tensor dense_key_out,
    at::Tensor dense_value_out) {
  return picokv_prefill_gather_pack_from_packed_out(
      full_key,
      full_value,
      packed_key_all,
      packed_value_token,
      eas_indices,
      kept_indices,
      sink_size,
      local_size,
      dense_key_out,
      dense_value_out);
}

std::vector<at::Tensor> picokv_prefill_prepare_fast_fn(
    at::Tensor kv,
    int64_t off,
    int64_t len,
    at::Tensor enc_k,
    at::Tensor dec_k,
    at::Tensor enc_v,
    at::Tensor dec_v,
    at::Tensor kept_indices,
    int64_t sink_size,
    int64_t local_size,
    int64_t zerorope,
    c10::optional<at::Tensor> dense_key_out,
    c10::optional<at::Tensor> dense_value_out) {
  return picokv_prefill_prepare_fast(kv, off, len, enc_k, dec_k, enc_v, dec_v, kept_indices,
                                     sink_size, local_size, zerorope, dense_key_out,
                                     dense_value_out);
}

// PICOKV_FILL_TAB: the decode backend's _fill_fresh launch table (17 int64 per
// job) built here instead of per job in Python (~8 us each, 32 per fresh row).
// Only the layout the kernel takes is handled; anything else (an unexpected
// type / rank, a check _fill_fresh fails, a Python error) returns None and the
// Python path decides as before, so the table and verdict are the same.
namespace picokv_fill_tab {

inline bool tensor_of(PyObject* o, at::Tensor& t) {
  if (o == nullptr || !THPVariable_Check(o)) return false;
  t = THPVariable_Unpack(o);
  return t.defined();
}

inline bool int_of(PyObject* o, int64_t& x) {
  if (o == nullptr || !PyLong_Check(o)) return false;
  int overflow = 0;
  x = PyLong_AsLongLongAndOverflow(o, &overflow);
  if (overflow != 0 || (x == -1 && PyErr_Occurred())) {
    PyErr_Clear();
    return false;
  }
  return true;
}

inline bool attr_tensor(PyObject* o, const char* name, at::Tensor& t) {
  PyObject* a = PyObject_GetAttrString(o, name);
  if (a == nullptr) {
    PyErr_Clear();
    return false;
  }
  const bool ok = tensor_of(a, t);
  Py_DECREF(a);  // the cache object keeps the tensor
  return ok;
}

// _pk_full(t): t's view base if it carries _pk_full (truthy), else undefined;
// false on a Python error
inline bool pk_full(const at::Tensor& t, at::Tensor& full) {
  full = at::Tensor();
  if (!t.is_view()) return true;  // Python t._base: None
  const at::TensorBase& b = t._base();
  if (!b.defined()) return true;
  PyObject* bo = THPVariable_Wrap(b);
  if (bo == nullptr) {
    PyErr_Clear();
    return false;
  }
  PyObject* f = PyObject_GetAttrString(bo, "_pk_full");
  int yes = 0;
  if (f == nullptr) {
    PyErr_Clear();
  } else {
    yes = PyObject_IsTrue(f);
    Py_DECREF(f);
  }
  Py_DECREF(bo);
  if (yes < 0) {
    PyErr_Clear();
    return false;
  }
  if (yes) full = at::Tensor(b);
  return true;
}

inline int64_t ptr(const at::Tensor& t) {
  return reinterpret_cast<int64_t>(t.data_ptr());
}

}  // namespace picokv_fill_tab

// jobs: [(pk, pv, kb, vb, i, c, (k, v), g, d)] as _fill_fresh takes them;
// returns the table on kb's device (1-D, copied from pinned memory with
// non_blocking like the Python path) or None
py::object picokv_fill_table_fn(py::object jobs_obj, int64_t H, int64_t D) {
  using namespace picokv_fill_tab;
  PyObject* jobs = jobs_obj.ptr();
  if (!PyList_Check(jobs)) return py::none();
  const Py_ssize_t nj = PyList_GET_SIZE(jobs);
  if (nj == 0) return py::none();
  auto tab = at::empty({static_cast<int64_t>(nj) * 17},
                       at::TensorOptions().dtype(at::kLong).pinned_memory(true));
  int64_t* o = tab.data_ptr<int64_t>();
  int64_t HKW = -1;
  at::Device dev(at::kCPU);
  at::Tensor pk, pv, kb, vb, k, v, cpk, cpv, kf, vf;
  for (Py_ssize_t j = 0; j < nj; ++j, o += 17) {
    PyObject* job = PyList_GET_ITEM(jobs, j);
    if (!PyTuple_Check(job) || PyTuple_GET_SIZE(job) != 9) return py::none();
    int64_t i, g, d;
    if (!tensor_of(PyTuple_GET_ITEM(job, 0), pk) || !tensor_of(PyTuple_GET_ITEM(job, 1), pv)
        || !tensor_of(PyTuple_GET_ITEM(job, 2), kb) || !tensor_of(PyTuple_GET_ITEM(job, 3), vb)
        || !int_of(PyTuple_GET_ITEM(job, 4), i) || !int_of(PyTuple_GET_ITEM(job, 7), g)
        || !int_of(PyTuple_GET_ITEM(job, 8), d))
      return py::none();
    PyObject* c = PyTuple_GET_ITEM(job, 5);
    PyObject* s = PyTuple_GET_ITEM(job, 6);
    if (!PyTuple_Check(s) || PyTuple_GET_SIZE(s) != 2 || !tensor_of(PyTuple_GET_ITEM(s, 0), k)
        || !tensor_of(PyTuple_GET_ITEM(s, 1), v))
      return py::none();
    if (pk.dim() != 4 || pv.dim() != 4 || kb.dim() != 4 || vb.dim() != 4 || k.dim() != 4
        || v.dim() != 4)
      return py::none();
    if (j == 0) {
      HKW = pk.size(1) * pk.size(2);
      dev = kb.device();
    }
    // window buffers
    if (!(kb.scalar_type() == at::kBFloat16 && vb.scalar_type() == at::kBFloat16
          && kb.size(2) == H && kb.size(3) == D && kb.sizes() == vb.sizes()
          && kb.is_contiguous() && vb.is_contiguous()))
      return py::none();
    const int64_t kbp = ptr(kb), vbp = ptr(vb), cap = kb.size(1), kr = cap * H * D * 2;
    // packed rectangle (or one row)
    if (!(pk.scalar_type() == at::kLong && pv.scalar_type() == at::kLong && pk.size(1) == H
          && pv.size(1) == H && pk.size(1) * pk.size(2) == HKW && pk.is_contiguous()
          && pv.is_contiguous()))
      return py::none();
    const int64_t L = pk.size(3), w = pv.size(2) * pv.size(3);
    const int64_t rk = HKW * L * 8, rv = H * w * 8, G = pv.size(2), CV = pv.size(3);
    const int64_t pi = pk.size(0) > 1 ? i : 0;
    const int64_t pkd = ptr(pk) + pi * rk, pvd = ptr(pv) + pi * rv;
    int64_t n = g, Lr = L, wr = w, ps = 0, vs = 0, nv = 0;
    if (c != Py_None) {
      if (!attr_tensor(c, "packed_key", cpk) || !attr_tensor(c, "packed_value", cpv)
          || cpk.dim() != 4 || cpv.dim() != 4)
        return py::none();
      const int64_t ck = cpk.size(3), cg = cpv.size(2), cc = cpv.size(3);
      if (!(ck == g && g <= L && cpk.size(1) * cpk.size(2) == HKW && cc == CV && cg <= G))
        return py::none();
      ps = ptr(cpk);
      vs = ptr(cpv);
      nv = cg * cc;
      if (!(cpk.is_contiguous() && cpv.is_contiguous())) {
        // PICOKV_PREALLOC prefix views: their whole buffers
        if (!pk_full(cpk, kf) || !pk_full(cpv, vf)) return py::none();
        if (!kf.defined() || !vf.defined() || kf.dim() < 1 || vf.dim() < 3 || ptr(kf) != ps
            || ptr(vf) != vs)
          return py::none();
        n = kf.size(-1);
        nv = vf.size(2) * cc;
      }
      if (ps == pkd && vs == pvd) n = nv = Lr = wr = 0;  // adopted buffer
    } else {
      n = 0;
    }
    if (!(d <= cap && k.size(2) >= d && v.size(2) >= d && k.size(1) == H && v.size(1) == H
          && k.stride(3) == 1 && v.stride(3) == 1 && k.scalar_type() == at::kBFloat16
          && v.scalar_type() == at::kBFloat16))
      return py::none();
    const int64_t row[17] = {ps, vs, ptr(k), ptr(v), pkd, pvd, kbp + i * kr, vbp + i * kr,
                             n, nv, d, Lr, wr, k.stride(2), k.stride(1), v.stride(2),
                             v.stride(1)};
    std::copy(row, row + 17, o);
  }
  return py::cast(tab.to(dev, at::kLong, /*non_blocking=*/true, /*copy=*/false));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("picokv_prefill_learn_projection_fn", &picokv_prefill_learn_projection_fn, "PiCoKV CUDA projection learning");
  m.def("picokv_prefill_quantize_fn", &picokv_prefill_quantize_fn, "PiCoKV prefill projection quantization");
  m.def("picokv_prefill_quantize_pack_fn", &picokv_prefill_quantize_pack_fn, "PiCoKV fused prefill quantization and token/code packing");
  m.def("picokv_prefill_blockk_fast_fn", &picokv_prefill_blockk_fast_fn, "PiCoKV fast fused block-k prefill learning and quantization");
  m.def("picokv_prefill_blockk_fixed_fn", &picokv_prefill_blockk_fixed_fn, "PiCoKV fully fused fixed-projection cascade quantization and packing");
  m.def("picokv_prefill_blockk_fixed_variant_fn", &picokv_prefill_blockk_fixed_variant_fn, "blockk_fixed with a forced kernel: band_nb -1 auto, 0 frag, >0 band kernel with band_nb bands per CTA");
  m.def("picokv_prefill_select_eas_fn", &picokv_prefill_select_eas_fn, "PiCoKV prefill EAS selection");
  m.def("picokv_prefill_pack_key_transposed_fn", &picokv_prefill_pack_key_transposed_fn, "PiCoKV pack int8 key codes to [B,H,C/64,L]");
  m.def("picokv_prefill_pack_value_token_fn", &picokv_prefill_pack_value_token_fn, "PiCoKV pack int8 value codes to [B,H,L,C/64]");
  m.def("picokv_prefill_gather_pack_fn", &picokv_prefill_gather_pack_fn, "PiCoKV prefill dense gather and packed code output");
  m.def("picokv_prefill_gather_pack_from_packed_fn", &picokv_prefill_gather_pack_from_packed_fn, "PiCoKV prefill dense gather and packed output from fused packed codes");
  m.def("picokv_prefill_gather_pack_from_packed_out_fn", &picokv_prefill_gather_pack_from_packed_out_fn, "PiCoKV prefill gather_pack_from_packed into caller-provided dense outputs");
  m.def("picokv_prefill_prepare_fast_fn", &picokv_prefill_prepare_fast_fn, "PiCoKV fast prepare of one compressed prompt layer (eas=0): K/V split + blockk_fixed V/K + value regroup in one call",
        py::arg("kv"), py::arg("off"), py::arg("len"), py::arg("enc_k"), py::arg("dec_k"),
        py::arg("enc_v"), py::arg("dec_v"), py::arg("kept_indices"), py::arg("sink_size"),
        py::arg("local_size"), py::arg("zerorope"), py::arg("dense_key_out") = py::none(),
        py::arg("dense_value_out") = py::none());
  m.def("picokv_fill_table_fn", &picokv_fill_table_fn, "PiCoKV decode fill-launch table of _fill_fresh jobs on the jobs' device, or None (PICOKV_FILL_TAB)",
        py::arg("jobs"), py::arg("H"), py::arg("D"));
}
