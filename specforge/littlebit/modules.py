import os

import torch
import torch.nn as nn

from .packing import binary_packer


class LittleBitLinear(nn.Module):
    def __quant_convert__(
        self,
        do_train: bool,
        quant_func,
        *,
        split_dim: int = 1024,
        eff_bit: float | None = None,
        residual: bool = False,
        ratio_factor: float = 1.0,
        min_split_dim: int = 8,
        **kwargs,
    ):
        self.do_train = do_train
        self.quant_func = quant_func
        self.residual = residual
        self._binarized = False

        a, b = self.in_features, self.out_features
        split_calc_float = self._estimate_split_dim(a, b, eff_bit, residual)
        if split_calc_float is not None:
            split_calc_float *= ratio_factor
        final_split_dim = self._finalize_split_dim(
            split_calc_float, split_dim, min_split_dim
        )
        eff_bit_actual = self._compute_eff_bits(a, b, final_split_dim, residual)
        buffer_device = self._get_buffer_device()

        self.register_buffer(
            "_eff_bit_target",
            torch.tensor(
                -1.0 if eff_bit is None else float(eff_bit), device=buffer_device
            ),
        )
        self.register_buffer(
            "_split_dim_final",
            torch.tensor(final_split_dim, device=buffer_device),
        )
        self.register_buffer(
            "_eff_bit_actual",
            torch.tensor(eff_bit_actual, device=buffer_device),
        )
        self.split_dim = final_split_dim

        if self.do_train and hasattr(self, "weight") and self.weight is not None:
            self._initialize_parameters()
        else:
            self._initialize_empty_parameters()

    def _get_buffer_device(self):
        if hasattr(self, "weight") and self.weight is not None:
            return self.weight.device
        if self.bias is not None:
            return self.bias.device
        return torch.device("cpu")

    @staticmethod
    def _estimate_split_dim(a, b, eff_bit_target, residual) -> float | None:
        if eff_bit_target is None or a * b == 0:
            return None

        base = a + b + 16
        if residual:
            numerator = a * b * eff_bit_target - 32 * (a + b)
            denominator = 2 * base
        else:
            numerator = a * b * eff_bit_target - 16 * (a + b)
            denominator = base
        return numerator / denominator if denominator else None

    @staticmethod
    def _finalize_split_dim(
        split_float: float | None,
        split_default: int,
        min_split_dim: int,
    ) -> int:
        candidate = split_float if split_float is not None else split_default
        candidate = int(candidate) if candidate is not None else 0
        candidate = (candidate // 8) * 8
        if candidate == 0:
            candidate = min_split_dim
        return max(candidate, min_split_dim)

    @staticmethod
    def _compute_eff_bits(a: int, b: int, s: int, residual: bool) -> float:
        if a * b == 0:
            return float("inf")

        if residual:
            numerator = s * 2 * (a + b + 16) + 32 * (a + b)
        else:
            numerator = s * (a + b + 16) + 16 * (a + b)
        return numerator / (a * b)

    def forward(self, x):
        *seqlen, hidden_dim = x.shape
        output_shape = tuple(seqlen + [self.out_features])
        x = x.view(-1, hidden_dim)

        y = self._compute_forward(
            x, self.V, self.U, self.v2, self.v1, self.u2, self.u1
        )
        if self.residual:
            y = y + self._compute_forward(
                x, self.V_R, self.U_R, self.v2_R, self.v1_R, self.u2_R, self.u1_R
            )
        if self.bias is not None:
            y += self.bias
        return y.reshape(output_shape)

    def _compute_forward(self, x, V, U, v2, v1, u2, u1):
        Vq = self.quantize(V.to(x.dtype))
        Uq = self.quantize(U.to(x.dtype))
        v1u2 = v1 * u2
        return ((((x * v2) @ Vq.t()) * v1u2) @ Uq.t()) * u1

    def quantize(self, x):
        if self._binarized:
            return x
        return self.quant_func(x)

    def _initialize_empty_parameters(self):
        dtype = torch.bfloat16
        device = "meta"

        def create_param(*shape):
            return nn.Parameter(
                torch.empty(*shape, device=device, dtype=dtype),
                requires_grad=self.do_train,
            )

        self.U = create_param(self.out_features, self.split_dim)
        self.V = create_param(self.split_dim, self.in_features)
        self.u1 = create_param(1, self.out_features)
        self.u2 = create_param(1, self.split_dim)
        self.v1 = create_param(1, self.split_dim)
        self.v2 = create_param(1, self.in_features)

        if self.residual:
            self.U_R = create_param(self.out_features, self.split_dim)
            self.V_R = create_param(self.split_dim, self.in_features)
            self.u1_R = create_param(1, self.out_features)
            self.u2_R = create_param(1, self.split_dim)
            self.v1_R = create_param(1, self.split_dim)
            self.v2_R = create_param(1, self.in_features)

        if hasattr(self, "weight"):
            del self.weight
        self.register_parameter("weight", None)

    def _initialize_parameters(self):
        weight = (
            self.weight.data.float()
            if self.do_train and self.weight is not None
            else None
        )
        U, V, u1, u2, v1, v2 = self._decompose_matrix(weight)

        def create_param(tensor):
            return nn.Parameter(tensor, requires_grad=self.do_train)

        self.U = create_param(U)
        self.V = create_param(V)
        self.u1 = create_param(u1)
        self.u2 = create_param(u2)
        self.v1 = create_param(v1)
        self.v2 = create_param(v2)

        if self.residual:
            residual_weight = None
            if self.do_train:
                local_rank = int(os.environ.get("LOCAL_RANK", "0"))
                calc_device = (
                    torch.device(f"cuda:{local_rank}")
                    if torch.cuda.is_available()
                    else self.weight.device
                )

                U_g = self.quantize(U).to(calc_device)
                V_g = self.quantize(V).to(calc_device)
                u1_g, u2_g = u1.to(calc_device), u2.to(calc_device)
                v1_g, v2_g = v1.to(calc_device), v2.to(calc_device)
                approx = (U_g * (u1_g.t() @ u2_g)) @ (V_g * (v1_g.t() @ v2_g))
                residual_weight = self.weight.data.float() - approx.to(self.weight.device)
                del U_g, V_g, u1_g, u2_g, v1_g, v2_g, approx

            U_R, V_R, u1_R, u2_R, v1_R, v2_R = self._decompose_matrix(
                residual_weight
            )
            self.U_R = create_param(U_R)
            self.V_R = create_param(V_R)
            self.u1_R = create_param(u1_R)
            self.u2_R = create_param(u2_R)
            self.v1_R = create_param(v1_R)
            self.v2_R = create_param(v2_R)

        self.register_parameter("weight", None)
        self._binarized = False

    def _decompose_matrix(self, matrix=None):
        if self.do_train:
            assert matrix is not None
            original_device = matrix.device
            local_rank = int(os.environ.get("LOCAL_RANK", "0"))
            calc_device = (
                torch.device(f"cuda:{local_rank}")
                if torch.cuda.is_available()
                else original_device
            )
            matrix_calc = matrix.to(calc_device)
            U_t, S_t, V_t = torch.svd_lowrank(matrix_calc, q=self.split_dim)
            Vh_t = V_t.t()
            sqrt_s = torch.sqrt(torch.diag(S_t))[:, : self.split_dim]

            U = (U_t @ sqrt_s).contiguous()
            V = (sqrt_s.t() @ Vh_t).contiguous()
            v1, v2 = self._rank_one_decompose(torch.abs(V), calc_device=calc_device)
            u1, u2 = self._rank_one_decompose(torch.abs(U), calc_device=calc_device)

            dtype = matrix.dtype
            U = U.to(device=original_device, dtype=dtype)
            V = V.to(device=original_device, dtype=dtype)
            u1 = u1.to(device=original_device, dtype=dtype)
            u2 = u2.to(device=original_device, dtype=dtype)
            v1 = v1.to(device=original_device, dtype=dtype)
            v2 = v2.to(device=original_device, dtype=dtype)
            del matrix_calc, U_t, S_t, V_t, Vh_t
        else:
            U = torch.empty(self.out_features, self.split_dim)
            V = torch.empty(self.split_dim, self.in_features)
            u1 = torch.empty(1, self.out_features)
            u2 = torch.empty(1, self.split_dim)
            v1 = torch.empty(1, self.split_dim)
            v2 = torch.empty(1, self.in_features)
        return U, V, u1, u2, v1, v2

    def _rank_one_decompose(self, matrix, calc_device=None):
        original_device = matrix.device
        if calc_device is None:
            local_rank = int(os.environ.get("LOCAL_RANK", "0"))
            calc_device = (
                torch.device(f"cuda:{local_rank}")
                if torch.cuda.is_available()
                else original_device
            )

        matrix_calc = matrix.to(calc_device)
        U, S, V = torch.svd_lowrank(matrix_calc, q=1)
        Vh = V.t()
        sqrt_s0 = torch.sqrt(S[0])
        u_component = (U[:, :1] * sqrt_s0).t().contiguous()
        v_component = (sqrt_s0 * Vh[:1, :]).contiguous()
        return u_component, v_component

    def pack_weights(self):
        packed = {}

        def pack_param(param, name):
            param_bin = self.quantize(param.data).to(torch.int8)
            packed[f"{name}_packed"] = binary_packer(param_bin)
            packed[f"{name}_shape"] = torch.tensor(param.shape, dtype=torch.long)

        pack_param(self.U, "U")
        pack_param(self.V, "V")
        if self.residual:
            pack_param(self.U_R, "U_R")
            pack_param(self.V_R, "V_R")
        return packed

    def state_dict(self, *args, **kwargs):
        prefix = kwargs.get("prefix", "")
        state = super().state_dict(*args, **kwargs)
        keys_to_remove = [
            key
            for key in state.keys()
            if key.startswith(prefix + "U") or key.startswith(prefix + "V")
        ]
        for key in keys_to_remove:
            state.pop(key, None)

        packed_weights = self.pack_weights()
        for key, value in packed_weights.items():
            state[prefix + key] = value
        return state

    @property
    def eff_bit_target(self):
        value = self._eff_bit_target.item()
        return None if value < 0 else value

    @property
    def eff_bit_actual(self):
        return self._eff_bit_actual.item()

    @property
    def split_dim_used(self):
        return int(self._split_dim_final.item())

    @property
    def total_bit_usage(self):
        return self.eff_bit_actual * self.in_features * self.out_features
