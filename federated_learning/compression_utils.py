import torch
import copy

def tensor_delta_mse(a: torch.Tensor, b: torch.Tensor) -> float:
    """
    使用 MSE 函数计算两个 tensor 之间的差距
    """
    delta_mse = torch.mean((a - b) ** 2)
    return float(delta_mse)

def quant_tensor(quant_type: int, t: torch.Tensor) -> dict:
    '''
    quant_type 为量化到何种类型 (1-31)
    32: non quant
    33: half
    1-33: int (由 quant_type 指定, 范围是 1~32)
    '''
    if quant_type == 32:
        qt = t
        t_min, scale = 0., 0.
    elif quant_type == 33:
        qt = t.to(torch.half)
        t_min, scale = 0., 0.
    else:
        qbits = quant_type
        t_min = torch.min(t)
        t_max = torch.max(t)
        scale = (t_max - t_min) / (2 ** qbits - 1)
        qt = torch.round((t - t_min) / scale).to(torch.int64)
    q_result = {
        'qt': qt,
        't_min': t_min,
        'scale': scale,
        'type': quant_type
    }
    return q_result

def dequant_tensor(q_result: dict) -> torch.Tensor:
    if q_result['type'] == 32:
        dqt = q_result['qt']
    elif q_result['type'] == 33:
        dqt = q_result['qt'].to(torch.float32)
    else:
        dqt = q_result['qt'] * q_result['scale'] + q_result['t_min']
        dqt.to(torch.float32)
    return dqt

def quant_and_dequant_tensor(quant_type: int, t: torch.Tensor) -> torch.Tensor:
    q_result = quant_tensor(quant_type, t)
    return dequant_tensor(q_result)

def topk_compression(ratio: float, t: torch.Tensor) -> torch.Tensor:
    """
    ratio 表示选取多少个绝对值最大的元素
    """
    with torch.no_grad():
        bottomk = int(t.nelement() * (1 - ratio))
        _, indices = torch.topk(t.abs(), bottomk, largest=False, sorted=False)
        topk_t = copy.deepcopy(t)
        topk_t[indices] = 0
    return indices, topk_t
