# PyPTO 深度学习：8 个核心问题完整解析

> 分析日期：2026-08-17
> 源码仓库：/data/tianzg/source_code/learn_pypto
> 分支：tianzg_dev

---

## 问题 1：`@pypto.jit` 是如何实现的？

### 通俗解释

`@pypto.jit` 是一个 **Python 装饰器**。它的作用是：当你写一个普通的 Python 函数并用 `@pypto.jit` 装饰后，PyPTO 框架会"拦截"这个函数，不会立即执行它，而是先把它**编译**成昇腾 NPU 能跑的代码，然后在你调用时真正在 NPU 上执行。

### 源码解析

整个 JIT 的入口在 `python/pypto/frontend/__init__.py:67-69`：

```python
from .parser import function, jit
```

实际实现在 `python/pypto/frontend/parser/entry.py:1133-1238`，核心逻辑分三步：

**第一步：装饰器包装（`entry.py:1203-1231`）**

```python
def decorator_wrapper(f: Callable) -> JitCallableWrapper:
    # 捕获闭包变量（函数定义时的局部变量）
    captured_locals = None
    frame = inspect.currentframe()
    if frame and frame.f_back:
        captured_locals = dict(frame.f_back.f_locals)

    # 创建 JitCallableWrapper，但不立即编译（懒加载）
    wrapper = JitCallableWrapper(
        None,       # pto_function 初始为 None
        f,          # 原始 Python 函数
        None,       # handler 初始为 None
        codegen_options=codegen_options,
        host_options=host_options,
        pass_options=pass_options,
        runtime_options=runtime_options,
        ...
    )
    return wrapper
```

关键点：**装饰时只创建 Wrapper，不编译**。真正的编译发生在第一次调用时（懒加载/Lazy Compilation）。

**第二步：首次调用触发编译（`entry.py:297-322`）**

```python
def __call__(self, *args, **kwargs):
    # 1. 解析参数：分离 tensor 参数和非 tensor 参数
    in_tensors, non_tensor_values, input_tensor_defs = self._parse_call_args(args, kwargs)
    # 2. 验证都是 torch.Tensor
    self._validate_exact_torch_tensors(in_tensors)
    # 3. 获取或创建 KernelModule（触发编译）
    self._get_or_create_kmodule(non_tensor_values)
    # 4. 在设备上执行 kernel
    self._execute_kernel(in_tensors, input_tensor_defs)
```

**第三步：编译与执行（`entry.py:468-514`）**

```python
def compile(self, tensors, tensor_defs=None):
    # 1. 将 torch.Tensor 转换为 pypto.Tensor
    args = self._convert_tensors_with_metadata(tensors, tensor_defs)
    # 2. 创建 Parser，解析 Python 函数源码为 AST
    self._parser = self._create_parser()
    self._parser.parse()
    # 3. 设置编译选项
    self._set_config_option()
    # 4. 绑定动态维度
    self._parser.bind_dynamic_dims_to_input_tensors()
    # 5. 执行解析，生成 pypto.Function（编译产物）
    self._pto_function = self._parser.execute()
```

### 整体流程图

```
用户定义函数
     │
     ▼
@pypto.jit 装饰器
     │
     ▼
JitCallableWrapper(f)  ← 懒加载，不编译
     │
     ▼
用户首次调用 kernel(x, y, out)
     │
     ├── _parse_call_args()       → 分离 tensor 和非 tensor 参数
     ├── _get_or_create_kmodule() → 触发编译（有缓存则复用）
     │       ├── _convert_tensors_with_metadata() → torch.Tensor → pypto.Tensor
     │       ├── Parser.parse()                   → Python AST → IR
     │       └── Parser.execute()                 → pypto.Function
     ├── _execute_kernel()        → 在 NPU/SIM 上执行
     │       └── LaunchKernelTorch() / _run_with_cpu()
     └── return None
```

---

## 问题 2：以 add 算子为例，函数调用链路是怎样的？

### 通俗解释

当你在 PyPTO kernel 中写 `x + y` 时，Python 的 `+` 运算符会触发 `Tensor.__add__` 方法，然后层层调用直到 C++ 底层。我们来追踪这个链路。

### 完整调用链路

**第 1 层：Python 运算符重载（`tensor.py:339`）**

```python
def __add__(self, other: 'Tensor | int | float') -> 'Tensor':
    return self.add(other)
```

**第 2 层：Tensor.add 方法（`tensor.py:658-659`）**

```python
def add(self, other: 'Tensor | int | float') -> 'Tensor':
    return pypto.add(self, other)
```

这里调用了 `pypto.add`，它来自 `python/pypto/op/__init__.py` 的导出。

**第 3 层：`op_wrapper` 装饰器（`_op_wrapper.py:46-61`）**

```python
def op_wrapper(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        args = _to_base(args)    # ← 将 pypto.Tensor 转为 pypto_impl.Tensor（C++ 对象）
        kwargs = _to_base(kwargs)
        set_source_location()     # ← 记录源码位置（用于调试）
        out = func(*args, **kwargs)
        clear_source_location()
        return _from_base(out)   # ← 将 C++ 返回值转回 pypto.Tensor
    return wrapper
```

`_to_base` 函数（`_op_wrapper.py:22-30`）把 Python 层的 `Tensor` 对象转换为 C++ 层的 `pypto_impl.Tensor`：

```python
def _to_base(arg):
    if isinstance(arg, (Tensor, Element, ShmemTensor)):
        return arg.base()    # ← 调用 Tensor._base，返回 pypto_impl.Tensor
```

**第 4 层：`pypto.add` 函数（`op/math.py:57-101`）**

```python
@op_wrapper
def add(input_tensor: Tensor, other: Union[Tensor, float, int]) -> Tensor:
    if isinstance(other, pypto_impl.Tensor):
        return pypto_impl.Add(input_tensor, other)       # ← Tensor + Tensor
    else:
        return pypto_impl.Add(input_tensor,
            pypto_impl.Element(input_tensor.dtype, other))  # ← Tensor + 标量
```

**第 5 层：C++ 底层（`pypto_impl.Add`）**

`pypto_impl` 是通过 pybind11 绑定的 C++ 模块。`pypto_impl.Add` 调用 C++ 中的 add 算子实现，在 IR 图中**构建一个 Add 节点**，而不是立即执行计算。

### 完整调用链路图

```
x + y                           # Python 用户代码
  │
  ▼
Tensor.__add__(x, y)            # tensor.py:339
  │
  ▼
Tensor.add(x, y)                # tensor.py:658
  │
  ▼
pypto.add(x, y)                 # op/math.py:58
  │
  ├── op_wrapper 装饰器           # _op_wrapper.py:46
  │     ├── _to_base(): pypto.Tensor → pypto_impl.Tensor
  │     └── _from_base(): pypto_impl.Tensor → pypto.Tensor
  │
  ▼
pypto_impl.Add(a, b)            # C++ 层（pybind11 绑定）
  │
  ▼
在 IR 图中构建 Add 节点           # 构建计算图，不立即执行
  │
  ▼
返回一个新的 pypto_impl.Tensor    # 代表计算结果
```

**关键理解**：PyPTO 和 PyTorch 的 eager 模式不同。PyPTO 调用 `add` 时**不立即执行计算**，而是在**构建一个计算图（IR 图）**。真正的计算发生在整个函数编译完成后，由 NPU 执行。

---

## 问题 3：`set_vec_tile_shapes` 干了什么？

### 通俗解释

**Tile** 是 PyPTO 的核心概念。NPU 上有不同层次的内存（GM 全局内存、L1 缓存、L0 缓存），计算时数据不会一次性全部加载，而是分成小块（Tile）逐步加载和计算。`set_vec_tile_shapes` 就是告诉编译器：**向量计算时，每个 Tile 在每个维度上应该多大**。

### 源码解析

**定义（`_controller.py:77-101`）：**

```python
def set_vec_tile_shapes(*shapes: int):
    """set the tile shapes in vector computation"""
    concrete_shapes = [it.concrete() if isinstance(it, SymbolicScalar) else it for it in shapes]
    pypto_impl.SetScope({"vec_tile_shapes": concrete_shapes})
```

**作用链路**：`set_vec_tile_shapes(32, 32)` → `pypto_impl.SetScope({"vec_tile_shapes": [32, 32]})` → 写入 C++ 配置作用域

### 配置系统的 Scope 机制

在 `config.py:587-666` 中定义了 `_Options` 类，它是一个上下文管理器。`vec_tile_shapes` 是配置项之一（`config.py:599`）：

```python
INIT_FIELDS = [
    ...
    "vec_tile_shapes",
    "cube_tile_shapes",
    "conv_tile_shapes",
    ...
]
```

### 实际效果

以 Hello World 为例：

```python
pypto.set_vec_tile_shapes(32, 32)   # 设置 2D tile 大小为 32x32
out[:] = x + y                       # 后续的 add 操作会使用这个 tile 大小
```

这告诉编译器：
- 将 64×64 的矩阵分成多个 32×32 的小块
- 每个小块在 NPU 的向量计算单元上并行处理
- Tile 大小选择直接影响**计算效率**和**内存带宽利用率**

直观示意：

```
64x64 矩阵，tile_shape = [32, 32]

┌────────┬────────┐
│ Tile   │ Tile   │
│ (0,0)  │ (0,1)  │
│ 32x32  │ 32x32  │
├────────┼────────┤
│ Tile   │ Tile   │
│ (1,0)  │ (1,1)  │
│ 32x32  │ 32x32  │
└────────┴────────┘
```

---

## 问题 4：用户调用 `x + y` 的 Python 代码后，如何生成最终的 NPU 指令？

### 通俗解释

这是 PyPTO 最核心的编译流程。从 Python 代码到 NPU 可执行指令，经历了 **4 层 IR 转换 + 代码生成** 共 5 个阶段。

### 完整流程图

```
┌─────────────────────────────────────────────────────────────────────┐
│ 阶段 0: 前端解析 (Frontend Parsing)                                    │
│ 输入: Python 函数源码                                                  │
│ 输出: Tensor Graph (高层计算图)                                        │
│ 位置: python/pypto/frontend/parser/                                   │
└─────────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 阶段 1: Tensor Graph Passes                                           │
│ 输入: Tensor Graph (操作粒度：add, mul, matmul...)                     │
│ 输出: 优化后的 Tensor Graph                                            │
│ 位置: framework/src/passes/tensor_graph_pass/                         │
│ 主要优化: 算子融合、公共子表达式消除、常量折叠                            │
└─────────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 阶段 2: Tile Graph Passes                                             │
│ 输入: Tensor Graph                                                     │
│ 输出: Tile Graph (操作粒度：Tile 级别的数据搬运和计算)                    │
│ 位置: framework/src/passes/tile_graph_pass/                           │
│ 主要优化: Tile 分块策略、数据复用优化、内存分配                           │
│ 关键: 根据 set_vec_tile_shapes 等配置进行分块                           │
└─────────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 阶段 3: Block Graph Passes                                            │
│ 输入: Tile Graph                                                       │
│ 输出: Block Graph (操作粒度：硬件块级别的调度)                           │
│ 位置: framework/src/passes/block_graph_pass/                          │
│ 主要优化: 硬件核分配、同步插入、流水线调度                                │
└─────────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 阶段 4: CodeGen (代码生成)                                             │
│ 输入: Execution Graph                                                  │
│ 输出: PTO 虚拟指令 → CCE 代码 → NPU 二进制                              │
│ 位置: framework/src/codegen/                                          │
│ 步骤:                                                                  │
│   4a. Execution Graph → PTO 虚拟指令 (codegen.cpp)                     │
│   4b. PTO 虚拟指令 → CCE (Cube Compute Engine) 代码 (codegen_cce.cpp)  │
│   4c. CCE 代码 → NPU 可执行二进制                                       │
└─────────────────────────────────────────────────────────────────────┘
```

### 源码验证

**编译阶段定义（`config.py:24-30`）：**

```python
class CompStage(enum.Enum):
    ALL_COMPLETE = 0
    TENSOR_GRAPH = 1
    TILE_GRAPH = 2
    EXECUTE_GRAPH = 3
    CODEGEN_INSTRUCTION = 4
    CODEGEN_BINARY = 5
```

**编译流水线（`pil/compile_pipeline.py:29-42`）：**

```python
def _build_default_pipeline():
    dce = ir.Pass.aggressive_dce()           # 死代码消除
    canonicalize = ir.Pass.canonicalize()     # 规范化
    merge_stmts = ir.Pass.merge_stmts_into_if()
    create_root_functions = ir.Pass.create_root_functions()
    finalize = ir.Pass.finalize_dynamic_function()

    return [
        ("first_canonicalize_dce", lambda p:dce(canonicalize(p))),
        ("second_canonicalize_dce", lambda p:dce(canonicalize(p))),
        ("canonicalize(merge_stmts)", lambda p:canonicalize(merge_stmts(p))),
        ("create_root_functions", create_root_functions),
        ("finalize", finalize),
    ]
```

**C++ Pass 目录结构证实了 4 层 IR：**

```
framework/src/passes/
├── tensor_graph_pass/    ← 阶段 1: Tensor Graph → 优化后 Tensor Graph
├── tile_graph_pass/      ← 阶段 2: Tensor Graph → Tile Graph
├── block_graph_pass/     ← 阶段 3: Tile Graph → Block Graph
└── algorithms/           ← 通用优化算法

framework/src/codegen/
├── codegen.cpp           ← 阶段 4a: Execution Graph → PTO 虚拟指令
├── codegen_cce.cpp       ← 阶段 4b: PTO 虚拟指令 → CCE 代码
└── npu/                  ← 阶段 4c: CCE 代码 → NPU 二进制
```

### 以 add 为例

```
用户代码: out[:] = x + y
           │
           ▼
阶段 0:  解析为 Tensor Graph 中的一个 Add 节点
         add(x, y) → result
           │
           ▼
阶段 1:  Tensor Graph Passes
         - 可能将 add 与后续操作融合
         - 消除冗余计算
           │
           ▼
阶段 2:  Tile Graph Passes
         - 根据 set_vec_tile_shapes(32, 32) 将 64x64 矩阵分成 4 个 tile
         - 为每个 tile 生成数据搬运 (DMA) 和计算指令
         - Tile(0,0): load x[0:32,0:32], y[0:32,0:32] → add → store out[0:32,0:32]
         - Tile(0,1): load x[0:32,32:64], y[0:32,32:64] → add → store out[0:32,32:64]
         - ...
           │
           ▼
阶段 3:  Block Graph Passes
         - 将 tile 操作分配到 NPU 的计算核 (AIC/AIV)
         - 插入同步屏障 (barrier)
         - 优化流水线 (pipeline)
           │
           ▼
阶段 4:  CodeGen
         - 生成 PTO 虚拟指令 (如 VEC_ADD, DMA_COPY, SYNC 等)
         - 虚拟指令转 CCE (Cube Compute Engine) 代码
         - CCE 编译器生成 NPU 可执行二进制
```

---

## 问题 5：整体代码框架中，哪些部分在 Host 侧，哪些部分在 Device 侧？

### 通俗解释

**Host 侧** = CPU（你的 x86/ARM 服务器），负责编译、调度、内存管理
**Device 侧** = NPU（昇腾 AI 处理器），负责实际的计算执行

### Host/Device 划分

```
┌─────────────────────────────────────────────────────────────────┐
│                        HOST (CPU) 侧                              │
│                                                                   │
│  python/pypto/                                                    │
│  ├── frontend/parser/    ← Python 源码解析 (AST → IR)             │
│  ├── op/                 ← 算子定义 (构建 IR 图)                   │
│  ├── tensor.py           ← Tensor 数据结构                        │
│  ├── config.py           ← 编译选项配置                           │
│  └── converter.py        ← torch.Tensor ↔ pypto.Tensor 转换       │
│                                                                   │
│  framework/src/          ← C++ 编译框架                           │
│  ├── passes/             ← 编译 Pass (Tensor→Tile→Block→Exec)     │
│  ├── codegen/            ← 代码生成 (IR → PTO指令 → CCE → Binary) │
│  ├── interface/          ← 对外接口                               │
│  └── platform/           ← 平台适配                               │
│                                                                   │
│  framework/src/cann_host_runtime/  ← Host 侧运行时                 │
│  ├── 内存分配/释放                                                │
│  ├── 加载 NPU 可执行码到 Device                                    │
│  ├── 启动 Device 执行 (LaunchKernelTorch)                          │
│  └── 同步等待 Device 完成                                          │
└─────────────────────────────────────────────────────────────────┘
                              │
                     PCIe / HCCS 总线
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      DEVICE (NPU) 侧                               │
│                                                                   │
│  NPU 上执行的可执行码 (由 CodeGen 生成)                             │
│  ├── 向量计算指令 (AIV - AI Vector)                                │
│  │   - add, mul, sub, div, exp, log, sigmoid...                   │
│  ├── 矩阵计算指令 (AIC - AI Cube)                                  │
│  │   - matmul, conv...                                            │
│  ├── 数据搬运指令 (DMA)                                            │
│  │   - GM → L1, L1 → L0, L0 → L1, L1 → GM                        │
│  └── 同步/控制指令                                                 │
│      - barrier, fence...                                          │
│                                                                   │
│  NPU 内存层次 (由编译时 Tile 配置决定数据驻留)                       │
│  ├── GM  (Global Memory)    ← 全局内存 (大容量，高延迟)              │
│  ├── L1  (Level 1 Cache)    ← 一级缓存 (中等容量)                   │
│  └── L0  (Level 0 Cache)    ← 零级缓存 (小容量，低延迟，计算近邻)    │
└─────────────────────────────────────────────────────────────────┘
```

### 源码验证

**Host 侧执行入口（`entry.py:636-660`）：**

```python
def _execute_kernel(self, torch_tensors, tensor_defs):
    if self._runtime_options.get("run_mode", None) == RunMode.NPU:
        pypto_impl.LaunchKernelTorch(self, _current_stream(), torch_tensors, tensor_defs)
        # ↑ Host 侧：将编译好的 kernel 加载到 NPU 并启动执行
    else:
        # SIM 模式：在 CPU 上模拟执行
        self._run_with_cpu(pto_tensors, [])
```

**Host 侧运行时执行（`entry.py:975-1018`）：**

```python
def _run(self, in_tensor_data, out_tensor_data, device, ctrl_cache=0):
    # 1. 查询需要的 workspace 大小
    workspace_size = pypto_impl.GetWorkSpaceSize(self._handler, in_tensor_data, out_tensor_data)
    # 2. 在设备上分配 workspace
    workspace_tensor = torch.empty(workspace_size, dtype=torch.uint8, device=device)
    # 3. 启动设备执行
    runtime_error_msg = pypto_impl.OperatorDeviceRunOnceDataFromDevice(
        self._handler,
        in_tensor_data + out_tensor_data,
        [],
        torch.npu.current_stream().npu_stream,
        workspace_tensor.data_ptr(),
        ctrl_cache,
    )
    # ↑ 这是 Host 侧调用 NPU 驱动的入口，驱动将任务下发到 NPU
```

**C++ Host 侧运行时目录：**

```
framework/src/cann_host_runtime/   ← Host 侧运行时实现
```

**C++ CodeGen 生成的 Device 侧代码：**

```
framework/src/codegen/
├── codegen_cce.cpp   ← 生成 CCE (NPU 设备侧) 代码
└── npu/              ← NPU 设备侧指令生成
```

### 关键理解

| 方面 | Host 侧 | Device 侧 |
|------|---------|-----------|
| **硬件** | CPU | NPU (昇腾) |
| **Python 代码** | 全部在 Host 侧 | 无 |
| **编译** | 全部在 Host 侧 | 无 |
| **计算** | 内存分配、调度 | 实际 add/mul/matmul... |
| **数据** | 管理数据指针 | 持有实际数据 |
| **运行时** | LaunchKernelTorch | 执行 PTO 指令 |

---

## 问题 6：融合算子是什么？如何实现的？

### 通俗解释

**融合算子**（Operator Fusion）是指将多个连续的算子合并成一个算子，在**一次 NPU 调用**中完成。这避免了中间结果在 GM（全局内存）和 L1 缓存之间反复搬运，大幅提升性能。

### 直观示例

在 `examples/02_intermediate/operators/activation/activation.py` 中，SiLU 函数就是一个融合算子：

```python
# SiLU: x * sigmoid(x) = x / (1 + exp(-x))
out[:] = x * pypto.sigmoid(x)
```

这涉及 3 个操作：
1. `neg(x)` — 取负
2. `exp(neg_x)` — 指数运算
3. `add(1, exp_x)` — 加 1
4. `div(x, add_result)` — 除法

**没有融合**：每个操作单独执行，中间结果都要写回 GM 再读出
**有融合**：所有操作在 NPU 上一次完成，中间结果在 L0/L1 缓存中直接传递

```
没有融合：
  GM → L1 → L0 → neg → L1 → GM    (写回中间结果)
  GM → L1 → L0 → exp → L1 → GM    (写回中间结果)
  GM → L1 → L0 → add → L1 → GM    (写回中间结果)
  GM → L1 → L0 → div → L1 → GM    (写回最终结果)
  共 4 次 GM 读写

有融合：
  GM → L1 → L0 → neg → exp → add → div → L1 → GM
  共 1 次 GM 读、1 次 GM 写
```

### 融合算子的实现机制

PyPTO 的融合算子在**编译阶段**自动实现，关键在 Tensor Graph Passes 和 Tile Graph Passes 中。

**级别 1：Tensor Graph 融合（`framework/src/passes/tensor_graph_pass/`）**

在 Tensor Graph 层面，编译器会识别可以融合的算子模式。例如：
- `mul(x, sigmoid(x))` → 识别为 SiLU 模式
- `add(x, y)` 和 `mul(result, z)` → 可能融合为一个 kernel

**级别 2：Tile Graph 融合（`framework/src/passes/tile_graph_pass/`）**

在 Tile Graph 层面，多个 Tile 操作被合并成一个 Super Node（超级节点），在同一个 NPU 核上执行，通过 L1 缓存直接传递数据。

**配置控制融合（`config.py:213-300`）：**

```python
def set_pass_options(
    *,
    vec_nbuffer_setting=None,      # 向量融合配置
    cube_l1_reuse_setting=None,    # Cube L1 复用配置
    cube_nbuffer_setting=None,     # Cube 融合配置
    sg_set_scope=None,             # 超级节点 scope 配置
    ...
):
```

### 以 SwiGLU 为例

```python
# activation.py:191-205
@pypto.frontend.jit(runtime_options={"run_mode": global_run_mode})
def swiglu_activation_kernel(gate: pypto.Tensor(), up: pypto.Tensor(), out: pypto.Tensor()):
    sigmoid = pypto.sigmoid(gate)    # op1: sigmoid
    swish = gate * sigmoid            # op2: mul
    out[:] = swish * up               # op3: mul
```

编译后，这 3 个操作（sigmoid + mul + mul）被融合成一个 NPU kernel，所有中间数据在 L0/L1 缓存中传递，不经过 GM。

---

## 问题 7：用户传入的 `x` 和 `y` 是 `torch.Tensor`，如何变成 pypto 的 Tensor？两者关系是什么？

### 通俗解释

PyPTO 本身不管理内存，它**复用 PyTorch 的内存**。`torch.Tensor` 和 `pypto.Tensor` 的关系是：**pypto.Tensor 指向 torch.Tensor 的同一块内存（通过 data_ptr），只是在 PyPTO 编译框架中记录了 Tensor 的元信息（shape、dtype、format 等）。**

### 源码解析

**转换入口（`converter.py:79-174`）：**

```python
@_count_calls
def from_torch(
    tensor,
    name: str = "",
    dynamic_axis: Optional[List[int]] = None,
    tensor_format: Optional[TileOpFormat] = None,
    dtype: Optional[DataType] = None,
):
    # 1. 类型检查：必须是 torch.Tensor
    if not isinstance(tensor, torch.Tensor):
        raise FeError(TypeError("input type is not currently supported."))

    # 2. dtype 转换：torch.float32 → pypto.DT_FP32
    dtype = _dtype_from(tensor.dtype) if dtype is None else dtype

    # 3. format 推断：如果是 NPU tensor 且格式为 NZ (29)，则使用 NZ
    if tensor_format is None:
        tensor_format = TileOpFormat.TILEOP_ND
        if tensor.device.type == "npu":
            torch_npu = get_torch_npu()
            if torch_npu is not None and torch_npu.get_npu_format(tensor) == 29:
                tensor_format = TileOpFormat.TILEOP_NZ

    # 4. 创建 pypto.Tensor，关键：复用 torch.Tensor 的 data_ptr
    return Tensor(
        shape=dyn_shape,
        dtype=dtype,
        name=name,
        data_ptr=tensor.data_ptr(),   # ← 直接复用 torch.Tensor 的内存地址！
        format=tensor_format,
        device=tensor.device,
        ori_shape=list(tensor.shape),
    )
```

**调用时机（`entry.py:410-429`）：**

```python
@staticmethod
def _convert_tensors_with_metadata(torch_tensors, tensor_defs):
    pto_tensors = []
    for torch_tensor, tensor_def in zip(torch_tensors, tensor_defs):
        pto_tensors.append(
            pypto.from_torch(
                torch_tensor,
                name=tensor_def.name,
                dynamic_axis=dynamic_axis if dynamic_axis else None,
                dtype=tensor_def.explicit_dtype,
                tensor_format=tensor_def.explicit_format,
            )
        )
    return pto_tensors
```

### torch.Tensor 和 pypto.Tensor 的关系

```
torch.Tensor                          pypto.Tensor
┌─────────────────────┐              ┌──────────────────────┐
│ .data_ptr() = 0x1000│──────────────│ .data_ptr = 0x1000   │  ← 指向同一块 NPU 内存
│ .shape = (64, 64)   │              │ .shape = [64, 64]    │
│ .dtype = float32     │              │ .dtype = DT_FP32     │
│ .device = npu:0      │              │ .device = npu:0      │
│ 内存管理: PyTorch     │              │ .format = TILEOP_ND  │  ← PyPTO 特有
│ 实际数据: [1.0, 2.0..]│              │ .name = "x"          │  ← PyPTO 特有
└─────────────────────┘              │ .ori_shape = [64,64] │  ← PyPTO 特有
                                     │ 内存管理: 无 (复用)    │
                                     └──────────────────────┘
```

**关键点**：
1. **内存共享**：`pypto.Tensor` 的 `data_ptr` 就是 `torch.Tensor` 的 `data_ptr`，指向同一块 NPU 内存
2. **PyPTO 不负责内存分配和释放**：由 PyTorch 管理
3. **PyPTO 负责元信息**：记录 shape、dtype、format、name 等编译所需信息
4. **计算在 NPU 上原地执行**：NPU 计算直接修改 torch.Tensor 指向的内存，用户无需额外拷贝

**dtype 转换表（`converter.py:177-195`）：**

```python
_dtype_dict = {
    "torch.float16": DataType.DT_FP16,
    "torch.bfloat16": DataType.DT_BF16,
    "torch.float32": DataType.DT_FP32,
    "torch.int8": DataType.DT_INT8,
    "torch.int32": DataType.DT_INT32,
    ...
}
```

---

## 问题 8：如何开始开发一个简单的 add 融合算子？

### 完整步骤

以 **add + relu 融合算子** 为例（即 `out = relu(x + y)`）。

#### 步骤 1：创建 Python 文件

```python
#!/usr/bin/env python3
"""
自定义 add+relu 融合算子示例
out = relu(x + y)
"""

import argparse
import os
import sys

import torch
import pypto


def get_run_mode():
    """获取运行模式"""
    if 'TILE_FWK_DEVICE_ID' in os.environ:
        return pypto.RunMode.NPU
    return pypto.RunMode.SIM


run_mode = get_run_mode()


# ═══════════════════════════════════════════════════════════════
# 步骤 2: 定义融合算子 kernel
# ═══════════════════════════════════════════════════════════════
@pypto.jit(runtime_options={"run_mode": run_mode})
def add_relu_kernel(
    x: pypto.Tensor[...],     # 输入 1，自动推断 shape/dtype
    y: pypto.Tensor[...],     # 输入 2
    out: pypto.Tensor[...],   # 输出
):
    # 步骤 2a: 设置 Tile 大小（关键！影响性能）
    pypto.set_vec_tile_shapes(32, 32)

    # 步骤 2b: 融合计算
    # add + relu 被融合在一次 kernel 调用中完成
    tmp = x + y               # add 操作
    out[:] = pypto.relu(tmp)  # relu 操作，结果写入 out


# ═══════════════════════════════════════════════════════════════
# 步骤 3: 编写验证代码
# ═══════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_mode", choices=["npu", "sim"], default="npu")
    args = parser.parse_args()

    # 步骤 3a: 设置设备
    if args.run_mode == "npu":
        device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
        torch.npu.set_device(device_id)
        device = f"npu:{device_id}"
    else:
        device = "cpu"

    # 步骤 3b: 创建输入数据
    shape = (64, 64)
    x = torch.randn(shape, dtype=torch.float32, device=device)
    y = torch.randn(shape, dtype=torch.float32, device=device)
    out = torch.empty(shape, dtype=torch.float32, device=device)

    # 步骤 3c: 调用融合 kernel
    add_relu_kernel(x, y, out)

    # 步骤 3d: 验证结果
    expected = torch.relu(x + y)  # PyTorch 参考实现
    torch.testing.assert_close(out, expected, atol=1e-3, rtol=1e-3)
    print("✓ add_relu 融合算子测试通过！")

    # 步骤 3e: 查看编译产物（可选）
    func = pypto.get_last_function()
    if func:
        print(f"\n编译的函数名: {func.raw_name}")
        print(f"函数类型: {func.function_type_str}")


if __name__ == "__main__":
    main()
```

#### 步骤 2：关键决策点

**2.1 Tile 大小选择**

```python
pypto.set_vec_tile_shapes(32, 32)
```

| Tile 大小 | 效果 |
|-----------|------|
| 较小 (如 16, 16) | 更多 Tile、更多 DMA 次数、但单次计算更少 |
| 较大 (如 64, 64) | 更少 Tile、更少 DMA 次数、但 L1 缓存可能不够 |
| 32, 32 | 经验值，通常是不错的起点 |

**2.2 输出方式**

```python
# 方式 1: 全切片赋值（推荐）
out[:] = pypto.relu(x + y)

# 方式 2: 使用 assemble
pypto.assemble(pypto.relu(x + y), [0, 0], out)
```

**2.3 不使用 return**

PyPTO kernel **不支持 return 值**，必须通过修改输出参数来传递结果。

#### 步骤 3：更复杂的融合示例

```python
@pypto.jit(runtime_options={"run_mode": run_mode})
def add_mul_relu_kernel(
    x: pypto.Tensor[...],
    y: pypto.Tensor[...],
    z: pypto.Tensor[...],
    out: pypto.Tensor[...],
):
    pypto.set_vec_tile_shapes(32, 32)

    # 融合: add → mul → relu
    # 所有操作融合在一次 NPU launch 中
    tmp1 = x + y           # add
    tmp2 = tmp1 * z        # mul
    out[:] = pypto.relu(tmp2)  # relu
```

#### 步骤 4：运行和调试

```bash
# NPU 模式运行
export TILE_FWK_DEVICE_ID=0
python add_relu_fusion.py --run_mode npu

# SIM 模式运行（不需要 NPU 硬件）
python add_relu_fusion.py --run_mode sim
```

### 开发 checklist

| 步骤 | 说明 |
|------|------|
| ✅ 1. 导入 `pypto` 和 `torch` | 两个都必不可少 |
| ✅ 2. 用 `@pypto.jit` 装饰 kernel | 触发 JIT 编译 |
| ✅ 3. 用 `pypto.Tensor[...]` 标注参数 | 自动推断 shape/dtype |
| ✅ 4. 设置 `set_vec_tile_shapes` | 控制 Tile 大小 |
| ✅ 5. 用 `out[:] = ...` 写输出 | 不要用 return |
| ✅ 6. 用 PyTorch 创建输入输出 tensor | 复用 PyTorch 内存 |
| ✅ 7. 用 PyTorch 验证结果 | 对比参考实现 |

---

## 总结

这 8 个问题覆盖了 PyPTO 的完整开发链路：

| 问题 | 核心概念 | 关键源码位置 |
|------|---------|-------------|
| 1. JIT 实现 | `JitCallableWrapper` 懒加载编译 | `entry.py:1133-1238` |
| 2. add 调用链路 | `__add__` → `pypto.add` → `pypto_impl.Add` | `tensor.py:339`, `op/math.py:57-101` |
| 3. Tile 设置 | 配置 NPU 数据分块大小 | `_controller.py:77-101` |
| 4. 编译流程 | Tensor → Tile → Block → Exec → CodeGen | `config.py:24-30`, `framework/src/passes/` |
| 5. Host/Device | Host 编译+调度，Device 执行计算 | `entry.py:636-660`, `entry.py:975-1018` |
| 6. 融合算子 | 多算子合并，减少 GM 读写 | `activation.py:107-120`, `config.py:213-300` |
| 7. torch/pypto Tensor | 共享内存，pypto 只记录元信息 | `converter.py:79-174` |
| 8. 开发 add 融合 | 上述所有知识的综合应用 | 本文步骤 1-4 |