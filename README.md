# code-mcp

面向 AI 编码代理的本地代码上下文 MCP：精准符号切片、因果依赖打包、定点写回、健康度审查。

## 为什么

AI 改代码时习惯读整个文件、写整个文件。文件一大，读进去的 token 大半和这次改动无关，
成本被放大，上下文也被稀释 —— 改到后面几轮，前面的内容已经被压缩掉了。

code-mcp 让代理只取「要害」：

- **精准读**：按符号取片段，附上调用方 / 被调用方 / import / 相关测试
- **精准写**：只替换目标符号的字节区间，其余字节原样不动
- **自证收益**：每次返回都报告相对「直接读文件」的实际开销。省不动就不返回内容

## 它到底值不值

先说结论，免得期望错位。

我们拿一个 472 行的 Python 仓库做了三轮真实任务演练：植入一个跨文件的 API 变更 bug，
只允许用本工具修复，不允许直接读整个文件。

| 轮次 | 改动 | 调用次数 | token |
|---|---|---:|---:|
| 第一轮 | 初版 | 26 | 5,304 |
| 第二轮 | 修完所有功能缺陷 | 19 | 5,850 |
| 第三轮 | 专门做减法 | 19 | **3,735** |

**第二轮那 +10% 是最值得记住的一条**：功能全修好了，token 反而涨了。因为每一处修复
都是「让输出更完整」，而完整度是要花钱的。第三轮什么都不加，只砍输出，才降回 36%。

**它的价值随文件规模增长，不是普适的：**

| 场景 | 实际扛事的工具 | 效果 |
|---|---|---|
| 几百行的中小仓库 | `find` + `slice` + `edit` | token 与直接读文件打平，胜在 edit 零失败 |
| 上千行的大函数 | `pack`（分段大纲） | `TavernView` 1085 行，省 92.8% |
| 跨语言改签名 | `impact`（Tauri 桥接） | Rust 命令 → 前端调用点，一路到底 |

`pack` 只有在「目标很大、邻居很小」时才划算。小文件上它会直接告诉你读原文件。

## 实测

节省率按**响应实际 token** 计，含 JSON 信封开销（每片的文件名、符号名、行号等）：

| 仓库 | 样本 | 中位节省 | 均值 | 最差 |
|---|---:|---:|---:|---:|
| rwmod（Python，128 文件 / 2885 符号） | 40 | 71.4% | 69.4% | +11.7% |
| my-chat（TS+Rust，270 文件 / 2920 符号） | 30 | 80.9% | 74.2% | +14.1% |

全量索引 rwmod 1.1s、my-chat 2.1s；无变更时增量 0.1s 级。

单符号最大收益：`TavernView.tsx` 的 1085 行组件，正文 14,018 token、整文件 14,772 token，
分段后响应 4,078 token，省 72.4%（正文口径 92.8%）。

## 安装

```bash
git clone <repo> && cd code-mcp
uv sync
```

MCP 配置（Claude Code / Cherry Studio 等）：

```json
{
  "code-mcp": {
    "command": "uv",
    "args": ["run", "--directory", "<repo-path>", "code-mcp-server"]
  }
}
```

## 工具

| 工具 | 作用 | 什么时候用 |
|---|---|---|
| `index` | 建立 / 增量更新索引 | 每个仓库第一次 |
| `find` | 按名字或注释文字定位符号 | 知道要找什么、不知道在哪，替代 grep |
| `pack` | 目标符号 + 因果邻域 + 成本对比 | 目标文件大、邻居少的时候 |
| `slice_symbol` | 精准读单个符号或行区间 | 已经知道符号名，最常用 |
| `edit` | 定点替换符号 / 常量 / import | 替代整文件重写 |
| `impact` | 改动影响面，带行号 | 改签名 / 重命名 / 删除前 |
| `review` | 外部 linter + 结构规则 | 提交前体检 |
| `gate` | 跑项目门禁命令 | 改完验证 |

CLI 同名：`code-mcp index|find|pack|slice|edit|impact|review|gate`，
`--json` 输出机器可读格式，每条响应都带 `response_tokens` 字段。

## 打包策略

`pack` 的成本判断按**序列化后的实际大小**算，不只看正文字数 —— JSON 信封本身要花钱。

三道闸：

1. 文件低于 300 token → 直接拒绝，告诉你读原文件
2. 打包结果 ≥ 文件本身 → **不返回内容**，只回「读原文件」的建议
3. 预算超支 → 按 import → callee → caller → 测试的顺序丢弃，测试保留到最后

目标符号过大（默认 4000 token 以上）时改返回**签名 + 分段大纲**：

```
target <file>:<start>-<end> [function] (outline, 14 segments, 14018 tokens)
  #1  L49-159    def TavernView
  #2  L160-262   const graph = useMemo(() => {
```

再用 `segment=N` 或 `slice_symbol(start_line, end_line)` 取具体那段。

想要被拒绝的结果时用 `force=true`：跳过前两道闸，但如果全量打包仍比读文件贵，
会自动降级为只给目标符号（`target_only`），并在 note 里说明成本对比。

## 配置

仓库根放 `.code-mcp.toml`（参考 `.code-mcp.toml.example`）：

```toml
[project]
exclude = ["sample_data", "vendor"]

[gate]
[[gate.steps]]
name = "tests"
command = "python run_tests.py"
timeout = 900
```

没有配置文件时，gate 从 marker 文件自动探测（`run_tests.py` / `pyproject.toml` /
`Cargo.toml` / `package.json`）。

索引数据库存在 `~/.code-mcp/indexes/<hash>.db`，**不会往被索引的仓库里写任何文件**。
可用 `CODE_MCP_HOME` 改位置。

## 索引范围

- 在 git 仓库里用 `git ls-files`，比遍历目录快两个数量级（有个项目有 14.6 万个未跟踪文件）
- 非 git 目录退回目录遍历
- 自动跳过 minified / bundle 文件、超过 1.5MB 的文件

## 索引保鲜

文件在索引之外被改过（编辑器、git checkout、另一个会话），按旧偏移切片会切出**隔壁函数**的代码。
所以每次读取前都会比对 mtime 和大小，漂移就重新索引那一个文件再继续。

`edit` 写盘后会重建该文件的符号偏移、引用和跨语言桥接，连续编辑不会错位。

## 配合 AI 使用

工具装了不等于会被用。两个办法：

1. 在项目 `CLAUDE.md` 写一句「改代码前先调 `pack` 或 `slice`，不要整文件读」
2. 挂 PreToolUse hook 拦截大文件读取：

```json
{
  "hooks": {
    "PreToolUse": [{
      "matcher": "Read",
      "hooks": [{
        "type": "command",
        "command": "python -m code_mcp.hooks --enforce"
      }]
    }]
  }
}
```

**`--enforce` 不能省**：不带它时 hook 退出码是 0，stderr 只会显示给用户，模型看不到；
带上它才会以退出码 2 阻断读取，让提示真正进入模型上下文。

## 已知边界

诚实说明，别指望它做不到的事：

- **调用关系是证据分级的名字匹配，不是类型解析**。同文件、或 import 能指向目标文件的，
  标为 `exact`；仅全局同名的默认**不建边**。要精确到类型级需要接 LSP，这一版没做。
- **Tauri 前后端桥接是写死规则**：`#[tauri::command]` ↔ `invoke("name")` 字符串字面量匹配。
  其他跨语言字符串约定（如自定义 IPC）不识别。
- **动态调用、反射、字符串拼接出来的调用**看不见。
- **`impact` 有过度近似**：改的是引用关系，不是语义。已经迁移过的调用点仍会列出来。
- **TS 的对象字面量函数会被索引成方法**（`useStore.sendMessage`），这是为了让 store action
  可寻址；副作用是所有对象字面量里的函数都成了符号。

## 开发

```bash
uv run pytest                                  # 226 tests
uv run python scripts/bench.py <repo> --sample 30
```

`scripts/bench.py` 报的是信封口径的节省率 —— 和工具自己判断划算与否用的是同一个数。

## License

MIT
