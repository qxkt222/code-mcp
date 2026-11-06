# code-mcp

面向 AI 编码代理的本地代码上下文 MCP：精准符号切片、因果依赖打包、健康度审查与定点写回。

## 为什么

AI 改代码时习惯读整个文件、写整个文件。文件一大，读进去的 token 大半和这次改动无关，
成本被放大，上下文也被稀释 —— 改到后面几轮，前面的内容已经被压缩掉了。

code-mcp 让代理只取「要害」：

- **精准读**：按符号取片段，附上调用方 / 被调用方 / import / 相关测试
- **精准写**：只替换目标符号的字节区间，其余字节原样不动
- **自证收益**：每次返回都报告相对「直接读文件」省了多少 token。省不动就直说「直接读文件」

## 实测

在 rwmod（Python，266 文件 / 2885 符号）上采样 30 个函数：

| 指标 | 数值 |
|---|---|
| 全量索引 | 1.07s |
| 增量索引（无变更） | 0.10s |
| pack 节省中位数 | **84.5%** |
| pack 节省均值 | 77.2% |
| 最差样本 | 32.3% |

在 code-mcp 自身（小文件为主）上采样 25 个函数，中位数 65.5%。文件越小收益越低，
小到一定程度工具会直接建议你读原文件。

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
| `index` | 建立 / 增量更新索引 | 每个仓库第一次，以及大批外部改动后 |
| `find` | 按名字或注释文字定位符号 | 知道要找什么、不知道在哪，替代 grep |
| `pack` | 目标符号 + 因果邻域 + token 对比 | **替代读整个文件**，改代码前的标准动作 |
| `slice_symbol` | 精准读单个符号或行区间 | 已经知道符号名 |
| `edit` | 定点替换符号实现 | 替代整文件重写 |
| `impact` | 改动影响面，按距离分级 | 改签名 / 重命名 / 删除前 |
| `review` | 外部 linter + 结构规则 | 提交前体检 |
| `gate` | 跑项目门禁命令 | 改完验证 |

同名 CLI 都有：`code-mcp index|find|pack|slice|edit|impact|review|gate`，
加 `--json` 输出机器可读格式。

## 打包策略

`pack` 返回四类上下文，按这个顺序丢弃（预算超支时）：
**import → callee → caller → 测试**。测试保留到最后。

目标符号永远给全文。周边符号默认只给签名，只有预算宽松**且**总开销仍低于
直接读文件 70% 时才升级为全文 —— 这是防止「打包比读文件还贵」的硬约束。

返回里的 `worth_using=false` 表示打包不划算，`note` 会直接建议读原文件。

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

- 在 git 仓库里用 `git ls-files`（含未跟踪但未被忽略的文件），比遍历目录快两个数量级
- 非 git 目录退回目录遍历
- 自动跳过 minified / bundle 文件、超过 1.5MB 的文件

## 配合 AI 使用

工具装了不等于会被用。两个办法：

1. 在项目 `CLAUDE.md` 写一句「改代码前先调 `pack`，不要整文件读」
2. 挂 PreToolUse hook 拦截大文件读取：

```json
{
  "hooks": {
    "PreToolUse": [{
      "matcher": "Read",
      "hooks": [{
        "type": "command",
        "command": "python -m code_mcp.hooks"
      }]
    }]
  }
}
```

hook 会在 Read 的文件超过 300 行时提示改用 `pack`。

## 已知边界

诚实说明，别指望它做不到的事：

- **调用关系是名字匹配，不是精确解析**。`obj.foo()` 里 `foo` 指向哪个定义，靠名字和
  限定名唯一性判断；同名多个时不猜。要精确到类型级，需要接 LSP（pyright / rust-analyzer），
  这一版没做。
- **Tauri 前后端桥接是写死规则**：`#[tauri::command]` ↔ `invoke("name")` 字符串字面量匹配。
  其他跨语言字符串约定（如自定义 IPC）不识别。
- **动态调用、反射、字符串拼接出来的调用**一概看不见。
- `edit` 要求索引是最新的：文件在索引之外被改过会拒绝写入，需要先 `index`。

## 开发

```bash
uv run pytest          # 106 tests
uv run python scripts/bench.py <repo> --sample 30
```

## License

MIT
