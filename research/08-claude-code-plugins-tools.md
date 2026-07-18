# Claude Code — Plugin & công cụ tăng hiệu suất coding

> Khác với file 01–07 (paper notes), file này là **ghi chú công cụ/workflow**: plugin của Claude Code là gì, tìm ở đâu, và cái nào đáng dùng cho đồ án này. Cập nhật: 2026-07.

## Plugin của Claude Code là gì

Một **plugin** = gói đóng sẵn có thể chứa 5 loại thành phần:

1. **Slash commands** — lệnh `/tên-lệnh` tự định nghĩa (ví dụ `/commit-and-pr`).
2. **Subagents** — agent con chuyên biệt (reviewer, security auditor…).
3. **Skills** — bộ hướng dẫn chuyên sâu Claude tự nạp khi gặp đúng loại task.
4. **Hooks** — script chạy tự động trước/sau tool call (format code sau khi edit, chặn lệnh nguy hiểm…).
5. **MCP servers** — kết nối Claude Code với dịch vụ ngoài (GitHub, database, browser…).

Plugin được phân phối qua **marketplace** (một repo GitHub chứa file `marketplace.json`). Quy trình 2 bước:

```
/plugin marketplace add <owner>/<repo>     # đăng ký catalog
/plugin install <tên-plugin>@<marketplace> # cài plugin từ catalog đó
```

Marketplace chính thức của Anthropic (`claude-plugins-official`) có sẵn ngay khi cài Claude Code — gõ `/plugin` → tab **Discover** để duyệt. Hệ sinh thái hiện khá lớn: ~190 marketplace cộng đồng với ~2.500 plugin (thống kê 6/2026 của repo tracker bên dưới).

## Nguồn chính thức (đọc trước tiên)

- **Docs: Discover and install prebuilt plugins** — <https://code.claude.com/docs/en/discover-plugins>
  Hướng dẫn chính thức: khái niệm marketplace, lệnh `/plugin`, cách quản lý/gỡ.
- **anthropics/claude-plugins-official** — <https://github.com/anthropics/claude-plugins-official>
  Directory plugin chất lượng cao do Anthropic quản lý (plugin nội bộ của Anthropic + plugin đối tác/cộng đồng đã duyệt). Catalog dạng web: <https://claude.com/plugins>.

## Danh sách tổng hợp (awesome lists) trên GitHub

- **hesreallyhim/awesome-claude-code** — <https://github.com/hesreallyhim/awesome-claude-code>
  Awesome list nổi nhất, tuyển chọn tay: skills, agents, status lines, dev tooling, plugins. Điểm khởi đầu tốt nhất để "dạo chợ".
- **ComposioHQ/awesome-claude-plugins** — <https://github.com/ComposioHQ/awesome-claude-plugins>
  Curated list tập trung riêng vào plugin (commands/agents/hooks/MCP qua plugin system).
- **rohitg00/awesome-claude-code-toolkit** — <https://github.com/rohitg00/awesome-claude-code-toolkit>
  Toolkit "tất cả trong một": 135 agents, 35 skills, 42 commands, 176+ plugins, 20 hooks, 14 MCP configs… Nhiều nhưng cần tự sàng lọc.
- **jmanhype/awesome-claude-code** — <https://github.com/jmanhype/awesome-claude-code>
  List plugin + MCP servers + editor integrations + tài liệu học.
- **Chat2AnyLLM/awesome-claude-plugins** — <https://github.com/Chat2AnyLLM/awesome-claude-plugins>
  Tracker tự động đếm số marketplace/plugin qua GitHub API — dùng để xem bức tranh toàn cảnh hệ sinh thái.
- **jqueryscript/awesome-claude-code** — <https://github.com/jqueryscript/awesome-claude-code>
  List thiên về tools, IDE integrations, frameworks.

## MCP servers đáng chú ý cho coding

Theo nhiều bài tổng hợp 2026 ([Builder.io](https://www.builder.io/blog/best-mcp-servers-2026), [Firecrawl](https://www.firecrawl.dev/blog/best-mcp-servers-for-developers), [Totalum](https://www.totalum.app/blog/best-mcp-servers-2026)), bộ ba **Context7 + GitHub MCP + Playwright MCP** phủ ~80% workflow coding:

| MCP server | Dùng làm gì |
|---|---|
| **Context7** | Nạp docs **mới nhất** của thư viện vào context — tránh việc LLM viết code theo API bản cũ |
| **GitHub MCP** | PR, issues, code search ngay trong terminal, khỏi chuyển qua web UI |
| **Playwright MCP** | Điều khiển browser: test UI, tự động hóa web |
| **PostgreSQL / Supabase MCP** | Query schema + data thật khi viết code chạm database |
| **E2B MCP** | Sandbox cloud an toàn để agent thực thi code (Python/JS, cài package, chạy shell) |
| **Figma MCP** | Design-to-code từ layout thật thay vì đoán qua screenshot |

⚠️ Lưu ý chung từ các bài review: **chọn đúng 2–3 server mình thật sự cần**, đừng bật cả chục cái — mỗi MCP server tốn context window và tăng bề mặt rủi ro.

## Công cụ hiệu suất cộng đồng (ngoài plugin)

- **ccusage** — <https://github.com/ryoppippi/ccusage>
  CLI xem chi phí/token usage của Claude Code theo ngày/phiên. Gần như "must-have" để biết mình đang đốt bao nhiêu.
- **davila7/claude-code-templates** — <https://github.com/davila7/claude-code-templates>
  CLI cấu hình + giám sát Claude Code: bộ sưu tập agents, commands, settings, hooks, MCP configs, project templates dùng ngay.
- **SuperClaude Framework** — <https://github.com/SuperClaude-Org/SuperClaude_Framework>
  Framework mở rộng Claude Code bằng bộ commands/personas có cấu trúc — phổ biến nhưng khá "nặng", nên thử sau khi đã quen bản gốc.
- **pchalasani/claude-code-tools** — <https://github.com/pchalasani/claude-code-tools>
  Tập tool thực dụng (session search, tmux integration, git guard…) cho Claude Code và các CLI agent tương tự.

## Tính năng có sẵn — không cần cài gì (đừng bỏ qua)

Trước khi cài plugin, tận dụng những thứ built-in:

- **CLAUDE.md** — repo này đã có, đây chính là "plugin" quan trọng nhất: quy ước dự án được nạp mỗi phiên.
- **Skills có sẵn**: `/code-review` (soát bug diff hiện tại), `/security-review`, `/simplify`, `/init`.
- **Plan mode** — bắt Claude lên kế hoạch và được duyệt trước khi sửa code.
- **Hooks & settings** — tự cấu hình trong `.claude/settings.json` không cần plugin ngoài.
- **Subagents** — tách task tìm kiếm/khám phá lớn sang agent con để giữ context chính sạch.

## Áp dụng cho đồ án này (Vietnamese Chatbot RLHF)

Repo này chủ yếu là **notebook chạy trên Kaggle** chứ không phải app chạy local, nên thứ tự ưu tiên:

1. **Context7 MCP** — hữu ích nhất: API của `trl` (DPOTrainer), `unsloth`, `transformers`, `datasets` đổi liên tục giữa các version; docs mới giúp tránh viết notebook theo API đã deprecated (đúng vấn đề từng gặp với `datasets >= 3.0` ở §6.1).
2. **GitHub MCP** — nếu bắt đầu quản lý issue/PR cho repo này.
3. **ccusage** — theo dõi usage khi các phiên làm việc dài (viết notebook, sửa spec).
4. Plugin marketplace — mức độ ưu tiên thấp hơn; dạo `hesreallyhim/awesome-claude-code` khi rảnh để xem có gì hay.

## Đọc gì trước

Docs chính thức (discover-plugins) → `anthropics/claude-plugins-official` (xem chuẩn plugin "xịn" trông thế nào) → `hesreallyhim/awesome-claude-code` (dạo chợ) → chọn 1–2 MCP server thật sự cần.
