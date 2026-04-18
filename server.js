import "dotenv/config";
import express from "express";
import path from "node:path";
import { fileURLToPath } from "node:url";
import OpenAI from "openai";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const PORT = Number(process.env.PORT) || 3000;

const app = express();
app.use(express.json({ limit: "12mb" }));
app.use(express.static(__dirname));

// ─────────────────────────── System prompt ───────────────────────────
const ROLE_PROMPT = `你是一位资深求职顾问，专注帮助中国大学生完成求职申请。
- 语气专业但亲切，像学长/学姐，不打官腔
- 回答简洁有重点，优先结构化输出（适度使用 Markdown：加粗、有序/无序列表、分隔线）
- 基于用户的真实申请数据给个性化建议，不要重复用户已提供的数据
- 你可以参考用户的每个岗位详情（含 JD、备注、材料、面试记录）来做更具体的建议
- 如果用户上传了简历文件、文档或图片，请优先结合附件内容分析
- 首轮回复控制在 200 字以内，抓重点；后续追问再展开
- 问题超出求职范围时，礼貌引导回主题`;

const STAGE_ORDER = ["意向中", "已投递", "笔试/测评", "面试中", "Offer", "已结束"];
const MATERIAL_LABELS = {
  resume: "简历",
  coverLetter: "Cover Letter",
  transcript: "成绩单",
  portfolio: "作品集",
};

function compactText(text, maxLen = 240) {
  if (!text || typeof text !== "string") return "";
  const cleaned = text.replace(/\s+/g, " ").trim();
  if (!cleaned) return "";
  return cleaned.length > maxLen ? `${cleaned.slice(0, maxLen)}...` : cleaned;
}

function buildSnapshotBlock(snap) {
  if (!snap) return "";
  const d = snap.distribution || {};
  const distStr = STAGE_ORDER.map(k => `${k} ${d[k] || 0}`).join(" / ");
  const lines = ["【用户当前申请概况】"];
  lines.push(`总申请数：${snap.total ?? 0} 家`);
  lines.push(`阶段分布：${distStr}`);
  if (snap.urgentDeadlines?.length) {
    const urgent = snap.urgentDeadlines.map(u => {
      const left = u.daysLeft < 0 ? `已过 ${-u.daysLeft} 天`
        : u.daysLeft === 0 ? "今天"
        : `距今 ${u.daysLeft} 天`;
      return `${u.company}（${u.kind}截止 · ${left}）`;
    }).join("、");
    lines.push(`即将到期（7 天内）：${snap.urgentDeadlines.length} 个 — ${urgent}`);
  } else {
    lines.push(`即将到期（7 天内）：0 个`);
  }
  if (snap.activeInterviews?.length) lines.push(`当前面试中：${snap.activeInterviews.join("、")}`);
  if (snap.offers?.length) lines.push(`当前 Offer：${snap.offers.join("、")}`);
  return lines.join("\n");
}

function buildApplicationsBlock(applications) {
  if (!Array.isArray(applications) || applications.length === 0) return "";

  const maxApps = 20;
  const picked = applications.slice(0, maxApps);
  const lines = ["【用户申请详情】"];

  picked.forEach((app, index) => {
    const header = `${index + 1}. ${app.company || "未填写公司"}｜${app.position || "未填写岗位"}｜${app.type || "未填写类型"}｜${app.stage || "未填写阶段"}`;
    lines.push(header);

    if (app.appliedAt) lines.push(`投递日期：${app.appliedAt}`);
    if (app.closeReason) lines.push(`结束原因：${app.closeReason}`);

    if (Array.isArray(app.deadlines) && app.deadlines.length) {
      const deadlineText = app.deadlines
        .filter(d => d?.kind || d?.date)
        .map(d => `${d.kind || "未命名"} ${d.date || "未填日期"}`)
        .join("；");
      if (deadlineText) lines.push(`Deadline：${deadlineText}`);
    }

    const materials = Object.entries(app.materials || {})
      .filter(([, checked]) => !!checked)
      .map(([key]) => MATERIAL_LABELS[key] || key);
    lines.push(`已备材料：${materials.length ? materials.join("、") : "暂无"}`);

    if (Array.isArray(app.interviews) && app.interviews.length) {
      const interviewText = app.interviews.slice(0, 4).map(it => {
        const note = compactText(it.note, 60);
        return `第${it.round || "?"}轮 ${it.date || "未排期"} ${it.format || ""}${note ? `（${note}）` : ""}`;
      }).join("；");
      lines.push(`面试记录：${interviewText}`);
    } else {
      lines.push("面试记录：暂无");
    }

    if (app.note) lines.push(`备注：${compactText(app.note, 120)}`);
    if (app.jd) lines.push(`JD 摘要：${compactText(app.jd, 320)}`);
  });

  if (applications.length > maxApps) {
    lines.push(`其余 ${applications.length - maxApps} 个岗位未展开；如用户追问，可优先基于已展示岗位回答。`);
  }

  return lines.join("\n");
}

// ─────────────────────────── Health ───────────────────────────
function isConfigured() {
  return !!process.env.OPENAI_API_KEY && !!process.env.OPENAI_MODEL;
}

app.get("/api/health", (_req, res) => {
  res.json({ configured: isConfigured() });
});

// ─────────────────────────── Chat (SSE) ───────────────────────────
app.post("/api/chat", async (req, res) => {
  if (!isConfigured()) {
    return res.status(503).json({
      code: "NO_KEY",
      message: "AI 助手未配置，请在 .env 中填写 OPENAI_API_KEY 与 OPENAI_MODEL",
    });
  }
  const { messages = [], snapshot, applications = [] } = req.body || {};
  if (!Array.isArray(messages) || messages.length === 0) {
    return res.status(400).json({ code: "BAD_REQUEST", message: "messages 不能为空" });
  }

  const client = new OpenAI({
    apiKey: process.env.OPENAI_API_KEY,
    baseURL: process.env.OPENAI_BASE_URL || undefined,
  });

  const systemContent = [
    ROLE_PROMPT,
    buildSnapshotBlock(snapshot),
    buildApplicationsBlock(applications),
  ].filter(Boolean).join("\n\n");

  res.setHeader("Content-Type", "text/event-stream; charset=utf-8");
  res.setHeader("Cache-Control", "no-cache, no-transform");
  res.setHeader("Connection", "keep-alive");
  res.setHeader("X-Accel-Buffering", "no");
  res.flushHeaders?.();

  let clientClosed = false;
  const markClientClosed = (reason) => {
    if (clientClosed) return;
    clientClosed = true;
    console.log(`[chat] client disconnected: ${reason}`);
  };
  req.on("aborted", () => markClientClosed("request aborted"));
  res.on("close", () => {
    if (!res.writableEnded) markClientClosed("response closed");
  });
  const heartbeatId = setInterval(() => {
    if (clientClosed || res.writableEnded) return;
    res.write(": ping\n\n");
  }, 10000);

  try {
    const modelName = process.env.OPENAI_MODEL;
    const isReasoning = /^(gpt-5|o1|o3|o4)/i.test(modelName);

    const params = {
      model: modelName,
      stream: true,
      max_completion_tokens: isReasoning ? 8192 : 1024,
      messages: [
        { role: "system", content: systemContent },
        ...messages.map(m => ({ role: m.role, content: m.content })),
      ],
    };
    if (isReasoning) params.reasoning_effort = "low";

    console.log(`[chat] model=${modelName} reasoning=${isReasoning} msgs=${messages.length} maxTok=${params.max_completion_tokens}`);
    const stream = await client.chat.completions.create(params);

    let deltaCount = 0;
    let chunkCount = 0;
    let finishReason = null;
    for await (const chunk of stream) {
      if (clientClosed) break;
      chunkCount++;
      if (chunkCount <= 3) console.log(`[chat chunk #${chunkCount}]`, JSON.stringify(chunk));
      const delta = chunk.choices?.[0]?.delta?.content;
      if (delta) { deltaCount++; res.write(`data: ${JSON.stringify({ delta })}\n\n`); }
      if (chunk.choices?.[0]?.finish_reason) finishReason = chunk.choices[0].finish_reason;
    }
    console.log(`[chat] done chunks=${chunkCount} deltas=${deltaCount} finish=${finishReason} clientClosed=${clientClosed}`);

    // 流式 0 chunk 时，尝试非流式兜底以暴露真实错误或内容
    if (chunkCount === 0) {
      if (clientClosed) {
        console.log("[chat] ⚠ 流式 0 chunk，但客户端已断开，跳过兜底");
      } else {
        console.log("[chat] ⚠ 流式 0 chunk，进入非流式兜底…");
        try {
          const nonStreamParams = { ...params, stream: false };
          const result = await client.chat.completions.create(nonStreamParams);
          console.log("[chat non-stream] result:", JSON.stringify(result).slice(0, 1200));
          const content = result.choices?.[0]?.message?.content;
          const nsFinish = result.choices?.[0]?.finish_reason;
          if (content) {
            res.write(`data: ${JSON.stringify({ delta: content })}\n\n`);
          } else {
            res.write(`data: ${JSON.stringify({ error: { code: "EMPTY", message: `非流式也返回空 content（finish=${nsFinish}）。检查模型名 "${modelName}" / 账号权限 / 网络。` } })}\n\n`);
          }
        } catch (fbErr) {
          console.error("[chat non-stream error]", fbErr?.status, fbErr?.message, fbErr?.code);
          res.write(`data: ${JSON.stringify({ error: { code: "FALLBACK_FAIL", message: `非流式兜底失败：${fbErr?.status || ""} ${fbErr?.code || ""} ${fbErr?.message || "unknown"}` } })}\n\n`);
        }
      }
    } else if (deltaCount === 0 && !clientClosed) {
      res.write(`data: ${JSON.stringify({ error: { code: "EMPTY", message: `模型返回为空（chunks=${chunkCount} finish=${finishReason}）。` } })}\n\n`);
    }
    if (!clientClosed) res.write("data: [DONE]\n\n");
    res.end();
  } catch (err) {
    const status = err?.status || err?.response?.status;
    let payload = { code: "UPSTREAM", message: "AI 助手暂时不可用" };
    let httpCode = 502;
    if (status === 401) { payload = { code: "INVALID_KEY", message: "API Key 无效，请检查 .env" }; httpCode = 401; }
    else if (status === 429) { payload = { code: "RATE_LIMIT", message: "请求太频繁，请等几秒再试" }; httpCode = 429; }
    else if (status >= 400 && status < 500) { payload = { code: "BAD_UPSTREAM", message: `上游错误（${status}）` }; httpCode = status; }
    console.error("[chat error]", status, err?.message);

    if (res.headersSent) {
      if (!clientClosed) res.write(`data: ${JSON.stringify({ error: payload })}\n\n`);
      res.end();
    } else {
      res.status(httpCode).json(payload);
    }
  } finally {
    clearInterval(heartbeatId);
  }
});

// ─────────────────────────── Listen ───────────────────────────
app.listen(PORT, () => {
  console.log("");
  console.log("  ▸ 求职申请管理看板 + AI 求职助手");
  console.log(`  ▸ http://localhost:${PORT}`);
  console.log(`  ▸ OpenAI 配置：${isConfigured() ? "已就绪" : "未就绪 — 请在 .env 中填写 OPENAI_API_KEY 与 OPENAI_MODEL"}`);
  if (process.env.OPENAI_BASE_URL) console.log(`  ▸ Base URL：${process.env.OPENAI_BASE_URL}`);
  console.log("");
});
