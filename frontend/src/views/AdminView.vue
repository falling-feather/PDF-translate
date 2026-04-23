<script setup>
import { onMounted, ref } from "vue";
import { useRouter } from "vue-router";
import { authHeaders, clearSession, getUsername } from "../auth";

const router = useRouter();
const tab = ref("settings");

const settings = ref({});
const saving = ref(false);
const audit = ref([]);
const users = ref([]);
const jobs = ref([]);

const auditExpanded = ref({});

/** 用户可选翻译后端（与服务器 ALL_BACKENDS 一致） */
const ALL_BACKENDS = ["echo", "deepseek"];

const SF_BASE_PRESETS = [
  { id: "https://api.siliconflow.cn/v1", label: "国内 api.siliconflow.cn（推荐）" },
  { id: "https://api.siliconflow.com/v1", label: "国际 api.siliconflow.com" },
];

/** 硅基流动模型池（按角色拆分；DeepSeek 全部走 DeepSeek 官方 API，不在这里选） */
const SF_SURVEY_MODEL_PRESETS = [
  { id: "Qwen/Qwen2.5-VL-7B-Instruct", label: "Qwen2.5-VL-7B · 巡视/多模态（轻量、省成本）" },
  { id: "Qwen/Qwen2.5-VL-32B-Instruct", label: "Qwen2.5-VL-32B · 巡视/多模态（更强）" },
  { id: "Qwen/Qwen2.5-72B-Instruct-128K", label: "Qwen2.5-72B-128K · 巡视/长文本理解" },
  { id: "__custom__", label: "自定义（在下方填写完整模型 ID）" },
];

const SF_VISION_MODEL_PRESETS = [
  { id: "Qwen/Qwen2.5-VL-7B-Instruct", label: "Qwen2.5-VL-7B · 识图/图文理解（轻量）" },
  { id: "Qwen/Qwen2.5-VL-32B-Instruct", label: "Qwen2.5-VL-32B · 识图/图文理解（更强）" },
  { id: "__custom__", label: "自定义（在下方填写完整模型 ID）" },
];

const SF_PLANNER_MODEL_PRESETS = [
  { id: "Qwen/Qwen2.5-72B-Instruct-128K", label: "Qwen2.5-72B-128K · 规划/收束（长上下文）" },
  { id: "moonshotai/Kimi-K2-Instruct", label: "Kimi-K2-Instruct · 规划/收束（文本推理）" },
  { id: "THUDM/glm-4-9b-chat", label: "GLM-4-9B-Chat · 小模型试水" },
  { id: "__custom__", label: "自定义（在下方填写完整模型 ID）" },
];

const sfBaseChoice = ref("https://api.siliconflow.cn/v1");
const sfBaseCustom = ref("");

const sfSurveyPreset = ref("Qwen/Qwen2.5-VL-7B-Instruct");
const sfSurveyCustom = ref("");
const sfVisionPreset = ref("Qwen/Qwen2.5-VL-7B-Instruct");
const sfVisionCustom = ref("");
const sfPlannerPreset = ref("Qwen/Qwen2.5-72B-Instruct-128K");
const sfPlannerCustom = ref("");

const surveyEnabled = ref(false);
const plannerEnabled = ref(false);

function presetIds(kind) {
  const pool = kind === "survey" ? SF_SURVEY_MODEL_PRESETS : kind === "vision" ? SF_VISION_MODEL_PRESETS : SF_PLANNER_MODEL_PRESETS;
  return pool.filter((p) => p.id !== "__custom__").map((p) => p.id);
}

function matchPreset(stored, kind) {
  const s = (stored || "").trim();
  if (!s) return { preset: "__custom__", custom: "" };
  if (presetIds(kind).includes(s)) return { preset: s, custom: "" };
  return { preset: "__custom__", custom: s };
}

function resolveModel(preset, custom) {
  if (preset === "__custom__") return (custom || "").trim();
  return (preset || "").trim();
}

function syncSiliconflowUiFromSettings() {
  const url = (settings.value.siliconflow_base_url || "").trim();
  const hit = SF_BASE_PRESETS.find((p) => p.id === url);
  if (hit) {
    sfBaseChoice.value = hit.id;
    sfBaseCustom.value = "";
  } else if (url) {
    sfBaseChoice.value = "__custom__";
    sfBaseCustom.value = url;
  } else {
    sfBaseChoice.value = "https://api.siliconflow.cn/v1";
    sfBaseCustom.value = "";
  }

  const m1 = matchPreset(settings.value.siliconflow_survey_model, "survey");
  sfSurveyPreset.value = m1.preset;
  sfSurveyCustom.value = m1.custom;

  const m2 = matchPreset(settings.value.siliconflow_vision_model, "vision");
  sfVisionPreset.value = m2.preset;
  sfVisionCustom.value = m2.custom;

  const m3 = matchPreset(settings.value.planner_model, "planner");
  sfPlannerPreset.value = m3.preset;
  sfPlannerCustom.value = m3.custom;
}

function siliconflowBaseUrlForSave() {
  if (sfBaseChoice.value === "__custom__") return (sfBaseCustom.value || "").trim();
  return sfBaseChoice.value;
}

function formatAuditTime(iso) {
  if (!iso) return "—";
  const d = new Date(iso);
  if (!Number.isFinite(d.getTime())) return String(iso);
  return d.toLocaleString("zh-CN", {
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hour12: false,
  });
}

function toggleAuditDetail(id) {
  const cur = { ...auditExpanded.value };
  cur[id] = !cur[id];
  auditExpanded.value = cur;
}

async function loadSettings() {
  const r = await fetch("/api/admin/settings", { headers: authHeaders() });
  if (!r.ok) return;
  settings.value = await r.json();
  syncSiliconflowUiFromSettings();
  const se = settings.value.survey_enabled;
  surveyEnabled.value = se === true || se === "true";
  const pe = settings.value.planner_enabled;
  plannerEnabled.value = pe === true || pe === "true";
}

async function saveSettings() {
  saving.value = true;
  try {
    const body = {
      deepseek_api_key: settings.value.deepseek_api_key ?? "",
      deepseek_base_url: settings.value.deepseek_base_url ?? "",
      deepseek_model: settings.value.deepseek_model ?? "",
      default_backend: settings.value.default_backend ?? "deepseek",
      http_timeout_s: settings.value.http_timeout_s ?? "120",
      enabled_backends: settings.value.enabled_backends || ALL_BACKENDS,
      registration_open: !!settings.value.registration_open,
      survey_enabled: surveyEnabled.value,
      siliconflow_api_key: settings.value.siliconflow_api_key ?? "",
      siliconflow_base_url: siliconflowBaseUrlForSave(),
      siliconflow_survey_model: resolveModel(sfSurveyPreset.value, sfSurveyCustom.value),
      siliconflow_vision_model: resolveModel(sfVisionPreset.value, sfVisionCustom.value),
      survey_max_text_chars: settings.value.survey_max_text_chars ?? "12000",
      planner_enabled: plannerEnabled.value,
      planner_api_key: settings.value.planner_api_key ?? "",
      planner_base_url: settings.value.planner_base_url ?? "",
      planner_model: resolveModel(sfPlannerPreset.value, sfPlannerCustom.value),
    };
    const r = await fetch("/api/admin/settings", {
      method: "PUT",
      headers: authHeaders(true),
      body: JSON.stringify(body),
    });
    if (!r.ok) {
      alert("保存失败");
      return;
    }
    const d = await r.json();
    settings.value = d.settings || settings.value;
    syncSiliconflowUiFromSettings();
    const se = settings.value.survey_enabled;
    surveyEnabled.value = se === true || se === "true";
    const pe = settings.value.planner_enabled;
    plannerEnabled.value = pe === true || pe === "true";
    alert("已保存");
  } finally {
    saving.value = false;
  }
}

async function loadAudit() {
  const r = await fetch("/api/admin/audit?limit=300", { headers: authHeaders() });
  if (!r.ok) return;
  const d = await r.json();
  audit.value = d.events || [];
}

async function loadUsers() {
  const r = await fetch("/api/admin/users", { headers: authHeaders() });
  if (!r.ok) return;
  const d = await r.json();
  users.value = d.users || [];
}

async function loadJobs() {
  const r = await fetch("/api/admin/jobs?limit=500", { headers: authHeaders() });
  if (!r.ok) return;
  const d = await r.json();
  jobs.value = d.jobs || [];
}

function toggleBackend(b) {
  const cur = new Set(settings.value.enabled_backends || ALL_BACKENDS);
  if (cur.has(b)) cur.delete(b);
  else cur.add(b);
  settings.value.enabled_backends = Array.from(cur);
}

function logout() {
  clearSession();
  router.replace("/login");
}

async function adminDownload(jobId, kind, filename) {
  const r = await fetch(`/api/admin/jobs/${jobId}/artifact?kind=${kind}`, { headers: authHeaders() });
  if (!r.ok) {
    alert("下载失败");
    return;
  }
  const blob = await r.blob();
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = filename;
  a.click();
  URL.revokeObjectURL(a.href);
}

onMounted(() => {
  loadSettings();
  loadAudit();
  loadUsers();
  loadJobs();
});
</script>

<template>
  <div class="wrap">
    <header class="top">
      <div>
        <h1>管理后台</h1>
        <p class="muted small">{{ getUsername() }} · API 与审计</p>
      </div>
      <div class="tabs">
        <button :class="{ on: tab === 'settings' }" type="button" @click="tab = 'settings'">API 与策略</button>
        <button :class="{ on: tab === 'audit' }" type="button" @click="tab = 'audit'; loadAudit()">操作日志</button>
        <button :class="{ on: tab === 'users' }" type="button" @click="tab = 'users'; loadUsers()">用户</button>
        <button :class="{ on: tab === 'jobs' }" type="button" @click="tab = 'jobs'; loadJobs()">任务与文件</button>
        <button type="button" class="ghost" @click="logout">退出</button>
      </div>
    </header>

    <section v-show="tab === 'settings'" class="card">
      <h2>DeepSeek（主力文字翻译）</h2>
      <p class="muted small">
        使用 OpenAI 兼容接口。默认 Base <code>https://api.deepseek.com/v1</code>，模型如
        <code>deepseek-chat</code>。保存时<strong>留空的密钥框不会覆盖</strong>库里已有密钥。
      </p>

      <div class="cols">
        <label class="field"><span>DeepSeek API Key</span><input v-model="settings.deepseek_api_key" type="password" autocomplete="off" /></label>
        <label class="field"><span>DeepSeek Base URL</span><input v-model="settings.deepseek_base_url" placeholder="https://api.deepseek.com/v1" /></label>
        <label class="field"><span>DeepSeek Model</span><input v-model="settings.deepseek_model" placeholder="deepseek-chat" /></label>
      </div>
      <div class="cols">
        <label class="field"><span>默认后端</span><input v-model="settings.default_backend" placeholder="deepseek" /></label>
        <label class="field"><span>HTTP 超时(秒)</span><input v-model="settings.http_timeout_s" /></label>
      </div>

      <h3>允许用户选择的后端</h3>
      <div class="checks">
        <label v-for="b in ALL_BACKENDS" :key="b" class="ck">
          <input
            type="checkbox"
            :checked="(settings.enabled_backends || ALL_BACKENDS).includes(b)"
            @change="toggleBackend(b)"
          />
          {{ b }}
        </label>
      </div>

      <label class="ck block">
        <input type="checkbox" v-model="settings.registration_open" />
        允许自助注册
      </label>

      <div class="divider" />

      <h2>硅基流动（单 API 密钥 · 多模型）</h2>
      <p class="muted small">
        下方<strong>只需填写一个</strong>硅基流动 API Key；控制台 Base URL 已预置为国内/国际常用节点。巡视/识图/规划收束可在流程中分别调用不同模型（模型 ID 与硅基控制台一致）。
        <br />「译前巡视」开关（<code>survey_enabled</code>）对<strong>串/并联</strong>可选；用户端「精品翻译」会<strong>强制</strong>启用巡视并依赖此处配置。
        <br /><strong>注意：</strong>DeepSeek 模型请只在上方 DeepSeek 区域配置，不要填在硅基模型位。
      </p>

      <div class="cols">
        <label class="field"><span>硅基流动 API Key</span><input v-model="settings.siliconflow_api_key" type="password" autocomplete="off" placeholder="sk-…" /></label>
        <label class="field">
          <span>API Base URL</span>
          <select v-model="sfBaseChoice" class="select-wide">
            <option v-for="p in SF_BASE_PRESETS" :key="p.id" :value="p.id">{{ p.label }}</option>
            <option value="__custom__">自定义…</option>
          </select>
        </label>
        <label v-if="sfBaseChoice === '__custom__'" class="field">
          <span>自定义 Base URL</span><input v-model="sfBaseCustom" placeholder="https://…" />
        </label>
      </div>

      <label class="ck block">
        <input type="checkbox" v-model="surveyEnabled" />
        启用译前巡视（串/并联也可在管理员开启时使用；精品翻译不依赖此项）
      </label>

      <label class="field"><span>巡视块文本最大字符数</span><input v-model="settings.survey_max_text_chars" placeholder="12000" /></label>

      <h3 class="sf-h3">模型选择（下拉快速选 + 可自定义）</h3>
      <div class="sf-grid">
        <div class="sf-cell">
          <label class="field"><span>巡视模型（术语草拟、图文估计）</span></label>
          <select v-model="sfSurveyPreset">
            <option v-for="p in SF_SURVEY_MODEL_PRESETS" :key="'sv-' + p.id" :value="p.id">{{ p.label }}</option>
          </select>
          <input v-if="sfSurveyPreset === '__custom__'" v-model="sfSurveyCustom" class="custom-model" placeholder="完整模型 ID" />
        </div>
        <div class="sf-cell">
          <label class="field"><span>识图模型（预留，管线接入 VLM 后使用）</span></label>
          <select v-model="sfVisionPreset">
            <option v-for="p in SF_VISION_MODEL_PRESETS" :key="'vi-' + p.id" :value="p.id">{{ p.label }}</option>
          </select>
          <input v-if="sfVisionPreset === '__custom__'" v-model="sfVisionCustom" class="custom-model" placeholder="完整模型 ID" />
        </div>
        <div class="sf-cell">
          <label class="field"><span>规划收束模型（预留全文整理）</span></label>
          <select v-model="sfPlannerPreset">
            <option v-for="p in SF_PLANNER_MODEL_PRESETS" :key="'pl-' + p.id" :value="p.id">{{ p.label }}</option>
          </select>
          <input v-if="sfPlannerPreset === '__custom__'" v-model="sfPlannerCustom" class="custom-model" placeholder="完整模型 ID" />
        </div>
      </div>

      <label class="ck block muted">
        <input type="checkbox" v-model="plannerEnabled" />
        规划收束（预留，当前管线未接）
      </label>
      <div class="cols">
        <label class="field"><span>规划收束 API Key（可选，默认同硅基）</span><input v-model="settings.planner_api_key" type="password" autocomplete="off" /></label>
        <label class="field"><span>规划收束 Base URL（可选）</span><input v-model="settings.planner_base_url" placeholder="默认同硅基" /></label>
      </div>

      <button class="btn primary" type="button" :disabled="saving" @click="saveSettings">{{ saving ? "保存中…" : "保存设置" }}</button>
    </section>

    <section v-show="tab === 'audit'" class="card">
      <h2>审计日志（含登录 IP、任务与文件路径）</h2>
      <div class="scroll">
        <table class="table audit-table">
          <thead>
            <tr>
              <th>时间</th>
              <th>摘要</th>
              <th>用户</th>
              <th>IP</th>
              <th>任务</th>
              <th>原始动作</th>
              <th></th>
            </tr>
          </thead>
          <tbody>
            <template v-for="e in audit" :key="e.id">
              <tr>
                <td class="muted nowrap">{{ formatAuditTime(e.created_at) }}</td>
                <td class="summary-cell">{{ e.summary || e.action }}</td>
                <td>{{ e.username || "—" }}</td>
                <td>{{ e.ip || "—" }}</td>
                <td><code>{{ e.job_id || "—" }}</code></td>
                <td><code class="action-code">{{ e.action }}</code></td>
                <td class="nowrap">
                  <button type="button" class="linkish" @click="toggleAuditDetail(e.id)">
                    {{ auditExpanded[e.id] ? "收起" : "详情" }}
                  </button>
                </td>
              </tr>
              <tr v-if="auditExpanded[e.id]" class="audit-detail-row">
                <td colspan="7">
                  <pre class="audit-json">{{ JSON.stringify(e.detail, null, 2) }}</pre>
                </td>
              </tr>
            </template>
          </tbody>
        </table>
      </div>
    </section>

    <section v-show="tab === 'users'" class="card">
      <h2>用户列表</h2>
      <table class="table">
        <thead>
          <tr><th>ID</th><th>用户名</th><th>角色</th><th>注册时间</th></tr>
        </thead>
        <tbody>
          <tr v-for="u in users" :key="u.id">
            <td>{{ u.id }}</td>
            <td>{{ u.username }}</td>
            <td><code>{{ u.role }}</code></td>
            <td class="muted">{{ u.created_at }}</td>
          </tr>
        </tbody>
      </table>
    </section>

    <section v-show="tab === 'jobs'" class="card">
      <h2>全部任务与产物下载</h2>
      <div class="scroll">
        <table class="table">
          <thead>
            <tr>
              <th>任务 ID</th>
              <th>用户</th>
              <th>文件</th>
              <th>时间</th>
              <th>下载</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="j in jobs" :key="j.job_id">
              <td><code>{{ j.job_id }}</code></td>
              <td>{{ j.username }} ({{ j.user_id }})</td>
              <td>{{ j.original_filename }}</td>
              <td class="muted nowrap">{{ j.created_at }}</td>
              <td class="nowrap">
                <button type="button" class="linkish" @click="adminDownload(j.job_id, 'input', 'input.pdf')">PDF</button>
                <button type="button" class="linkish" @click="adminDownload(j.job_id, 'output_md', 'translated.md')">MD</button>
                <button type="button" class="linkish" @click="adminDownload(j.job_id, 'bundle_zip', j.job_id + '.zip')">ZIP</button>
              </td>
            </tr>
          </tbody>
        </table>
      </div>
    </section>
  </div>
</template>

<style scoped>
.wrap {
  max-width: 1100px;
  margin: 0 auto;
  padding: 1.25rem 1rem 3rem;
}
.top h1 {
  margin: 0 0 0.25rem;
  font-size: 1.45rem;
}
.tabs {
  display: flex;
  flex-wrap: wrap;
  gap: 0.35rem;
  margin-top: 0.75rem;
}
.tabs button {
  padding: 0.4rem 0.75rem;
  border-radius: 8px;
  border: 1px solid var(--border);
  background: #0d1117;
  color: var(--text);
  cursor: pointer;
  font-size: 0.88rem;
}
.tabs button.on {
  border-color: var(--accent);
  color: var(--accent);
}
.tabs button.ghost {
  margin-left: auto;
  background: transparent;
}
.card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 1.15rem 1.2rem;
  margin-top: 1rem;
}
h2 {
  margin: 0 0 0.5rem;
  font-size: 1.05rem;
}
h3 {
  margin: 1rem 0 0.5rem;
  font-size: 0.95rem;
}
.sf-h3 {
  margin-top: 1rem;
}
.divider {
  margin: 1.25rem 0;
  border-top: 1px solid var(--border);
}
.cols {
  display: grid;
  gap: 0.65rem;
  margin-bottom: 0.75rem;
}
@media (min-width: 720px) {
  .cols {
    grid-template-columns: repeat(3, 1fr);
  }
}
.field {
  display: flex;
  flex-direction: column;
  gap: 0.25rem;
}
.field span {
  font-size: 0.78rem;
  color: var(--muted);
}
.field input,
.field select {
  padding: 0.45rem 0.55rem;
  border-radius: 8px;
  border: 1px solid var(--border);
  background: #0d1117;
  color: var(--text);
}
.select-wide {
  width: 100%;
}
.sf-grid {
  display: grid;
  gap: 1rem;
  margin-bottom: 0.75rem;
}
@media (min-width: 900px) {
  .sf-grid {
    grid-template-columns: repeat(3, 1fr);
  }
}
.sf-cell select {
  width: 100%;
  padding: 0.45rem 0.55rem;
  border-radius: 8px;
  border: 1px solid var(--border);
  background: #0d1117;
  color: var(--text);
  margin-bottom: 0.35rem;
}
.custom-model {
  width: 100%;
  margin-top: 0.35rem;
}
.checks {
  display: flex;
  flex-wrap: wrap;
  gap: 0.65rem 1rem;
  margin: 0.5rem 0 1rem;
}
.ck {
  display: flex;
  align-items: center;
  gap: 0.35rem;
  font-size: 0.88rem;
  color: var(--muted);
}
.ck.block {
  margin-bottom: 1rem;
}
.btn.primary {
  padding: 0.55rem 1.2rem;
  border-radius: 8px;
  border: 1px solid #2563c9;
  background: var(--accent);
  color: #fff;
  font-weight: 600;
  cursor: pointer;
}
.scroll {
  max-height: 60vh;
  overflow: auto;
  margin-top: 0.5rem;
}
.table {
  width: 100%;
  border-collapse: collapse;
  font-size: 0.82rem;
}
.table th,
.table td {
  text-align: left;
  padding: 0.4rem 0.35rem;
  border-bottom: 1px solid var(--border);
  vertical-align: top;
}
.summary-cell {
  max-width: 280px;
  line-height: 1.35;
}
.action-code {
  font-size: 0.78rem;
  opacity: 0.85;
}
.audit-detail-row td {
  background: rgba(0, 0, 0, 0.2);
  padding: 0.5rem 0.35rem 0.65rem;
}
.audit-json {
  margin: 0;
  white-space: pre-wrap;
  word-break: break-all;
  font-size: 0.78rem;
  color: var(--muted);
  max-height: 240px;
  overflow: auto;
}
.nowrap {
  white-space: nowrap;
}
.linkish {
  background: none;
  border: none;
  color: var(--accent);
  cursor: pointer;
  text-decoration: underline;
  padding: 0 0.25rem;
  font-size: inherit;
}
.muted {
  color: var(--muted);
}
.small {
  font-size: 0.82rem;
}
</style>
