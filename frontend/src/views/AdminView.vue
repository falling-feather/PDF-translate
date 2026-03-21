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

const ALL_BACKENDS = ["echo", "openai", "deepseek", "ollama", "deepl", "hybrid"];

async function loadSettings() {
  const r = await fetch("/api/admin/settings", { headers: authHeaders() });
  if (!r.ok) return;
  settings.value = await r.json();
}

async function saveSettings() {
  saving.value = true;
  try {
    const body = {
      openai_api_key: settings.value.openai_api_key ?? "",
      openai_base_url: settings.value.openai_base_url ?? "",
      openai_model: settings.value.openai_model ?? "",
      ollama_base_url: settings.value.ollama_base_url ?? "",
      ollama_model: settings.value.ollama_model ?? "",
      deepl_api_key: settings.value.deepl_api_key ?? "",
      deepl_api_url: settings.value.deepl_api_url ?? "",
      deepseek_api_key: settings.value.deepseek_api_key ?? "",
      deepseek_base_url: settings.value.deepseek_base_url ?? "",
      deepseek_model: settings.value.deepseek_model ?? "",
      default_backend: settings.value.default_backend ?? "echo",
      http_timeout_s: settings.value.http_timeout_s ?? "120",
      enabled_backends: settings.value.enabled_backends || ALL_BACKENDS,
      registration_open: !!settings.value.registration_open,
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
      <h2>API 密钥与模型（写入 SQLite，优先于环境变量）</h2>
      <p class="muted small">
        DeepSeek 使用 OpenAI 兼容接口：默认 Base <code>https://api.deepseek.com/v1</code>，模型如
        <code>deepseek-chat</code>。保存时<strong>留空的密钥框不会覆盖</strong>库里已有密钥；仅填写要新增或修改的项即可。
      </p>

      <div class="cols">
        <label class="field"><span>OpenAI API Key</span><input v-model="settings.openai_api_key" type="password" autocomplete="off" /></label>
        <label class="field"><span>OpenAI Base URL</span><input v-model="settings.openai_base_url" /></label>
        <label class="field"><span>OpenAI Model</span><input v-model="settings.openai_model" /></label>
      </div>
      <div class="cols">
        <label class="field"><span>DeepSeek API Key</span><input v-model="settings.deepseek_api_key" type="password" autocomplete="off" /></label>
        <label class="field"><span>DeepSeek Base URL</span><input v-model="settings.deepseek_base_url" placeholder="https://api.deepseek.com/v1" /></label>
        <label class="field"><span>DeepSeek Model</span><input v-model="settings.deepseek_model" placeholder="deepseek-chat" /></label>
      </div>
      <div class="cols">
        <label class="field"><span>DeepL API Key</span><input v-model="settings.deepl_api_key" type="password" autocomplete="off" /></label>
        <label class="field"><span>DeepL API URL</span><input v-model="settings.deepl_api_url" /></label>
      </div>
      <div class="cols">
        <label class="field"><span>Ollama Base</span><input v-model="settings.ollama_base_url" /></label>
        <label class="field"><span>Ollama Model</span><input v-model="settings.ollama_model" /></label>
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

      <button class="btn primary" type="button" :disabled="saving" @click="saveSettings">{{ saving ? "保存中…" : "保存设置" }}</button>
    </section>

    <section v-show="tab === 'audit'" class="card">
      <h2>审计日志（含登录 IP、任务与文件路径）</h2>
      <div class="scroll">
        <table class="table">
          <thead>
            <tr>
              <th>时间</th>
              <th>动作</th>
              <th>用户</th>
              <th>IP</th>
              <th>任务</th>
              <th>详情</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="e in audit" :key="e.id">
              <td class="muted nowrap">{{ e.created_at }}</td>
              <td><code>{{ e.action }}</code></td>
              <td>{{ e.username || "—" }}</td>
              <td>{{ e.ip || "—" }}</td>
              <td><code>{{ e.job_id || "—" }}</code></td>
              <td class="detail"><pre>{{ JSON.stringify(e.detail, null, 0) }}</pre></td>
            </tr>
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
.field input {
  padding: 0.45rem 0.55rem;
  border-radius: 8px;
  border: 1px solid var(--border);
  background: #0d1117;
  color: var(--text);
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
.detail pre {
  margin: 0;
  white-space: pre-wrap;
  word-break: break-all;
  max-width: 320px;
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
</style>
