<script setup>
import { computed, onUnmounted, ref, onMounted } from "vue";
import { useRouter } from "vue-router";
import { authHeaders, clearSession, getUsername } from "../auth";

const router = useRouter();
const displayName = computed(() => getUsername());

const file = ref(null);
function onFileChange(e) {
  file.value = e.target.files?.[0] || null;
}
const tailFallback = ref(false);
const pagesPerChunk = ref(3);
const overlapPages = ref(1);
const backend = ref("");
const maxChunks = ref("");
/** @type {import('vue').Ref<'serial' | 'parallel'>} */
const translateMode = ref("serial");
const parallelMaxWorkers = ref(4);

const enabledBackends = ref([]);
const defaultBackend = ref("echo");

const jobId = ref("");
const job = ref(null);
const submitting = ref(false);
const pollTimer = ref(null);
const clockTimer = ref(null);
const startedAt = ref(null);
const nowTick = ref(0);
const myJobs = ref([]);

function fmtBytes(n) {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / (1024 * 1024)).toFixed(2)} MB`;
}

async function loadBackends() {
  const r = await fetch("/api/user/backends", { headers: authHeaders() });
  if (!r.ok) return;
  const d = await r.json();
  enabledBackends.value = d.enabled || [];
  defaultBackend.value = d.default_backend || "echo";
}

async function loadMyJobs() {
  const r = await fetch("/api/user/jobs", { headers: authHeaders() });
  if (!r.ok) return;
  const d = await r.json();
  myJobs.value = d.jobs || [];
}

onMounted(() => {
  loadBackends();
  loadMyJobs();
});

function logout() {
  clearSession();
  router.replace("/login");
}

async function submit() {
  if (!file.value) {
    alert("请选择 PDF 文件");
    return;
  }
  submitting.value = true;
  job.value = null;
  jobId.value = "";
  try {
    const fd = new FormData();
    fd.append("file", file.value);
    fd.append("tail_fallback", tailFallback.value ? "true" : "false");
    fd.append("pages_per_chunk", String(pagesPerChunk.value));
    fd.append("overlap_pages", String(overlapPages.value));
    const be = backend.value.trim();
    if (be) fd.append("backend", be);
    const mc = maxChunks.value.trim();
    if (mc) fd.append("max_chunks", mc);
    fd.append("translate_mode", translateMode.value);
    if (translateMode.value === "parallel") {
      fd.append("parallel_max_workers", String(Math.max(1, Math.min(32, Number(parallelMaxWorkers.value) || 4))));
    }

    const r = await fetch("/api/jobs", { method: "POST", headers: authHeaders(), body: fd });
    const data = await r.json().catch(() => ({}));
    if (!r.ok) {
      alert(typeof data.detail === "string" ? data.detail : JSON.stringify(data));
      return;
    }
    jobId.value = data.job_id;
    startedAt.value = Date.now();
    startPoll();
  } catch (e) {
    alert(String(e.message || e));
  } finally {
    submitting.value = false;
  }
}

async function pollOnce() {
  if (!jobId.value) return;
  const r = await fetch(`/api/jobs/${jobId.value}`, { headers: authHeaders() });
  if (!r.ok) return;
  job.value = await r.json();
  if (job.value.status === "done" || job.value.status === "error" || job.value.status === "cancelled") {
    stopPoll();
    loadMyJobs();
  }
}

async function cancelJob() {
  if (!jobId.value) return;
  try {
    const r = await fetch(`/api/jobs/${jobId.value}/cancel`, {
      method: "POST",
      headers: authHeaders(),
    });
    const data = await r.json().catch(() => ({}));
    if (!r.ok) {
      alert(typeof data.detail === "string" ? data.detail : JSON.stringify(data));
      return;
    }
    pollOnce();
  } catch (e) {
    alert(String(e.message || e));
  }
}

function startPoll() {
  stopPoll();
  nowTick.value = 0;
  clockTimer.value = setInterval(() => {
    nowTick.value++;
  }, 1000);
  pollTimer.value = setInterval(pollOnce, 700);
  pollOnce();
}

function stopPoll() {
  if (clockTimer.value) {
    clearInterval(clockTimer.value);
    clockTimer.value = null;
  }
  if (pollTimer.value) {
    clearInterval(pollTimer.value);
    pollTimer.value = null;
  }
}

onUnmounted(stopPoll);

const elapsedSec = computed(() => {
  void nowTick.value;
  if (!startedAt.value) return 0;
  return Math.floor((Date.now() - startedAt.value) / 1000);
});

const progressPct = computed(() => {
  const j = job.value;
  if (!j) return 0;
  if (j.status === "done") return 100;
  if (j.status === "cancelled") {
    const t = j.chunk_total || 1;
    const c = j.chunk_index || 0;
    return Math.min(99, Math.round((c / t) * 100));
  }
  if (!j.chunk_total) return j.phase === "split" ? 8 : j.phase === "init" ? 2 : 0;
  if (j.phase === "translate" || j.status === "running") {
    const cur = j.chunk_index || 0;
    const t = j.chunk_total || 1;
    let pct = Math.min(100, Math.round((cur / t) * 100));
    if (String(j.message || "").includes("正在请求模型") && pct < 8 && cur <= t) {
      pct = Math.max(pct, 5);
    }
    return pct;
  }
  return 0;
});

function filenameFromContentDisposition(cd) {
  if (!cd) return null;
  const u8 = cd.match(/filename\*=UTF-8''([^;]+)/i);
  if (u8) {
    try {
      return decodeURIComponent(u8[1].trim());
    } catch {
      return u8[1].trim();
    }
  }
  const q = cd.match(/filename="([^"]+)"/i);
  if (q) return q[1];
  return null;
}

async function downloadMd() {
  if (!jobId.value) return;
  const r = await fetch(`/api/jobs/${jobId.value}/download/full.md`, { headers: authHeaders() });
  if (!r.ok) {
    const t = await r.text();
    alert(t || "暂无可下载内容");
    return;
  }
  const blob = await r.blob();
  const fromHdr = filenameFromContentDisposition(r.headers.get("content-disposition"));
  const fallback = job.value?.suggested_download_filename || "translated.md";
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = fromHdr || fallback;
  a.click();
  URL.revokeObjectURL(a.href);
}

async function downloadZip() {
  if (!jobId.value) return;
  const r = await fetch(`/api/jobs/${jobId.value}/download/bundle.zip`, { headers: authHeaders() });
  if (!r.ok) return;
  const blob = await r.blob();
  const fromHdr = filenameFromContentDisposition(r.headers.get("content-disposition"));
  const fallback = job.value?.suggested_zip_filename || `${jobId.value}_bundle.zip`;
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = fromHdr || fallback;
  a.click();
  URL.revokeObjectURL(a.href);
}
</script>

<template>
  <div class="layout">
    <header class="top">
      <div>
        <h1>翻译工作台</h1>
        <p class="muted small">你好，{{ displayName }} · 仅可使用翻译与下载功能</p>
      </div>
      <div class="top-actions">
        <button type="button" class="btn ghost" @click="loadMyJobs">刷新我的任务</button>
        <button type="button" class="btn ghost" @click="logout">退出</button>
      </div>
    </header>

    <div class="grid">
      <section class="card">
        <h2>新建翻译</h2>
        <p class="muted small">可选后端：{{ enabledBackends.join(", ") }}；留空则使用服务器默认。</p>
        <label class="field">
          <span>PDF 文件</span>
          <input type="file" accept=".pdf,application/pdf" @change="onFileChange" />
        </label>
        <label class="check">
          <input type="checkbox" v-model="tailFallback" />
          未识别到 References 时，将最后约 15% 页作为参考文献
        </label>
        <div class="row">
          <label class="field sm">
            <span>每块页数</span>
            <select v-model.number="pagesPerChunk">
              <option :value="1">1</option>
              <option :value="2">2</option>
              <option :value="3">3</option>
            </select>
          </label>
          <label class="field sm">
            <span>重叠页</span>
            <input type="number" v-model.number="overlapPages" min="0" :max="Math.max(0, pagesPerChunk - 1)" />
          </label>
        </div>
        <label class="field">
          <span>翻译后端（可选，默认 {{ defaultBackend }}）</span>
          <input v-model="backend" placeholder="如 deepseek / openai / echo" />
        </label>
        <label class="field">
          <span>最大块数（调试用，留空全文）</span>
          <input v-model="maxChunks" />
        </label>
        <div class="mode-block">
          <span class="mode-label">翻译调度</span>
          <label class="radio-line">
            <input type="radio" value="serial" v-model="translateMode" />
            <span>串联式（推荐）</span>
          </label>
          <p class="muted small mode-desc">
            按顺序逐块调用模型，并携带前几块的摘要作为「记忆」，术语与风格更一致；耗时随页数线性增加。
          </p>
          <label class="radio-line">
            <input type="radio" value="parallel" v-model="translateMode" />
            <span>并联式（更快）</span>
          </label>
          <p class="muted small mode-desc">
            将待译块分批并行请求模型，再按原文顺序拼接；不跨块传递摘要，速度更高，但衔接与整体一致性可能略弱。
          </p>
          <label v-if="translateMode === 'parallel'" class="field sm parallel-workers">
            <span>每批并行数（1–32）</span>
            <input type="number" v-model.number="parallelMaxWorkers" min="1" max="32" />
          </label>
        </div>
        <button class="btn primary" :disabled="submitting" @click="submit">
          {{ submitting ? "提交中…" : "开始处理" }}
        </button>
      </section>

      <section class="card">
        <h2>当前任务</h2>
        <template v-if="!jobId">
          <p class="muted">提交后在此查看进度。</p>
        </template>
        <template v-else>
          <p>ID：<code>{{ jobId }}</code></p>
          <template v-if="job">
            <p>状态 <strong>{{ job.status }}</strong> · 阶段 <code>{{ job.phase }}</code></p>
            <p v-if="startedAt && (job.status === 'running' || job.phase === 'translate')" class="muted small">
              已运行 {{ elapsedSec }} 秒 · 网络 API 在整块返回前可能较久；失败会自动重试数次。
            </p>
            <p
              v-if="job.duration_seconds != null && (job.status === 'done' || job.status === 'cancelled' || job.status === 'error')"
              class="muted small"
            >
              总用时 <strong>{{ job.duration_seconds }}</strong> 秒
            </p>
            <p class="muted msg">{{ job.message }}</p>
            <p v-if="job.phase === 'translate' && job.chunk_total" class="muted small hint">
              长论文会按块依次调用 API；每一块都要等模型返回后进度才会增加，长时间停在 0% 或某一格通常仍属正常。
            </p>
            <div v-if="job.chunk_total" class="bar-wrap">
              <div class="bar" :style="{ width: progressPct + '%' }"></div>
            </div>
            <p v-if="job.error" class="err">{{ job.error }}</p>
            <div v-if="job.status === 'running' || job.phase === 'translate' || job.phase === 'split' || job.phase === 'init'" class="actions">
              <button type="button" class="btn warn" @click="cancelJob">终止翻译</button>
              <span class="muted small mid">已发送的请求仍会跑完当前块，随后停止。</span>
            </div>
            <div v-if="job.partial_output_ready || job.status === 'done' || job.status === 'cancelled'" class="actions">
              <button v-if="job.partial_output_ready" type="button" class="btn" @click="downloadMd">
                {{ job.status === "done" ? "下载完整译文 .md" : "下载已译部分 .md" }}
              </button>
              <span v-if="job.partial_output_ready && job.status !== 'done'" class="muted small mid">
                当前约 {{ fmtBytes(job.partial_output_bytes || 0) }}
              </span>
              <button v-if="job.status === 'done' || job.status === 'cancelled'" type="button" class="btn" @click="downloadZip">
                下载打包 .zip
              </button>
            </div>
          </template>
        </template>
      </section>
    </div>

    <section class="card full">
      <h2>我的任务</h2>
      <table v-if="myJobs.length" class="table">
        <thead>
          <tr>
            <th>任务 ID</th>
            <th>文件名</th>
            <th>创建时间</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="j in myJobs" :key="j.job_id">
            <td><code>{{ j.job_id }}</code></td>
            <td>{{ j.original_filename }}</td>
            <td class="muted">{{ j.created_at }}</td>
          </tr>
        </tbody>
      </table>
      <p v-else class="muted">暂无记录</p>
    </section>
  </div>
</template>

<style scoped>
.layout {
  max-width: 1000px;
  margin: 0 auto;
  padding: 1.5rem 1rem 3rem;
}
.top {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  gap: 1rem;
  margin-bottom: 1.25rem;
}
.top h1 {
  margin: 0;
  font-size: 1.5rem;
}
.top-actions {
  display: flex;
  gap: 0.5rem;
  flex-wrap: wrap;
}
.grid {
  display: grid;
  gap: 1rem;
}
@media (min-width: 880px) {
  .grid {
    grid-template-columns: 1fr 1fr;
  }
}
.card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 1.15rem 1.25rem;
}
.card.full {
  margin-top: 1rem;
}
h2 {
  margin: 0 0 0.75rem;
  font-size: 1.05rem;
}
.field {
  display: flex;
  flex-direction: column;
  gap: 0.35rem;
  margin-bottom: 0.85rem;
}
.field span {
  font-size: 0.82rem;
  color: var(--muted);
}
.field input,
.field select {
  padding: 0.5rem 0.6rem;
  border-radius: 8px;
  border: 1px solid var(--border);
  background: #0d1117;
  color: var(--text);
}
.row {
  display: flex;
  gap: 0.75rem;
  flex-wrap: wrap;
}
.field.sm {
  flex: 1;
  min-width: 120px;
}
.check {
  display: flex;
  gap: 0.45rem;
  align-items: flex-start;
  font-size: 0.88rem;
  color: var(--muted);
  margin-bottom: 0.85rem;
}
.btn {
  padding: 0.5rem 0.9rem;
  border-radius: 8px;
  border: 1px solid var(--border);
  background: #0d1117;
  color: var(--text);
  cursor: pointer;
}
.btn.primary {
  background: var(--accent);
  border-color: #2563c9;
  color: #fff;
  font-weight: 600;
  width: 100%;
  margin-top: 0.25rem;
}
.btn.ghost {
  background: transparent;
}
.btn.warn {
  border-color: #b45309;
  color: #fbbf24;
}
.mode-block {
  margin-bottom: 0.85rem;
  padding: 0.65rem 0.75rem;
  border-radius: 8px;
  border: 1px solid var(--border);
  background: rgba(0, 0, 0, 0.15);
}
.mode-label {
  display: block;
  font-size: 0.82rem;
  color: var(--muted);
  margin-bottom: 0.4rem;
}
.radio-line {
  display: flex;
  align-items: center;
  gap: 0.45rem;
  font-size: 0.9rem;
  margin-top: 0.35rem;
}
.mode-desc {
  margin: 0.2rem 0 0.35rem 1.35rem;
  line-height: 1.45;
}
.parallel-workers {
  margin: 0.5rem 0 0 1.35rem;
  max-width: 200px;
}
.bar-wrap {
  height: 8px;
  background: #0d1117;
  border-radius: 99px;
  overflow: hidden;
  margin: 0.5rem 0;
  border: 1px solid var(--border);
}
.bar {
  height: 100%;
  background: linear-gradient(90deg, var(--accent), #5eb0ff);
  transition: width 0.35s ease;
}
.actions {
  display: flex;
  flex-wrap: wrap;
  gap: 0.5rem;
  margin-top: 0.75rem;
  align-items: center;
}
.actions .mid {
  line-height: 1.4;
}
.err {
  color: var(--err);
  font-size: 0.88rem;
}
.msg {
  font-size: 0.88rem;
}
.hint {
  margin: 0.35rem 0 0.25rem;
  line-height: 1.45;
}
.table {
  width: 100%;
  border-collapse: collapse;
  font-size: 0.88rem;
}
.table th,
.table td {
  text-align: left;
  padding: 0.45rem 0.35rem;
  border-bottom: 1px solid var(--border);
}
</style>
