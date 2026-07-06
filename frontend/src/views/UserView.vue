<script setup>
import { computed, onMounted, onUnmounted, ref, watch } from "vue";
import { useRouter } from "vue-router";
import { authHeaders, clearSession, getUsername } from "../auth";
import GlossaryReviewModal from "../components/GlossaryReviewModal.vue";
import RepairPatchReviewModal from "../components/RepairPatchReviewModal.vue";
import TableMergedCellReviewModal from "../components/TableMergedCellReviewModal.vue";

const router = useRouter();
const displayName = computed(() => getUsername());

const selectedFiles = ref([]);
const dragActive = ref(false);
const submitting = ref(false);

const tailFallback = ref(false);
const pagesPerChunk = ref(3);
const overlapPages = ref(1);
const backend = ref("");
const backendLabels = ref({});
const maxChunks = ref("");
const translateMode = ref("serial");
const parallelMaxWorkers = ref(4);

const useCustomApi = ref(false);
const customApiBackend = ref("deepseek");
const customApiKey = ref("");
const customApiBaseUrl = ref("");
const customApiModel = ref("");

const enabledBackends = ref([]);
const defaultBackend = ref("deepseek");
const backendCatalog = ref([]);
const customApiBackends = ref(["deepseek"]);

const myJobs = ref([]);
const myJobsError = ref("");
const favoriteJobs = ref([]);
const favoriteMax = ref(3);

const taskOrder = ref([]);
const taskMap = ref({});
const pollTimer = ref(null);

const showSupportModal = ref(false);
const glossaryReviewJobId = ref("");
const repairPatchReviewJobId = ref("");
const tableReviewJobId = ref("");
const pageNow = ref(Date.now());
let footerTimer = null;

const FOOTER_START_AT = new Date("2026-03-22T00:00:00+08:00").getTime();
const MAX_PARALLEL_TASKS = 3;
const CACHE_KEY = "pdf_translate_user_view_state_v4";

function formatDisplayTime(iso) {
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

function isActiveJob(j) {
  if (!j) return false;
  return j.status === "queued" || j.status === "running" || ["init", "split", "translate"].includes(j.phase);
}

const activeTaskCount = computed(() => taskOrder.value.filter((id) => isActiveJob(taskMap.value[id])).length);
const remainSlots = computed(() => Math.max(0, MAX_PARALLEL_TASKS - activeTaskCount.value));
const statusHint = computed(() => `当前正在执行 ${activeTaskCount.value} 个文件的翻译，还可以上传至多 ${remainSlots.value} 个`);
const displayTaskIds = computed(() => taskOrder.value.slice(0, MAX_PARALLEL_TASKS));

const uptimeText = computed(() => {
  const sec = Math.max(0, Math.floor((pageNow.value - FOOTER_START_AT) / 1000));
  const d = Math.floor(sec / 86400);
  const h = Math.floor((sec % 86400) / 3600);
  const m = Math.floor((sec % 3600) / 60);
  const s = sec % 60;
  return `${d}天${h}小时${m}分${s}秒`;
});

function onFileChange(e) {
  selectedFiles.value = Array.from(e.target.files || []).filter((f) => f.name.toLowerCase().endsWith(".pdf"));
}

function onRootDragOver(e) {
  if (e.dataTransfer?.types?.includes("Files")) dragActive.value = true;
}

function onRootDragLeave() {
  dragActive.value = false;
}

function onDropzoneDrop(e) {
  dragActive.value = false;
  e.stopPropagation();
  const files = Array.from(e.dataTransfer?.files || []).filter((f) => f.name.toLowerCase().endsWith(".pdf"));
  if (!files.length) return;
  selectedFiles.value = files;
}

function labelForBackend(b) {
  const m = backendLabels.value;
  if (m && m[b]) return m[b];
  const hit = backendCatalog.value.find((item) => item.id === b);
  return hit?.label || b;
}

function backendSpec(b) {
  return backendCatalog.value.find((item) => item.id === b) || {};
}

function normalizeErrorPayload(payload) {
  const root = payload && typeof payload === "object" && !Array.isArray(payload) ? payload : {};
  const rawDetail = root.detail !== undefined ? root.detail : payload;
  if (typeof rawDetail === "string") return { ...root, message: root.message || rawDetail };
  if (rawDetail && typeof rawDetail === "object" && !Array.isArray(rawDetail)) return { ...root, ...rawDetail };
  return root;
}

function formatErrorPayload(payload) {
  const detail = normalizeErrorPayload(payload);
  if (!detail || !Object.keys(detail).length) return "请求失败，请稍后重试";
  const message =
    detail.user_message ||
    detail.message ||
    (typeof detail.detail === "string" ? detail.detail : "") ||
    detail.error ||
    "请求失败";
  const nextStep = detail.next_step || detail.error_next_step;
  const codeValue = detail.code || detail.error_code;
  const next = nextStep ? `建议：${nextStep}` : "";
  const code = codeValue ? `错误码：${codeValue}` : "";
  return [message, next, code].filter(Boolean).join(" · ");
}

function jobNextStep(j) {
  return j?.error_next_step || j?.next_step || "";
}

function jobErrorMeta(j) {
  if (!j) return "";
  const parts = [];
  if (j.error_code || j.code) parts.push(j.error_code || j.code);
  if (j.error_category || j.category) parts.push(j.error_category || j.category);
  if (j.error_http_status || j.http_status) parts.push(`HTTP ${j.error_http_status || j.http_status}`);
  return parts.join(" · ");
}

function showJobIssue(j) {
  if (!j) return false;
  return j.status === "error" || (j.status === "cancelled" && (j.error || j.error_code || j.code || jobNextStep(j)));
}

function jobIssueTitle(j) {
  return j?.status === "cancelled" ? "任务已终止" : "任务失败";
}

function artifactSummary(j) {
  if (!j) return "";
  const ready = [];
  if (j.input_pdf_ready) ready.push("原文 PDF");
  if (j.partial_output_ready) ready.push("译文 MD");
  if (j.translated_pdf_ready) ready.push("译文 PDF");
  if (j.bilingual_html_ready) ready.push("双语 HTML");
  if (j.glossary_review_ready) ready.push("术语确认");
  if (j.glossary_retranslation_plan_ready) ready.push("术语重译计划");
  if (j.glossary_retranslation_result_ready) ready.push("术语重译报告");
  if (j.glossary_retranslated_full_ready) ready.push("术语候选译文");
  if (j.table_merged_cell_review_ready) ready.push("表格确认");
  if (j.table_structure_publish_ready) ready.push("表格发布");
  if (j.repair_patch_review_ready) ready.push("补丁审核");
  if (j.repair_effectiveness_report_ready) ready.push("修复效果");
  if (j.repair_publish_report_ready) ready.push("修复报告");
  if (j.repair_published_full_ready) ready.push("修复发布稿");
  if (j.repair_rollback_report_ready) ready.push("回滚报告");
  if (j.repair_rollback_full_ready) ready.push("回滚演练稿");
  if (j.repair_formal_replace_report_ready) ready.push("正式替换报告");
  if (j.repair_formal_full_ready) ready.push("正式译文");
  if (j.repair_formal_rollback_report_ready) ready.push("正式回滚报告");
  if (j.repair_formal_backup_full_ready) ready.push("正式备份");
  if (j.bundle_zip_ready) ready.push("资料包 ZIP");
  return ready.length ? `可下载：${ready.join(" / ")}` : "";
}

async function loadBackends() {
  const r = await fetch("/api/user/backends", { headers: authHeaders() });
  if (!r.ok) return;
  const d = await r.json();
  enabledBackends.value = d.enabled || [];
  defaultBackend.value = d.default_backend || "deepseek";
  backendLabels.value = d.labels || {};
  backendCatalog.value = Array.isArray(d.catalog) ? d.catalog : [];
  customApiBackends.value = Array.isArray(d.custom_api_backends) && d.custom_api_backends.length ? d.custom_api_backends : ["deepseek"];
  if (backend.value && !enabledBackends.value.includes(backend.value)) backend.value = "";
  if (!customApiBackends.value.includes(customApiBackend.value)) customApiBackend.value = customApiBackends.value[0] || "deepseek";
}

async function loadMyJobs() {
  myJobsError.value = "";
  const r = await fetch("/api/user/jobs", { headers: authHeaders() });
  if (!r.ok) {
    myJobsError.value = "任务列表加载失败（若持续出现请联系管理员检查服务日志）";
    favoriteJobs.value = [];
    return;
  }
  const d = await r.json();
  myJobs.value = d.jobs || [];
  favoriteJobs.value = Array.isArray(d.favorites) ? d.favorites : [];
  favoriteMax.value = Number(d.favorite_max) || 3;
}

async function runWorkbenchCleanup() {
  try {
    const r = await fetch("/api/user/jobs/cleanup-stale", {
      method: "POST",
      headers: authHeaders(),
    });
    if (!r.ok && typeof console !== "undefined" && console.warn) {
      console.warn("cleanup-stale 未成功:", r.status);
    }
    const d = r.ok ? await r.json().catch(() => ({})) : {};
    const n = (d.deleted && d.deleted.length) || 0;
    if (n > 0) {
      const gone = new Set(d.deleted || []);
      taskOrder.value = taskOrder.value.filter((id) => !gone.has(id));
      const next = { ...taskMap.value };
      for (const id of gone) delete next[id];
      taskMap.value = next;
    }
  } catch {
    // ignore cleanup errors; lists still load
  }
}

async function favoriteJobRow(jobId) {
  const r = await fetch(`/api/user/favorites/${encodeURIComponent(jobId)}`, {
    method: "POST",
    headers: authHeaders(),
  });
  const data = await r.json().catch(() => ({}));
  if (!r.ok) {
    alert(formatErrorPayload(data));
    return;
  }
  await loadMyJobs();
}

async function unfavoriteJobRow(jobId) {
  const r = await fetch(`/api/user/favorites/${encodeURIComponent(jobId)}`, {
    method: "DELETE",
    headers: authHeaders(),
  });
  const data = await r.json().catch(() => ({}));
  if (!r.ok) {
    alert(formatErrorPayload(data));
    return;
  }
  await loadMyJobs();
}

async function refreshTaskLists() {
  stopPolling();
  selectedFiles.value = [];
  taskOrder.value = [];
  taskMap.value = {};
  await runWorkbenchCleanup();
  await loadMyJobs();
}

function ensureTaskTracked(jobId) {
  if (!taskOrder.value.includes(jobId)) taskOrder.value.unshift(jobId);
  taskOrder.value = taskOrder.value.slice(0, 12);
}

async function pollJob(jobId) {
  const r = await fetch(`/api/jobs/${jobId}`, { headers: authHeaders() });
  if (!r.ok) return;
  const d = await r.json();
  taskMap.value = { ...taskMap.value, [jobId]: d };
}

async function pollAllJobs() {
  const ids = taskOrder.value.slice();
  for (const id of ids) {
    try {
      await pollJob(id);
    } catch {
      // ignore single job polling failures
    }
  }
  if (ids.length) loadMyJobs();
}

function startPolling() {
  if (pollTimer.value) return;
  pollTimer.value = setInterval(pollAllJobs, 1000);
}

function stopPolling() {
  if (!pollTimer.value) return;
  clearInterval(pollTimer.value);
  pollTimer.value = null;
}

/** 从并行窗口移除卡片（不删服务端任务，仍可「我的任务」里加入） */
function removeFromWorkbench(jobId) {
  taskOrder.value = taskOrder.value.filter((id) => id !== jobId);
  const next = { ...taskMap.value };
  delete next[jobId];
  taskMap.value = next;
  const stillActive = taskOrder.value.some((id) => isActiveJob(taskMap.value[id]));
  if (!stillActive) stopPolling();
}

async function createJobForFile(file) {
  const fd = new FormData();
  fd.append("file", file);
  fd.append("tail_fallback", tailFallback.value ? "true" : "false");
  fd.append("pages_per_chunk", String(pagesPerChunk.value));
  fd.append("overlap_pages", String(overlapPages.value));
  const be = backend.value.trim();
  if (be) fd.append("backend", be);

  if (useCustomApi.value) {
    const cbe = customApiBackend.value.trim().toLowerCase();
    if (!customApiBackends.value.includes(cbe)) throw new Error(`API翻译仅支持 ${customApiBackends.value.join(" / ")}`);
    if (backendSpec(cbe).requires_api_key && !customApiKey.value.trim()) {
      throw new Error("当前 API 翻译后端需要你填写 API Key");
    }
    fd.append("use_custom_api", "true");
    fd.append("custom_backend", cbe);
    if (customApiKey.value.trim()) fd.append("custom_api_key", customApiKey.value.trim());
    if (customApiBaseUrl.value.trim()) fd.append("custom_api_base_url", customApiBaseUrl.value.trim());
    if (customApiModel.value.trim()) fd.append("custom_api_model", customApiModel.value.trim());
  }

  const mc = maxChunks.value.trim();
  if (mc) fd.append("max_chunks", mc);
  fd.append("translate_mode", translateMode.value);
  if (translateMode.value === "parallel") {
    fd.append("parallel_max_workers", String(Math.max(1, Math.min(32, Number(parallelMaxWorkers.value) || 4))));
  }

  const r = await fetch("/api/jobs", { method: "POST", headers: authHeaders(), body: fd });
  const data = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(formatErrorPayload(data));
  const jid = data.job_id;
  ensureTaskTracked(jid);
  await pollJob(jid);
}

async function submitSelectedFiles(fromDrop = false) {
  if (!selectedFiles.value.length) {
    if (!fromDrop) alert("请选择 PDF 文件");
    return;
  }
  if (remainSlots.value <= 0) {
    alert("当前并行任务已满（最多 3 个），请稍后再上传");
    return;
  }

  submitting.value = true;
  try {
    const canTake = remainSlots.value;
    const batch = selectedFiles.value.slice(0, canTake);
    const skipped = selectedFiles.value.length - batch.length;
    for (const f of batch) {
      await createJobForFile(f);
    }
    if (skipped > 0) alert(`并行槽位不足，已有 ${skipped} 个文件未提交。`);
    selectedFiles.value = [];
    await loadMyJobs();
    startPolling();
  } catch (e) {
    alert(String(e.message || e));
  } finally {
    submitting.value = false;
  }
}

async function openJobFromList(jid) {
  ensureTaskTracked(jid);
  await pollJob(jid);
  startPolling();
}

async function cancelJob(jid) {
  const r = await fetch(`/api/jobs/${jid}/cancel`, { method: "POST", headers: authHeaders() });
  const data = await r.json().catch(() => ({}));
  if (!r.ok) {
    alert(formatErrorPayload(data));
    return;
  }
  await pollJob(jid);
}

async function reviewRepairPatch(jid) {
  repairPatchReviewJobId.value = jid;
}

async function reviewGlossary(jid) {
  glossaryReviewJobId.value = jid;
}

async function reviewTableMergedCell(jid) {
  tableReviewJobId.value = jid;
}

async function onGlossaryReviewUpdated(job) {
  if (job?.job_id) {
    taskMap.value = { ...taskMap.value, [job.job_id]: job };
  }
  await loadMyJobs();
}

async function onTableReviewUpdated(job) {
  if (job?.job_id) {
    taskMap.value = { ...taskMap.value, [job.job_id]: job };
  }
  await loadMyJobs();
}

async function onRepairPatchReviewUpdated(job) {
  if (job?.job_id) {
    taskMap.value = { ...taskMap.value, [job.job_id]: job };
  }
  await loadMyJobs();
}

async function confirmTableStructurePublish(jid) {
  const job = taskMap.value[jid] || {};
  const blockingReviews = Number(job.table_merged_cell_review_required_count || 0);
  if (blockingReviews > 0) {
    alert(`仍有 ${blockingReviews} 个表格合并候选未完成确认，请先处理表格审核。`);
    return;
  }
  if (!window.confirm("确认生成表格结构副本？原始表格重建证据仍会保留。")) return;
  const r = await fetch(`/api/jobs/${jid}/table-structure-publish/confirm`, {
    method: "POST",
    headers: authHeaders(),
  });
  const data = await r.json().catch(() => ({}));
  if (!r.ok) {
    alert(formatErrorPayload(data));
    return;
  }
  taskMap.value = { ...taskMap.value, [jid]: data };
  await loadMyJobs();
  alert("已生成表格结构副本");
}

async function executeGlossaryRetranslation(jid) {
  const job = taskMap.value[jid] || {};
  const count = Number(job.glossary_retranslation_plan_retranslate_chunk_count || 0);
  if (count <= 0) {
    alert("当前没有需要重译的术语分块。");
    return;
  }
  if (!window.confirm(`确认执行 ${count} 个术语影响分块的候选重译？原始译文仍会保留。`)) return;
  const r = await fetch(`/api/jobs/${jid}/glossary-retranslation/execute`, {
    method: "POST",
    headers: authHeaders(true),
    body: JSON.stringify({ mode: "stale_only" }),
  });
  const data = await r.json().catch(() => ({}));
  if (!r.ok) {
    alert(formatErrorPayload(data));
    return;
  }
  taskMap.value = { ...taskMap.value, [jid]: data };
  await loadMyJobs();
  alert("已生成术语候选重译稿");
}

async function confirmRepairPublish(jid) {
  const job = taskMap.value[jid] || {};
  const blockingReviews = Number(job.repair_patch_review_blocking_count || 0);
  if (blockingReviews > 0) {
    alert(`仍有 ${blockingReviews} 个补丁未通过审核，请先处理补丁审核。`);
    return;
  }
  const openIssues = Number(job.repair_publish_open_issue_count || 0);
  const confirmText = openIssues > 0
    ? `当前仍有 ${openIssues} 个开放合并问题。确认后会生成修复发布稿，但原始译文仍会保留。是否继续？`
    : "确认生成修复发布稿？原始译文仍会保留。";
  if (!window.confirm(confirmText)) return;
  const r = await fetch(`/api/jobs/${jid}/repair-publish/confirm`, {
    method: "POST",
    headers: authHeaders(),
  });
  const data = await r.json().catch(() => ({}));
  if (!r.ok) {
    alert(formatErrorPayload(data));
    return;
  }
  taskMap.value = { ...taskMap.value, [jid]: data };
  await loadMyJobs();
  alert("已生成修复发布稿");
}

async function confirmRepairRollback(jid) {
  const job = taskMap.value[jid] || {};
  if (!job.repair_published_full_ready) {
    alert("修复发布稿尚未生成，暂不能执行回滚演练。");
    return;
  }
  if (!window.confirm("确认生成回滚演练稿？这会复制原始译文为 rollback_full.md，不会覆盖修复发布稿。")) return;
  const r = await fetch(`/api/jobs/${jid}/repair-rollback/confirm`, {
    method: "POST",
    headers: authHeaders(),
  });
  const data = await r.json().catch(() => ({}));
  if (!r.ok) {
    alert(formatErrorPayload(data));
    return;
  }
  taskMap.value = { ...taskMap.value, [jid]: data };
  await loadMyJobs();
  alert("已生成回滚演练稿");
}

async function confirmRepairFormalReplace(jid) {
  const job = taskMap.value[jid] || {};
  if (!job.repair_published_full_ready) {
    alert("修复发布稿尚未生成，暂不能生成正式译文。");
    return;
  }
  if (!window.confirm("确认把修复发布稿提升为正式译文？系统会保留正式译文修复前备份。")) return;
  const r = await fetch(`/api/jobs/${jid}/repair-formal-replace/confirm`, {
    method: "POST",
    headers: authHeaders(),
  });
  const data = await r.json().catch(() => ({}));
  if (!r.ok) {
    alert(formatErrorPayload(data));
    return;
  }
  taskMap.value = { ...taskMap.value, [jid]: data };
  await loadMyJobs();
  alert("已生成正式修复译文");
}

async function confirmRepairFormalRollback(jid) {
  const job = taskMap.value[jid] || {};
  if (!job.repair_formal_backup_full_ready) {
    alert("正式译文修复前备份尚未生成，暂不能正式回滚。");
    return;
  }
  if (!window.confirm("确认把正式译文恢复到修复前备份？当前修复正式稿会另存为回滚前副本。")) return;
  const r = await fetch(`/api/jobs/${jid}/repair-formal-rollback/confirm`, {
    method: "POST",
    headers: authHeaders(),
  });
  const data = await r.json().catch(() => ({}));
  if (!r.ok) {
    alert(formatErrorPayload(data));
    return;
  }
  taskMap.value = { ...taskMap.value, [jid]: data };
  await loadMyJobs();
  alert("已回滚正式译文");
}

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

async function downloadFrom(url, fallbackName) {
  const r = await fetch(url, { headers: authHeaders() });
  if (!r.ok) {
    const t = await r.text();
    if (!t) {
      alert("暂无可下载内容");
      return;
    }
    let payload = t;
    try {
      payload = JSON.parse(t);
    } catch {
      // Keep plain-text server responses readable.
    }
    alert(formatErrorPayload(payload));
    return;
  }
  const blob = await r.blob();
  const fromHdr = filenameFromContentDisposition(r.headers.get("content-disposition"));
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = fromHdr || fallbackName;
  a.click();
  URL.revokeObjectURL(a.href);
}

function progressForJob(j) {
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
    return Math.min(100, Math.round((cur / t) * 100));
  }
  return 0;
}

function elapsedText(j) {
  if (!j) return "";
  if (j.duration_seconds != null) return `${j.duration_seconds} 秒`;
  if (!j.run_started_at) return "";
  const start = Date.parse(j.run_started_at);
  if (!Number.isFinite(start)) return "";
  return `${Math.max(0, Math.floor((Date.now() - start) / 1000))} 秒`;
}

function openSupportModal() {
  showSupportModal.value = true;
}

function closeSupportModal() {
  showSupportModal.value = false;
}

function logout() {
  clearSession();
  router.replace("/login");
}

onMounted(async () => {
  stopPolling();
  selectedFiles.value = [];
  taskOrder.value = [];
  taskMap.value = {};

  try {
    const raw = localStorage.getItem(CACHE_KEY);
    if (raw) {
      const s = JSON.parse(raw);
      tailFallback.value = !!s.tailFallback;
      pagesPerChunk.value = Number(s.pagesPerChunk) || 3;
      overlapPages.value = Number(s.overlapPages) || 1;
      backend.value = String(s.backend || "");
      maxChunks.value = String(s.maxChunks || "");
      const tm = s.translateMode;
      translateMode.value = tm === "parallel" || tm === "premium" ? tm : "serial";
      parallelMaxWorkers.value = Number(s.parallelMaxWorkers) || 4;
      useCustomApi.value = !!s.useCustomApi;
      customApiBackend.value = String(s.customApiBackend || "deepseek");
      customApiBaseUrl.value = String(s.customApiBaseUrl || "");
      customApiModel.value = String(s.customApiModel || "");
    }
  } catch {
    // ignore invalid cache
  }

  await loadBackends();
  await runWorkbenchCleanup();
  await loadMyJobs();
  footerTimer = setInterval(() => {
    pageNow.value = Date.now();
  }, 1000);
});

watch(
  [
    tailFallback,
    pagesPerChunk,
    overlapPages,
    backend,
    maxChunks,
    translateMode,
    parallelMaxWorkers,
    useCustomApi,
    customApiBackend,
    customApiBaseUrl,
    customApiModel,
  ],
  () => {
    const payload = {
      tailFallback: tailFallback.value,
      pagesPerChunk: pagesPerChunk.value,
      overlapPages: overlapPages.value,
      backend: backend.value,
      maxChunks: maxChunks.value,
      translateMode: translateMode.value,
      parallelMaxWorkers: parallelMaxWorkers.value,
      useCustomApi: useCustomApi.value,
      customApiBackend: customApiBackend.value,
      customApiBaseUrl: customApiBaseUrl.value,
      customApiModel: customApiModel.value,
    };
    localStorage.setItem(CACHE_KEY, JSON.stringify(payload));
  },
  { deep: false },
);

onUnmounted(() => {
  stopPolling();
  if (footerTimer) {
    clearInterval(footerTimer);
    footerTimer = null;
  }
});
</script>

<template>
  <div
    class="layout"
    :class="{ 'drag-active': dragActive }"
    @dragover.prevent="onRootDragOver"
    @dragleave="onRootDragLeave"
  >
    <header class="top">
      <div>
        <h1>翻译工作台</h1>
        <p class="muted small">你好，{{ displayName }} · 仅可使用翻译与下载功能</p>
      </div>
      <div class="top-actions">
        <button type="button" class="btn ghost" @click="refreshTaskLists">刷新我的任务</button>
        <button type="button" class="btn ghost" @click="logout">退出</button>
      </div>
    </header>

    <div class="grid">
      <section class="card">
        <h2>新建翻译</h2>
        <p class="muted small">
          翻译后端由管理员启用；请选择列表中的选项。默认使用服务器配置：{{ defaultBackend }}（{{ labelForBackend(defaultBackend) }}）。
        </p>

        <label class="field">
          <span>PDF 文件（支持多选）</span>
          <input type="file" accept=".pdf,application/pdf" multiple @change="onFileChange" />
        </label>

        <div class="dropzone" @dragover.prevent @drop.stop.prevent="onDropzoneDrop">
          <p>将 PDF 拖到这里即可选中文件</p>
          <p class="muted small">拖入后不会自动上传；请确认选项后点击「上传并开始翻译」。并行上限为 3 个任务。</p>
        </div>

        <p v-if="selectedFiles.length" class="muted small" style="margin-top: 0.45rem">
          已选择 {{ selectedFiles.length }} 个文件：
          {{ selectedFiles.map((f) => f.name).join("、") }}
        </p>

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
          <span>翻译后端</span>
          <select v-model="backend" class="backend-select">
            <option value="">使用服务器默认（{{ defaultBackend }}）</option>
            <option v-for="b in enabledBackends" :key="b" :value="b">{{ labelForBackend(b) }}</option>
          </select>
        </label>

        <div class="mode-block">
          <label class="radio-line">
            <input type="checkbox" v-model="useCustomApi" />
            <span>API翻译（使用我自己的 API，而非管理员配置）</span>
          </label>
          <div v-if="useCustomApi" class="custom-api-panel">
            <label class="field">
              <span>API 后端</span>
              <select v-model="customApiBackend">
                <option v-for="b in customApiBackends" :key="b" :value="b">{{ labelForBackend(b) }}</option>
              </select>
            </label>
            <label class="field">
              <span>API Key（所选后端需要密钥时必填）</span>
              <input v-model="customApiKey" type="password" placeholder="输入你自己的 API Key" />
            </label>
            <label class="field">
              <span>接口地址（可选）</span>
              <input v-model="customApiBaseUrl" placeholder="例如 https://api.deepseek.com/v1" />
            </label>
            <label class="field">
              <span>模型名（可选）</span>
              <input v-model="customApiModel" placeholder="例如 deepseek-chat / gpt-4o-mini" />
            </label>
          </div>
        </div>

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
          <label class="radio-line">
            <input type="radio" value="parallel" v-model="translateMode" />
            <span>并联式（更快）</span>
          </label>
          <label class="radio-line">
            <input type="radio" value="premium" v-model="translateMode" />
            <span>精品翻译（译前巡视 + 术语草拟，需管理员配置硅基流动）</span>
          </label>
          <p v-if="translateMode === 'premium'" class="muted small mode-hint">
            固定为串联流程，并启用硅基流动巡视；与「并联」互斥。若提示未配置硅基流动，请联系管理员。
          </p>
          <label v-if="translateMode === 'parallel'" class="field sm parallel-workers">
            <span>每批并行数（1–32）</span>
            <input type="number" v-model.number="parallelMaxWorkers" min="1" max="32" />
          </label>
        </div>

        <button class="btn primary" :disabled="submitting" @click="submitSelectedFiles(false)">
          {{ submitting ? "提交中…" : "上传并开始翻译" }}
        </button>
      </section>

      <section class="card">
        <h2>当前任务（并行窗口）</h2>
        <p class="muted small">{{ statusHint }}</p>

        <div v-if="displayTaskIds.length" class="task-list">
          <article v-for="tid in displayTaskIds" :key="tid" class="task-item">
            <template v-if="taskMap[tid]">
              <div class="task-head">
                <p><strong>{{ taskMap[tid].original_filename || '未命名文件' }}</strong></p>
                <p class="muted small">ID：<code>{{ tid }}</code></p>
              </div>

              <p class="muted small">
                状态 <strong>{{ taskMap[tid].status }}</strong> · 阶段 <code>{{ taskMap[tid].phase }}</code>
              </p>
              <p v-if="artifactSummary(taskMap[tid])" class="muted small">{{ artifactSummary(taskMap[tid]) }}</p>
              <p class="muted small" v-if="elapsedText(taskMap[tid])">已运行/总用时 {{ elapsedText(taskMap[tid]) }}</p>
              <p class="muted msg">{{ taskMap[tid].message }}</p>

              <div v-if="showJobIssue(taskMap[tid])" class="job-issue" :class="{ cancelled: taskMap[tid].status === 'cancelled' }">
                <p class="job-issue-title">{{ jobIssueTitle(taskMap[tid]) }}</p>
                <p v-if="taskMap[tid].error" class="job-issue-text">{{ taskMap[tid].error }}</p>
                <p v-if="jobNextStep(taskMap[tid])" class="job-issue-text">建议：{{ jobNextStep(taskMap[tid]) }}</p>
                <p v-if="jobErrorMeta(taskMap[tid])" class="job-issue-meta">{{ jobErrorMeta(taskMap[tid]) }}</p>
              </div>

              <div class="bar-wrap" v-if="taskMap[tid].chunk_total || isActiveJob(taskMap[tid])">
                <div class="bar" :style="{ width: progressForJob(taskMap[tid]) + '%' }"></div>
              </div>

              <div class="actions">
                <button type="button" class="btn linkish" @click="downloadFrom(`/api/jobs/${tid}/download/input.pdf`, taskMap[tid].original_filename || 'input.pdf')">
                  原文件
                </button>
                <button type="button" class="btn" @click="downloadFrom(`/api/jobs/${tid}/download/full.md`, taskMap[tid].suggested_download_filename || 'translated.md')">
                  译文 .md
                </button>
                <button
                  v-if="taskMap[tid].status === 'done' || taskMap[tid].status === 'cancelled'"
                  type="button"
                  class="btn"
                  :disabled="!taskMap[tid].translated_pdf_ready"
                  @click="downloadFrom(`/api/jobs/${tid}/download/translated.pdf`, `${tid}_translated.pdf`)"
                >
                  译文 PDF
                </button>
                <button
                  v-if="taskMap[tid].glossary_review_ready"
                  type="button"
                  class="btn linkish"
                  :disabled="Number(taskMap[tid].glossary_review_reviewable_count || taskMap[tid].glossary_review_pending_count || 0) <= 0"
                  @click="reviewGlossary(tid)"
                >
                  确认术语
                </button>
                <button
                  v-if="taskMap[tid].glossary_retranslation_plan_ready"
                  type="button"
                  class="btn linkish"
                  :disabled="!taskMap[tid].glossary_retranslation_plan_ready"
                  @click="downloadFrom(`/api/jobs/${tid}/download/glossary-retranslation-plan.md`, `${tid}_glossary_retranslation_plan.md`)"
                >
                  重译计划
                </button>
                <button
                  v-if="taskMap[tid].glossary_retranslation_plan_ready"
                  type="button"
                  class="btn linkish"
                  :disabled="!taskMap[tid].glossary_retranslation_plan_ready"
                  @click="downloadFrom(`/api/jobs/${tid}/download/glossary-retranslation-plan.json`, `${tid}_glossary_retranslation_plan.json`)"
                >
                  计划 JSON
                </button>
                <button
                  v-if="taskMap[tid].status === 'done' && taskMap[tid].glossary_retranslation_plan_ready"
                  type="button"
                  class="btn"
                  :disabled="Number(taskMap[tid].glossary_retranslation_plan_retranslate_chunk_count || 0) <= 0"
                  @click="executeGlossaryRetranslation(tid)"
                >
                  执行重译
                </button>
                <button
                  v-if="taskMap[tid].glossary_retranslation_result_ready"
                  type="button"
                  class="btn linkish"
                  :disabled="!taskMap[tid].glossary_retranslation_result_ready"
                  @click="downloadFrom(`/api/jobs/${tid}/download/glossary-retranslation-result.md`, `${tid}_glossary_retranslation_result.md`)"
                >
                  重译报告
                </button>
                <button
                  v-if="taskMap[tid].glossary_retranslated_full_ready"
                  type="button"
                  class="btn linkish"
                  :disabled="!taskMap[tid].glossary_retranslated_full_ready"
                  @click="downloadFrom(`/api/jobs/${tid}/download/glossary-retranslated-full.md`, `${tid}_glossary_retranslated_full.md`)"
                >
                  候选译文
                </button>
                <button
                  v-if="taskMap[tid].status === 'done' || taskMap[tid].status === 'cancelled'"
                  type="button"
                  class="btn linkish"
                  :disabled="!taskMap[tid].repair_publish_report_ready"
                  @click="downloadFrom(`/api/jobs/${tid}/download/repair-publish.md`, `${tid}_repair_publish.md`)"
                >
                  修复报告
                </button>
                <button
                  v-if="taskMap[tid].status === 'done' || taskMap[tid].status === 'cancelled'"
                  type="button"
                  class="btn linkish"
                  :disabled="!taskMap[tid].repair_effectiveness_report_ready"
                  @click="downloadFrom(`/api/jobs/${tid}/download/repair-effectiveness.md`, `${tid}_repair_effectiveness.md`)"
                >
                  修复效果
                </button>
                <button
                  v-if="taskMap[tid].status === 'done' || taskMap[tid].status === 'cancelled'"
                  type="button"
                  class="btn linkish"
                  :disabled="!taskMap[tid].repair_rollback_report_ready"
                  @click="downloadFrom(`/api/jobs/${tid}/download/repair-rollback.md`, `${tid}_repair_rollback.md`)"
                >
                  回滚报告
                </button>
                <button
                  v-if="taskMap[tid].status === 'done' || taskMap[tid].status === 'cancelled'"
                  type="button"
                  class="btn linkish"
                  :disabled="!taskMap[tid].repair_patch_review_ready"
                  @click="downloadFrom(`/api/jobs/${tid}/download/repair-patch-review.md`, `${tid}_repair_patch_review.md`)"
                >
                  补丁审核
                </button>
                <button
                  v-if="taskMap[tid].status === 'done' || taskMap[tid].status === 'cancelled'"
                  type="button"
                  class="btn linkish"
                  :disabled="!taskMap[tid].table_merged_cell_review_ready"
                  @click="downloadFrom(`/api/jobs/${tid}/download/table-merged-cell-review.md`, `${tid}_table_merged_cell_review.md`)"
                >
                  表格确认
                </button>
                <button
                  v-if="taskMap[tid].status === 'done'"
                  type="button"
                  class="btn linkish"
                  :disabled="!taskMap[tid].table_merged_cell_review_ready"
                  @click="reviewTableMergedCell(tid)"
                >
                  审核表格
                </button>
                <button
                  v-if="taskMap[tid].status === 'done'"
                  type="button"
                  class="btn"
                  :disabled="!taskMap[tid].table_merged_cell_review_ready || taskMap[tid].table_reconstruction_confirmed_ready || Number(taskMap[tid].table_merged_cell_review_required_count || 0) > 0"
                  @click="confirmTableStructurePublish(tid)"
                >
                  确认表格结构
                </button>
                <button
                  v-if="taskMap[tid].status === 'done' || taskMap[tid].status === 'cancelled'"
                  type="button"
                  class="btn linkish"
                  :disabled="!taskMap[tid].table_structure_publish_ready"
                  @click="downloadFrom(`/api/jobs/${tid}/download/table-structure-publish.md`, `${tid}_table_structure_publish.md`)"
                >
                  表格发布
                </button>
                <button
                  v-if="taskMap[tid].status === 'done' || taskMap[tid].status === 'cancelled'"
                  type="button"
                  class="btn linkish"
                  :disabled="!taskMap[tid].table_reconstruction_confirmed_ready"
                  @click="downloadFrom(`/api/jobs/${tid}/download/table-reconstruction-confirmed.json`, `${tid}_table_reconstruction_confirmed.json`)"
                >
                  表格副本
                </button>
                <button
                  v-if="taskMap[tid].status === 'done'"
                  type="button"
                  class="btn linkish"
                  :disabled="!taskMap[tid].repair_patch_review_ready"
                  @click="reviewRepairPatch(tid)"
                >
                  审核补丁
                </button>
                <button
                  v-if="taskMap[tid].status === 'done'"
                  type="button"
                  class="btn"
                  :disabled="!taskMap[tid].repair_publish_report_ready || taskMap[tid].repair_published_full_ready || Number(taskMap[tid].repair_patch_review_blocking_count || 0) > 0"
                  @click="confirmRepairPublish(tid)"
                >
                  确认修复稿
                </button>
                <button
                  v-if="taskMap[tid].status === 'done' || taskMap[tid].status === 'cancelled'"
                  type="button"
                  class="btn linkish"
                  :disabled="!taskMap[tid].repair_published_full_ready"
                  @click="downloadFrom(`/api/jobs/${tid}/download/published-full.md`, `${tid}_published_full.md`)"
                >
                  修复发布稿
                </button>
                <button
                  v-if="taskMap[tid].status === 'done'"
                  type="button"
                  class="btn"
                  :disabled="!taskMap[tid].repair_published_full_ready || taskMap[tid].repair_rollback_full_ready"
                  @click="confirmRepairRollback(tid)"
                >
                  回滚演练
                </button>
                <button
                  v-if="taskMap[tid].status === 'done' || taskMap[tid].status === 'cancelled'"
                  type="button"
                  class="btn linkish"
                  :disabled="!taskMap[tid].repair_rollback_full_ready"
                  @click="downloadFrom(`/api/jobs/${tid}/download/rollback-full.md`, `${tid}_rollback_full.md`)"
                >
                  回滚演练稿
                </button>
                <button
                  v-if="taskMap[tid].status === 'done' || taskMap[tid].status === 'cancelled'"
                  type="button"
                  class="btn linkish"
                  :disabled="!taskMap[tid].repair_formal_replace_report_ready"
                  @click="downloadFrom(`/api/jobs/${tid}/download/repair-formal-replace.md`, `${tid}_repair_formal_replace.md`)"
                >
                  正式替换报告
                </button>
                <button
                  v-if="taskMap[tid].status === 'done'"
                  type="button"
                  class="btn"
                  :disabled="!taskMap[tid].repair_published_full_ready || taskMap[tid].repair_formal_full_ready"
                  @click="confirmRepairFormalReplace(tid)"
                >
                  生成正式译文
                </button>
                <button
                  v-if="taskMap[tid].status === 'done' || taskMap[tid].status === 'cancelled'"
                  type="button"
                  class="btn linkish"
                  :disabled="!taskMap[tid].repair_formal_full_ready"
                  @click="downloadFrom(`/api/jobs/${tid}/download/formal-full.md`, `${tid}_formal_full.md`)"
                >
                  正式译文
                </button>
                <button
                  v-if="taskMap[tid].status === 'done'"
                  type="button"
                  class="btn"
                  :disabled="!taskMap[tid].repair_formal_backup_full_ready || taskMap[tid].repair_formal_rollback_applied"
                  @click="confirmRepairFormalRollback(tid)"
                >
                  正式回滚
                </button>
                <button
                  v-if="taskMap[tid].status === 'done' || taskMap[tid].status === 'cancelled'"
                  type="button"
                  class="btn linkish"
                  :disabled="!taskMap[tid].repair_formal_rollback_report_ready"
                  @click="downloadFrom(`/api/jobs/${tid}/download/repair-formal-rollback.md`, `${tid}_repair_formal_rollback.md`)"
                >
                  正式回滚报告
                </button>
                <button
                  v-if="taskMap[tid].status === 'done' || taskMap[tid].status === 'cancelled'"
                  type="button"
                  class="btn linkish"
                  :disabled="!taskMap[tid].repair_formal_backup_full_ready"
                  @click="downloadFrom(`/api/jobs/${tid}/download/formal-before-repair-full.md`, `${tid}_formal_before_repair.md`)"
                >
                  修复前备份
                </button>
                <button
                  v-if="taskMap[tid].status === 'done' || taskMap[tid].status === 'cancelled'"
                  type="button"
                  class="btn"
                  @click="downloadFrom(`/api/jobs/${tid}/download/bundle.zip`, taskMap[tid].suggested_zip_filename || `${tid}_bundle.zip`)"
                >
                  打包 .zip
                </button>
                <button v-if="isActiveJob(taskMap[tid])" type="button" class="btn warn" @click="cancelJob(tid)">终止</button>
                <button
                  v-if="taskMap[tid].status === 'done' || taskMap[tid].status === 'cancelled'"
                  type="button"
                  class="btn linkish"
                  @click="removeFromWorkbench(tid)"
                >
                  移出工作台
                </button>
              </div>
            </template>
            <template v-else>
              <p class="muted">任务 {{ tid }} 加载中…</p>
            </template>
          </article>
        </div>

        <p v-else class="muted">暂无并行任务。上传文件后将在此显示（最多 3 个并行流程）。</p>
      </section>
    </div>

    <section class="card full">
      <h2>我的任务</h2>
      <p v-if="myJobsError" class="err">{{ myJobsError }}</p>
      <table v-else-if="myJobs.length" class="table">
        <thead>
          <tr>
            <th>任务 ID</th>
            <th>文件名</th>
            <th>创建时间</th>
            <th>操作</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="j in myJobs" :key="j.job_id">
            <td><code>{{ j.job_id }}</code></td>
            <td>{{ j.original_filename }}</td>
            <td class="muted nowrap">{{ formatDisplayTime(j.created_at) }}</td>
            <td class="nowrap">
              <button type="button" class="btn linkish" @click="openJobFromList(j.job_id)">加入并行窗口</button>
              <button
                type="button"
                class="btn linkish"
                :disabled="favoriteJobs.length >= favoriteMax"
                @click="favoriteJobRow(j.job_id)"
              >
                收藏
              </button>
            </td>
          </tr>
        </tbody>
      </table>
      <p v-else class="muted">暂无记录。</p>

      <p class="muted small" style="margin-top: 0.75rem">
        每账号最多 {{ favoriteMax }} 条收藏；在任务行点击「收藏」后，条目会出现在下方列表。
      </p>

      <template v-if="favoriteJobs.length">
        <h2 class="sub-heading">收藏的任务</h2>
        <p class="muted small">取消收藏后任务会回到「我的任务」，创建时间按取消收藏时刻更新。</p>
        <table class="table">
          <thead>
            <tr>
              <th>任务 ID</th>
              <th>文件名</th>
              <th>列表时间</th>
              <th>操作</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="j in favoriteJobs" :key="'fav-' + j.job_id">
              <td><code>{{ j.job_id }}</code></td>
              <td>{{ j.original_filename }}</td>
              <td class="muted nowrap">{{ formatDisplayTime(j.created_at) }}</td>
              <td class="nowrap">
                <button type="button" class="btn linkish" @click="openJobFromList(j.job_id)">加入并行窗口</button>
                <button type="button" class="btn linkish" @click="unfavoriteJobRow(j.job_id)">取消收藏</button>
              </td>
            </tr>
          </tbody>
        </table>
      </template>
    </section>

    <footer class="site-footer">
      <p class="brand-line">
        made with
        <a href="#" @click.prevent="openSupportModal">落入白川的羽</a>
        <a class="heart-link" href="https://github.com/falling-feather/PDF-translate" target="_blank" rel="noopener noreferrer">♡</a>
      </p>
      <p class="muted">本站已在随时准备跑路的状态下以极其不稳定的方式运行了 {{ uptimeText }}</p>
    </footer>

    <div v-if="showSupportModal" class="support-mask" @click.self="closeSupportModal">
      <div class="support-modal">
        <button type="button" class="close-btn" @click="closeSupportModal">×</button>
        <h3>感谢支持</h3>
        <div class="support-images">
          <figure>
            <img src="/sucai/add-friend.webp" alt="加好友" />
            <figcaption>加好友</figcaption>
          </figure>
          <figure>
            <img src="/sucai/payment.webp" alt="收款码" />
            <figcaption>收款码</figcaption>
          </figure>
        </div>
      </div>
    </div>
    <TableMergedCellReviewModal
      v-if="tableReviewJobId"
      :job-id="tableReviewJobId"
      @updated="onTableReviewUpdated"
      @close="tableReviewJobId = ''"
    />
    <GlossaryReviewModal
      v-if="glossaryReviewJobId"
      :job-id="glossaryReviewJobId"
      @updated="onGlossaryReviewUpdated"
      @close="glossaryReviewJobId = ''"
    />
    <RepairPatchReviewModal
      v-if="repairPatchReviewJobId"
      :job-id="repairPatchReviewJobId"
      @updated="onRepairPatchReviewUpdated"
      @close="repairPatchReviewJobId = ''"
    />
  </div>
</template>

<style scoped>
.layout {
  max-width: 1120px;
  margin: 0 auto;
  padding: 1.5rem 1rem 3rem;
}
.layout.drag-active {
  outline: 2px dashed #6da9ff;
  outline-offset: 6px;
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
@media (min-width: 980px) {
  .grid {
    grid-template-columns: 1.08fr 1fr;
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
.sub-heading {
  margin: 1.35rem 0 0.5rem;
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
.dropzone {
  border: 1px dashed #5f84b6;
  border-radius: 10px;
  min-height: 110px;
  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: center;
  text-align: center;
  background: rgba(99, 155, 255, 0.06);
  margin-bottom: 0.85rem;
  padding: 0.7rem;
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
.btn.linkish {
  padding: 0.25rem 0.5rem;
  font-size: 0.82rem;
  background: transparent;
  border-color: var(--accent);
  color: var(--accent);
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
.custom-api-panel {
  margin-top: 0.55rem;
  padding: 0.7rem;
  border: 1px dashed var(--border);
  border-radius: 8px;
}
.parallel-workers {
  margin: 0.5rem 0 0 1.35rem;
  max-width: 200px;
}
.mode-hint {
  margin: 0.35rem 0 0.5rem 1.35rem;
  line-height: 1.45;
}
.task-list {
  display: grid;
  gap: 0.7rem;
}
.task-item {
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 0.7rem;
  background: rgba(0, 0, 0, 0.16);
}
.task-head p {
  margin: 0;
}
.msg {
  font-size: 0.88rem;
}
.job-issue {
  margin: 0.55rem 0 0.65rem;
  padding: 0.65rem 0.7rem;
  border: 1px solid rgba(255, 113, 113, 0.45);
  border-left: 3px solid var(--err);
  border-radius: 8px;
  background: rgba(255, 113, 113, 0.08);
}
.job-issue.cancelled {
  border-color: rgba(245, 170, 72, 0.45);
  border-left-color: #f0a33b;
  background: rgba(245, 170, 72, 0.08);
}
.job-issue-title {
  margin: 0 0 0.3rem;
  font-weight: 700;
  font-size: 0.9rem;
}
.job-issue-text,
.job-issue-meta {
  margin: 0.18rem 0;
  font-size: 0.84rem;
  overflow-wrap: anywhere;
}
.job-issue-meta {
  color: var(--muted);
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
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
  margin-top: 0.65rem;
  align-items: center;
}
.err {
  color: var(--err);
  font-size: 0.88rem;
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
.site-footer {
  margin-top: 1.25rem;
  padding: 1rem 0.5rem 0.25rem;
  text-align: center;
}
.brand-line {
  margin: 0 0 0.45rem;
  font-size: 1.05rem;
}
.brand-line a {
  color: #dbe8ff;
  text-decoration: none;
  margin-left: 0.3rem;
}
.heart-link {
  font-size: 1.15rem;
  margin-left: 0.35rem;
}
.support-mask {
  position: fixed;
  inset: 0;
  background: rgba(0, 0, 0, 0.55);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 20;
}
.support-modal {
  width: min(92vw, 760px);
  background: #0f1420;
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 1rem 1rem 1.1rem;
  position: relative;
}
.support-modal h3 {
  margin: 0 0 0.75rem;
  text-align: center;
}
.close-btn {
  position: absolute;
  top: 0.35rem;
  right: 0.5rem;
  border: none;
  background: transparent;
  color: #a8b3c7;
  font-size: 1.4rem;
  cursor: pointer;
}
.support-images {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
  gap: 0.85rem;
}
.support-images figure {
  margin: 0;
  background: rgba(255, 255, 255, 0.03);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 0.55rem;
  text-align: center;
}
.support-images img {
  width: 100%;
  max-height: 280px;
  object-fit: contain;
  border-radius: 8px;
}
.support-images figcaption {
  margin-top: 0.45rem;
  font-size: 0.85rem;
  color: var(--muted);
}
</style>
