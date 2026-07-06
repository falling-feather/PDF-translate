<script setup>
import { computed, onMounted, ref } from "vue";
import { useRouter } from "vue-router";
import { authHeaders, clearSession, getUsername } from "../auth";
import GlossaryReviewModal from "../components/GlossaryReviewModal.vue";
import RepairPatchReviewModal from "../components/RepairPatchReviewModal.vue";
import TableMergedCellReviewModal from "../components/TableMergedCellReviewModal.vue";

const router = useRouter();
const tab = ref("settings");

const settings = ref({});
const saving = ref(false);
const audit = ref([]);
const users = ref([]);
const jobs = ref([]);
const reconcile = ref(null);
const reconciling = ref(false);
const glossaryReviewJobId = ref("");
const repairPatchReviewJobId = ref("");
const tableReviewJobId = ref("");

const auditExpanded = ref({});

const backendCatalog = computed(() => {
  const catalog = settings.value.backend_catalog;
  return Array.isArray(catalog) && catalog.length ? catalog : [
    { id: "echo", label: "echo（联调/测试）" },
    { id: "deepseek", label: "DeepSeek" },
  ];
});
const allBackendIds = computed(() => backendCatalog.value.map((b) => b.id));

function backendLabel(id) {
  const labels = settings.value.backend_labels || {};
  if (labels[id]) return labels[id];
  const hit = backendCatalog.value.find((b) => b.id === id);
  return hit?.label || id;
}

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

function artifactLabel(job) {
  const status = job.artifact_consistency_status || "missing_status";
  const labels = {
    ready: "产物就绪",
    partial: "部分产物",
    pending: "生成中",
    no_output: "暂无译文",
    inconsistent: "产物异常",
    missing_status: "状态缺失",
  };
  return labels[status] || status;
}

function artifactClass(job) {
  return {
    "artifact-ok": job.artifact_consistency_status === "ready",
    "artifact-warn": ["partial", "pending", "no_output"].includes(job.artifact_consistency_status),
    "artifact-err": !job.artifact_consistent,
  };
}

function artifactWarnings(job) {
  const warnings = Array.isArray(job.artifact_warnings) ? job.artifact_warnings : [];
  return warnings.join("、");
}

function artifactReadySummary(job) {
  const ready = [];
  if (job.input_pdf_ready) ready.push("原文PDF");
  if (job.partial_output_ready) ready.push("MD");
  if (job.translated_pdf_ready) ready.push("译文PDF");
  if (job.bilingual_html_ready) ready.push("HTML");
  if (job.glossary_review_ready) ready.push("术语确认");
  if (job.glossary_retranslation_plan_ready) ready.push("术语重译计划");
  if (job.table_merged_cell_review_ready) ready.push("表格确认");
  if (job.table_structure_publish_ready) ready.push("表格发布");
  if (job.repair_patch_review_ready) ready.push("补丁审核");
  if (job.repair_effectiveness_report_ready) ready.push("修复效果");
  if (job.repair_publish_report_ready) ready.push("修复报告");
  if (job.repair_published_full_ready) ready.push("修复稿");
  if (job.repair_rollback_report_ready) ready.push("回滚报告");
  if (job.repair_rollback_full_ready) ready.push("回滚稿");
  if (job.repair_formal_replace_report_ready) ready.push("正式替换");
  if (job.repair_formal_full_ready) ready.push("正式稿");
  if (job.repair_formal_rollback_report_ready) ready.push("正式回滚");
  if (job.repair_formal_backup_full_ready) ready.push("正式备份");
  if (job.bundle_zip_ready) ready.push("ZIP");
  return ready.length ? `可用：${ready.join(" / ")}` : "暂无可下载产物";
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
      enabled_backends: settings.value.enabled_backends || allBackendIds.value,
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

async function reconcileJobs(apply = false) {
  if (apply && !window.confirm("确认清理数据库缺目录记录和未索引任务目录？运行中的任务会跳过。")) return;
  reconciling.value = true;
  try {
    const r = await fetch("/api/admin/jobs/reconcile", {
      method: "POST",
      headers: authHeaders(true),
      body: JSON.stringify({ apply }),
    });
    if (!r.ok) {
      alert("巡检失败");
      return;
    }
    reconcile.value = await r.json();
    if (apply) await loadJobs();
  } finally {
    reconciling.value = false;
  }
}

const reconcileSummary = computed(() => {
  const drift = reconcile.value?.drift;
  if (!drift) return "";
  const missing = drift.missing_work_dir_count || 0;
  const unindexed = drift.unindexed_work_dir_count || 0;
  const deletedDb = reconcile.value.deleted_db_rows?.length || 0;
  const deletedDirs = reconcile.value.deleted_work_dirs?.length || 0;
  const skipped = reconcile.value.skipped_active?.length || 0;
  const base = `数据库缺目录 ${missing} 条，未索引目录 ${unindexed} 个`;
  if (!reconcile.value.apply) return base;
  return `${base}；已清数据库 ${deletedDb} 条、目录 ${deletedDirs} 个，跳过运行中 ${skipped} 个`;
});

function toggleBackend(b) {
  const cur = new Set(settings.value.enabled_backends || allBackendIds.value);
  if (cur.has(b)) cur.delete(b);
  else cur.add(b);
  settings.value.enabled_backends = Array.from(cur);
}

function logout() {
  clearSession();
  router.replace("/login");
}

function formatAdminError(payload) {
  if (!payload) return "操作失败";
  if (typeof payload === "string") return payload;
  const detail = payload.detail && typeof payload.detail === "object" ? payload.detail : payload;
  return detail.user_message || detail.message || detail.detail || detail.error || "操作失败";
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

async function adminReviewRepairPatch(job) {
  repairPatchReviewJobId.value = job.job_id;
}

async function adminReviewGlossary(job) {
  glossaryReviewJobId.value = job.job_id;
}

async function adminReviewTableMergedCell(job) {
  tableReviewJobId.value = job.job_id;
}

async function onGlossaryReviewUpdated() {
  await loadJobs();
  await loadAudit();
}

async function onTableReviewUpdated() {
  await loadJobs();
  await loadAudit();
}

async function onRepairPatchReviewUpdated() {
  await loadJobs();
  await loadAudit();
}

async function adminConfirmTableStructurePublish(job) {
  const blockingReviews = Number(job.table_merged_cell_review_required_count || 0);
  if (blockingReviews > 0) {
    alert(`仍有 ${blockingReviews} 个表格合并候选未完成确认，请先处理表格审核。`);
    return;
  }
  if (!window.confirm("确认生成表格结构副本？原始表格重建证据仍会保留。")) return;
  const r = await fetch(`/api/jobs/${job.job_id}/table-structure-publish/confirm`, {
    method: "POST",
    headers: authHeaders(),
  });
  const data = await r.json().catch(() => ({}));
  if (!r.ok) {
    alert(formatAdminError(data));
    return;
  }
  await loadJobs();
  await loadAudit();
  alert("已生成表格结构副本");
}

async function adminConfirmRepairPublish(job) {
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
  const r = await fetch(`/api/jobs/${job.job_id}/repair-publish/confirm`, {
    method: "POST",
    headers: authHeaders(),
  });
  const data = await r.json().catch(() => ({}));
  if (!r.ok) {
    alert(formatAdminError(data));
    return;
  }
  await loadJobs();
  await loadAudit();
  alert("已生成修复发布稿");
}

async function adminConfirmRepairRollback(job) {
  if (!job.repair_published_full_ready) {
    alert("修复发布稿尚未生成，暂不能执行回滚演练。");
    return;
  }
  if (!window.confirm("确认生成回滚演练稿？这会复制原始译文为 rollback_full.md，不会覆盖修复发布稿。")) return;
  const r = await fetch(`/api/jobs/${job.job_id}/repair-rollback/confirm`, {
    method: "POST",
    headers: authHeaders(),
  });
  const data = await r.json().catch(() => ({}));
  if (!r.ok) {
    alert(formatAdminError(data));
    return;
  }
  await loadJobs();
  await loadAudit();
  alert("已生成回滚演练稿");
}

async function adminConfirmRepairFormalReplace(job) {
  if (!job.repair_published_full_ready) {
    alert("修复发布稿尚未生成，暂不能生成正式译文。");
    return;
  }
  if (!window.confirm("确认把修复发布稿提升为正式译文？系统会保留正式译文修复前备份。")) return;
  const r = await fetch(`/api/jobs/${job.job_id}/repair-formal-replace/confirm`, {
    method: "POST",
    headers: authHeaders(),
  });
  const data = await r.json().catch(() => ({}));
  if (!r.ok) {
    alert(formatAdminError(data));
    return;
  }
  await loadJobs();
  await loadAudit();
  alert("已生成正式修复译文");
}

async function adminConfirmRepairFormalRollback(job) {
  if (!job.repair_formal_backup_full_ready) {
    alert("正式译文修复前备份尚未生成，暂不能正式回滚。");
    return;
  }
  if (!window.confirm("确认把正式译文恢复到修复前备份？当前修复正式稿会另存为回滚前副本。")) return;
  const r = await fetch(`/api/jobs/${job.job_id}/repair-formal-rollback/confirm`, {
    method: "POST",
    headers: authHeaders(),
  });
  const data = await r.json().catch(() => ({}));
  if (!r.ok) {
    alert(formatAdminError(data));
    return;
  }
  await loadJobs();
  await loadAudit();
  alert("已回滚正式译文");
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
        <label class="field">
          <span>默认后端</span>
          <select v-model="settings.default_backend">
            <option v-for="b in allBackendIds" :key="'default-' + b" :value="b">{{ backendLabel(b) }}</option>
          </select>
        </label>
        <label class="field"><span>HTTP 超时(秒)</span><input v-model="settings.http_timeout_s" /></label>
      </div>

      <h3>允许用户选择的后端</h3>
      <div class="checks">
        <label v-for="b in allBackendIds" :key="b" class="ck">
          <input
            type="checkbox"
            :checked="(settings.enabled_backends || allBackendIds).includes(b)"
            @change="toggleBackend(b)"
          />
          {{ backendLabel(b) }}
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
      <div class="job-tools">
        <button type="button" class="linkish" :disabled="reconciling" @click="reconcileJobs(false)">巡检任务存储</button>
        <button type="button" class="linkish danger-link" :disabled="reconciling" @click="reconcileJobs(true)">清理孤儿任务</button>
        <span v-if="reconcileSummary" class="muted small">{{ reconcileSummary }}</span>
      </div>
      <div class="scroll">
        <table class="table">
          <thead>
            <tr>
              <th>任务 ID</th>
              <th>用户</th>
              <th>文件</th>
              <th>时间</th>
              <th>产物</th>
              <th>下载</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="j in jobs" :key="j.job_id">
              <td><code>{{ j.job_id }}</code></td>
              <td>{{ j.username }} ({{ j.user_id }})</td>
              <td>{{ j.original_filename }}</td>
              <td class="muted nowrap">{{ j.created_at }}</td>
              <td>
                <span class="artifact-pill" :class="artifactClass(j)">{{ artifactLabel(j) }}</span>
                <span class="muted small artifact-note">{{ artifactReadySummary(j) }}</span>
                <span v-if="artifactWarnings(j)" class="muted small artifact-note">{{ artifactWarnings(j) }}</span>
              </td>
              <td class="nowrap">
                <button type="button" class="linkish" :disabled="!j.glossary_review_ready || Number(j.glossary_review_reviewable_count || j.glossary_review_pending_count || 0) <= 0" @click="adminReviewGlossary(j)">确认术语</button>
                <button type="button" class="linkish" :disabled="j.status !== 'done' || !j.repair_patch_review_ready" @click="adminReviewRepairPatch(j)">审核补丁</button>
                <button type="button" class="linkish" :disabled="j.status !== 'done' || !j.table_merged_cell_review_ready" @click="adminReviewTableMergedCell(j)">审核表格</button>
                <button type="button" class="linkish" :disabled="j.status !== 'done' || !j.table_merged_cell_review_ready || j.table_reconstruction_confirmed_ready || Number(j.table_merged_cell_review_required_count || 0) > 0" @click="adminConfirmTableStructurePublish(j)">确认表格</button>
                <button type="button" class="linkish" :disabled="!j.input_pdf_ready" @click="adminDownload(j.job_id, 'input', 'input.pdf')">原PDF</button>
                <button type="button" class="linkish" :disabled="!j.partial_output_ready" @click="adminDownload(j.job_id, 'output_md', 'translated.md')">MD</button>
                <button type="button" class="linkish" :disabled="!j.translated_pdf_ready" @click="adminDownload(j.job_id, 'output_pdf', 'translated.pdf')">译PDF</button>
                <button type="button" class="linkish" :disabled="!j.glossary_retranslation_plan_ready" @click="adminDownload(j.job_id, 'glossary_retranslation_plan_md', 'glossary_retranslation_plan.md')">重译计划</button>
                <button type="button" class="linkish" :disabled="!j.glossary_retranslation_plan_ready" @click="adminDownload(j.job_id, 'glossary_retranslation_plan_json', 'glossary_retranslation_plan.json')">计划JSON</button>
                <button type="button" class="linkish" :disabled="!j.repair_publish_report_ready" @click="adminDownload(j.job_id, 'repair_publish', 'repair_publish.md')">修复报告</button>
                <button type="button" class="linkish" :disabled="!j.repair_effectiveness_report_ready" @click="adminDownload(j.job_id, 'repair_effectiveness', 'repair_effectiveness.md')">修复效果</button>
                <button type="button" class="linkish" :disabled="!j.repair_rollback_report_ready" @click="adminDownload(j.job_id, 'repair_rollback', 'repair_rollback.md')">回滚报告</button>
                <button type="button" class="linkish" :disabled="!j.repair_patch_review_ready" @click="adminDownload(j.job_id, 'repair_patch_review', 'repair_patch_review.md')">补丁审核</button>
                <button type="button" class="linkish" :disabled="!j.table_merged_cell_review_ready" @click="adminDownload(j.job_id, 'table_merged_cell_review', 'table_merged_cell_review.md')">表格确认</button>
                <button type="button" class="linkish" :disabled="!j.table_structure_publish_ready" @click="adminDownload(j.job_id, 'table_structure_publish', 'table_structure_publish.md')">表格发布</button>
                <button type="button" class="linkish" :disabled="!j.table_reconstruction_confirmed_ready" @click="adminDownload(j.job_id, 'table_reconstruction_confirmed', 'table_reconstruction_confirmed.json')">表格副本</button>
                <button type="button" class="linkish" :disabled="j.status !== 'done' || !j.repair_publish_report_ready || j.repair_published_full_ready" @click="adminConfirmRepairPublish(j)">确认修复</button>
                <button type="button" class="linkish" :disabled="!j.repair_published_full_ready" @click="adminDownload(j.job_id, 'repair_published_full', 'published_full.md')">修复稿</button>
                <button type="button" class="linkish" :disabled="j.status !== 'done' || !j.repair_published_full_ready || j.repair_rollback_full_ready" @click="adminConfirmRepairRollback(j)">回滚演练</button>
                <button type="button" class="linkish" :disabled="!j.repair_rollback_full_ready" @click="adminDownload(j.job_id, 'repair_rollback_full', 'rollback_full.md')">回滚稿</button>
                <button type="button" class="linkish" :disabled="!j.repair_formal_replace_report_ready" @click="adminDownload(j.job_id, 'repair_formal_replace', 'repair_formal_replace.md')">正式替换报告</button>
                <button type="button" class="linkish" :disabled="j.status !== 'done' || !j.repair_published_full_ready || j.repair_formal_full_ready" @click="adminConfirmRepairFormalReplace(j)">生成正式稿</button>
                <button type="button" class="linkish" :disabled="!j.repair_formal_full_ready" @click="adminDownload(j.job_id, 'repair_formal_full', 'formal_full.md')">正式稿</button>
                <button type="button" class="linkish" :disabled="j.status !== 'done' || !j.repair_formal_backup_full_ready || j.repair_formal_rollback_applied" @click="adminConfirmRepairFormalRollback(j)">正式回滚</button>
                <button type="button" class="linkish" :disabled="!j.repair_formal_rollback_report_ready" @click="adminDownload(j.job_id, 'repair_formal_rollback', 'repair_formal_rollback.md')">正式回滚报告</button>
                <button type="button" class="linkish" :disabled="!j.repair_formal_backup_full_ready" @click="adminDownload(j.job_id, 'repair_formal_backup_full', 'formal_full.before_repair.md')">修复前备份</button>
                <button type="button" class="linkish" :disabled="!j.bundle_zip_ready" @click="adminDownload(j.job_id, 'bundle_zip', j.job_id + '.zip')">ZIP</button>
              </td>
            </tr>
          </tbody>
        </table>
      </div>
    </section>
    <TableMergedCellReviewModal
      v-if="tableReviewJobId"
      :job-id="tableReviewJobId"
      title="表格合并候选审核"
      @updated="onTableReviewUpdated"
      @close="tableReviewJobId = ''"
    />
    <GlossaryReviewModal
      v-if="glossaryReviewJobId"
      :job-id="glossaryReviewJobId"
      title="术语确认"
      @updated="onGlossaryReviewUpdated"
      @close="glossaryReviewJobId = ''"
    />
    <RepairPatchReviewModal
      v-if="repairPatchReviewJobId"
      :job-id="repairPatchReviewJobId"
      title="局部修复补丁审核"
      @updated="onRepairPatchReviewUpdated"
      @close="repairPatchReviewJobId = ''"
    />
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
.job-tools {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 0.5rem 0.75rem;
  margin: 0.25rem 0 0.75rem;
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
.linkish:disabled {
  color: var(--muted);
  cursor: not-allowed;
  opacity: 0.55;
}
.danger-link {
  color: var(--err);
}
.artifact-pill {
  display: inline-block;
  padding: 0.12rem 0.38rem;
  border-radius: 999px;
  border: 1px solid var(--border);
  font-size: 0.76rem;
  white-space: nowrap;
}
.artifact-ok {
  color: var(--ok);
  border-color: rgba(52, 199, 89, 0.45);
}
.artifact-warn {
  color: #f5c542;
  border-color: rgba(245, 197, 66, 0.45);
}
.artifact-err {
  color: var(--err);
  border-color: rgba(255, 92, 92, 0.45);
}
.artifact-note {
  display: block;
  margin-top: 0.2rem;
  max-width: 220px;
  white-space: normal;
  word-break: break-word;
}
.muted {
  color: var(--muted);
}
.small {
  font-size: 0.82rem;
}
</style>
