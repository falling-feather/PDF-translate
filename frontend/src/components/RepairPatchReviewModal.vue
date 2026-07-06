<script setup>
import { computed, onMounted, ref } from "vue";
import { authHeaders } from "../auth";

const props = defineProps({
  jobId: {
    type: String,
    required: true,
  },
  title: {
    type: String,
    default: "局部修复补丁审核",
  },
});

const emit = defineEmits(["close", "updated"]);

const loading = ref(false);
const errorText = ref("");
const report = ref(null);
const comments = ref({});
const workingId = ref("");
const selectedReviewId = ref("");
const selectedReviewIds = ref({});
const statusFilter = ref("open");
const riskFilter = ref("all");
const searchText = ref("");
const batchComment = ref("");
const batchWorking = ref(false);

function formatErrorPayload(payload) {
  if (!payload) return "请求失败";
  if (typeof payload === "string") return payload;
  const detail = payload.detail && typeof payload.detail === "object" ? payload.detail : payload;
  return detail.user_message || detail.message || detail.detail || detail.error || "请求失败";
}

const reviews = computed(() => {
  const raw = Array.isArray(report.value?.patch_reviews) ? report.value.patch_reviews : [];
  const riskWeight = { critical: 0, high: 1, medium: 2, low: 3 };
  return [...raw].sort((a, b) => {
    const ab = a.publish_blocking ? 0 : 1;
    const bb = b.publish_blocking ? 0 : 1;
    if (ab !== bb) return ab - bb;
    const ar = riskWeight[a.risk_level] ?? 9;
    const br = riskWeight[b.risk_level] ?? 9;
    if (ar !== br) return ar - br;
    return String(a.review_id || "").localeCompare(String(b.review_id || ""));
  });
});

const summary = computed(() => report.value?.summary || {});
const filteredReviews = computed(() => reviews.value.filter(reviewMatchesFilters));
const selectedReview = computed(() =>
  reviews.value.find((item) => item.review_id === selectedReviewId.value) || filteredReviews.value[0] || reviews.value[0] || null,
);
const selectedCount = computed(() =>
  Object.values(selectedReviewIds.value).filter(Boolean).length,
);
const selectedFilteredCount = computed(() =>
  filteredReviews.value.filter((item) => isSelected(item.review_id)).length,
);
const allFilteredSelected = computed(
  () => filteredReviews.value.length > 0 && selectedFilteredCount.value === filteredReviews.value.length,
);

function reviewMatchesFilters(item) {
  const human = item.human_decision || "";
  const effective = item.effective_decision || "";
  if (statusFilter.value === "open" && !item.publish_blocking) return false;
  if (statusFilter.value === "pending" && human) return false;
  if (statusFilter.value === "approved" && effective !== "approve_candidate") return false;
  if (statusFilter.value === "rejected" && effective !== "reject_candidate") return false;
  if (statusFilter.value === "needs_revision" && effective !== "manual_review_required") return false;
  if (riskFilter.value !== "all" && (item.risk_level || "unknown") !== riskFilter.value) return false;
  const q = searchText.value.trim().toLowerCase();
  if (!q) return true;
  const target = item.merge_target && typeof item.merge_target === "object" ? item.merge_target : {};
  const structureContext =
    target.structure_patch_context && typeof target.structure_patch_context === "object"
      ? target.structure_patch_context
      : {};
  const haystack = [
    item.review_id,
    item.request_id,
    item.repair_id,
    item.chunk_id,
    item.issue_type,
    item.priority,
    item.action,
    item.scope,
    item.merge_status,
    item.merge_strategy,
    item.risk_level,
    item.default_decision,
    item.effective_decision,
    item.reason,
    item.decision_reason,
    item.result_excerpt,
    item.patched_chunk_path,
    item.result_path,
    target.table_index,
    structureContext.relevant_patch_count,
  ]
    .map((value) => String(value || "").toLowerCase())
    .join(" ");
  return haystack.includes(q);
}

function decisionLabel(decision) {
  const labels = {
    approve_candidate: "可发布",
    reject_candidate: "拒绝",
    manual_review_required: "需人工处理",
    approve: "已通过",
    reject: "已拒绝",
    needs_revision: "需复查",
  };
  return labels[decision] || decision || "未处理";
}

function riskLabel(risk) {
  const labels = {
    critical: "严重",
    high: "高",
    medium: "中",
    low: "低",
  };
  return labels[risk] || risk || "未知";
}

function statusClass(item) {
  if (item.effective_decision === "approve_candidate") return "st-approved";
  if (item.effective_decision === "reject_candidate") return "st-rejected";
  return "st-needs_revision";
}

function riskClass(risk) {
  return `risk-${risk || "unknown"}`;
}

function formatPages(item) {
  const pages = Array.isArray(item.pages_1based) ? item.pages_1based : [];
  if (!pages.length) return "-";
  if (pages.length === 1) return String(pages[0]);
  return `${pages[0]}-${pages[pages.length - 1]}`;
}

function mergeTargetText(item) {
  const target = item.merge_target && typeof item.merge_target === "object" ? item.merge_target : {};
  const parts = [];
  if (target.table_index !== undefined && target.table_index !== null && target.table_index !== "") {
    parts.push(`表格 ${Number(target.table_index) + 1}`);
  }
  if (target.cell_count) parts.push(`${target.cell_count} 单元格`);
  const structureContext =
    target.structure_patch_context && typeof target.structure_patch_context === "object"
      ? target.structure_patch_context
      : {};
  if (structureContext.relevant_patch_count) {
    parts.push(`结构上下文 ${structureContext.relevant_patch_count}`);
  }
  return parts.join(" · ") || "-";
}

function shortText(value, max = 160) {
  const text = String(value || "").replace(/\s+/g, " ").trim();
  if (!text) return "—";
  return text.length > max ? `${text.slice(0, max)}...` : text;
}

function isSelected(reviewId) {
  return !!selectedReviewIds.value[reviewId];
}

function setSelected(reviewId, checked) {
  if (!reviewId) return;
  const next = { ...selectedReviewIds.value };
  if (checked) {
    next[reviewId] = true;
  } else {
    delete next[reviewId];
  }
  selectedReviewIds.value = next;
}

function toggleFilteredSelection() {
  const next = { ...selectedReviewIds.value };
  if (allFilteredSelected.value) {
    for (const item of filteredReviews.value) delete next[item.review_id];
  } else {
    for (const item of filteredReviews.value) next[item.review_id] = true;
  }
  selectedReviewIds.value = next;
}

function clearSelection() {
  selectedReviewIds.value = {};
}

function pruneSelection(items) {
  const valid = new Set(items.map((item) => item.review_id));
  const next = {};
  for (const [reviewId, selected] of Object.entries(selectedReviewIds.value)) {
    if (selected && valid.has(reviewId)) next[reviewId] = true;
  }
  selectedReviewIds.value = next;
}

function selectReview(item) {
  if (item?.review_id) selectedReviewId.value = item.review_id;
}

async function loadReport() {
  loading.value = true;
  errorText.value = "";
  try {
    const r = await fetch(`/api/jobs/${props.jobId}/repair-patch-review`, {
      headers: authHeaders(),
    });
    const data = await r.json().catch(() => ({}));
    if (!r.ok) {
      errorText.value = formatErrorPayload(data);
      return;
    }
    report.value = data;
    const next = {};
    for (const item of Array.isArray(data.patch_reviews) ? data.patch_reviews : []) {
      next[item.review_id] = item.human_comment || "";
    }
    comments.value = next;
    pruneSelection(reviews.value);
    const visible = filteredReviews.value;
    if (!visible.find((item) => item.review_id === selectedReviewId.value)) {
      selectedReviewId.value = visible[0]?.review_id || reviews.value[0]?.review_id || "";
    }
  } catch (err) {
    errorText.value = String(err?.message || err);
  } finally {
    loading.value = false;
  }
}

async function submitDecision(item, decision) {
  if (!item?.review_id || workingId.value || batchWorking.value) return;
  workingId.value = item.review_id;
  errorText.value = "";
  try {
    const r = await fetch(
      `/api/jobs/${props.jobId}/repair-patch-review/${encodeURIComponent(item.review_id)}`,
      {
        method: "POST",
        headers: authHeaders(true),
        body: JSON.stringify({
          decision,
          comment: comments.value[item.review_id] || "",
        }),
      },
    );
    const data = await r.json().catch(() => ({}));
    if (!r.ok) {
      errorText.value = formatErrorPayload(data);
      return;
    }
    emit("updated", data);
    await loadReport();
  } catch (err) {
    errorText.value = String(err?.message || err);
  } finally {
    workingId.value = "";
  }
}

async function submitBatchDecision(decision) {
  const reviewIds = Object.entries(selectedReviewIds.value)
    .filter(([, selected]) => selected)
    .map(([reviewId]) => reviewId);
  if (!reviewIds.length || workingId.value || batchWorking.value) return;
  batchWorking.value = true;
  errorText.value = "";
  try {
    const r = await fetch(`/api/jobs/${props.jobId}/repair-patch-review/batch`, {
      method: "POST",
      headers: authHeaders(true),
      body: JSON.stringify({
        review_ids: reviewIds,
        decision,
        comment: batchComment.value || "",
      }),
    });
    const data = await r.json().catch(() => ({}));
    if (!r.ok) {
      errorText.value = formatErrorPayload(data);
      return;
    }
    clearSelection();
    batchComment.value = "";
    emit("updated", data);
    await loadReport();
  } catch (err) {
    errorText.value = String(err?.message || err);
  } finally {
    batchWorking.value = false;
  }
}

onMounted(loadReport);
</script>

<template>
  <div class="review-mask" @click.self="emit('close')">
    <section class="review-modal">
      <header class="review-head">
        <div>
          <h3>{{ title }}</h3>
          <p class="muted small">
            {{ summary.patch_count || 0 }} 项 · 阻断 {{ summary.publish_blocking_count || 0 }} · 已复核 {{ summary.human_reviewed_count || 0 }} · 可发布 {{ summary.effective_safe_count || 0 }}
          </p>
        </div>
        <button type="button" class="close-btn" @click="emit('close')">×</button>
      </header>

      <p v-if="errorText" class="err">{{ errorText }}</p>
      <p v-if="loading" class="muted">加载中...</p>
      <p v-else-if="!reviews.length" class="muted">暂无可审核补丁。</p>

      <div v-else class="review-workspace">
        <div class="review-toolbar">
          <label>
            <span>状态</span>
            <select v-model="statusFilter">
              <option value="open">发布阻断</option>
              <option value="pending">未人工处理</option>
              <option value="approved">可发布</option>
              <option value="rejected">已拒绝</option>
              <option value="needs_revision">需复查</option>
              <option value="all">全部</option>
            </select>
          </label>
          <label>
            <span>风险</span>
            <select v-model="riskFilter">
              <option value="all">全部</option>
              <option value="critical">严重</option>
              <option value="high">高</option>
              <option value="medium">中</option>
              <option value="low">低</option>
            </select>
          </label>
          <label class="search-field">
            <span>搜索</span>
            <input v-model.trim="searchText" type="search" placeholder="ID、chunk、问题、原因或片段" />
          </label>
          <button type="button" :disabled="!filteredReviews.length" @click="toggleFilteredSelection">
            {{ allFilteredSelected ? "取消当前筛选" : "选择当前筛选" }}
          </button>
          <button type="button" :disabled="!selectedCount" @click="clearSelection">清空选择</button>
          <span class="muted small">显示 {{ filteredReviews.length }} / 已选 {{ selectedCount }}</span>
        </div>

        <div class="batch-bar">
          <input v-model.trim="batchComment" type="text" placeholder="批量备注（可选）" />
          <button type="button" :disabled="!selectedCount || batchWorking" @click="submitBatchDecision('approve')">
            批量通过
          </button>
          <button type="button" :disabled="!selectedCount || batchWorking" @click="submitBatchDecision('reject')">
            批量拒绝
          </button>
          <button type="button" :disabled="!selectedCount || batchWorking" @click="submitBatchDecision('needs_revision')">
            批量复查
          </button>
          <button type="button" :disabled="!selectedCount || batchWorking" @click="submitBatchDecision('clear')">
            批量清空
          </button>
        </div>

        <p v-if="!filteredReviews.length" class="muted">当前筛选没有补丁。</p>

        <div v-else class="review-body">
          <aside class="detail-panel">
            <div class="detail-title">
              <strong>补丁详情</strong>
              <span v-if="selectedReview" class="muted small">{{ selectedReview.review_id }}</span>
            </div>
            <template v-if="selectedReview">
              <dl>
                <dt>定位</dt>
                <dd>{{ selectedReview.chunk_id || "-" }} · 页 {{ formatPages(selectedReview) }}</dd>
                <dt>问题</dt>
                <dd>{{ selectedReview.issue_type || "-" }} · {{ selectedReview.action || "-" }} · {{ selectedReview.scope || "-" }}</dd>
                <dt>目标</dt>
                <dd>{{ mergeTargetText(selectedReview) }}</dd>
                <dt>建议</dt>
                <dd>{{ selectedReview.decision_reason || "-" }}</dd>
                <dt>原因</dt>
                <dd>{{ selectedReview.reason || "-" }}</dd>
                <dt>文件</dt>
                <dd>
                  <code>{{ selectedReview.patched_chunk_path || "-" }}</code>
                  <code>{{ selectedReview.result_path || "-" }}</code>
                </dd>
              </dl>
              <pre v-if="selectedReview.result_excerpt" class="excerpt">{{ selectedReview.result_excerpt }}</pre>
              <p v-else class="muted small">当前补丁没有候选片段。</p>
            </template>
            <p v-else class="muted small">选择一条补丁查看详情。</p>
          </aside>

          <div class="review-scroll">
            <table class="review-table">
              <thead>
                <tr>
                  <th class="select-col">
                    <input
                      type="checkbox"
                      :checked="allFilteredSelected"
                      :disabled="!filteredReviews.length"
                      @change="toggleFilteredSelection"
                    />
                  </th>
                  <th>补丁</th>
                  <th>定位</th>
                  <th>风险</th>
                  <th>状态</th>
                  <th>内容</th>
                  <th>备注</th>
                  <th>操作</th>
                </tr>
              </thead>
              <tbody>
                <tr
                  v-for="item in filteredReviews"
                  :key="item.review_id"
                  :class="{ active: selectedReviewId === item.review_id }"
                  @click="selectReview(item)"
                >
                  <td class="select-col" @click.stop>
                    <input
                      type="checkbox"
                      :checked="isSelected(item.review_id)"
                      @change="setSelected(item.review_id, $event.target.checked)"
                    />
                  </td>
                  <td>
                    <code>{{ item.review_id }}</code>
                    <span class="muted block">{{ item.repair_id || item.request_id || "-" }}</span>
                  </td>
                  <td class="location-cell">
                    <span>{{ item.chunk_id || "-" }} · 页 {{ formatPages(item) }}</span>
                    <span class="muted block">{{ mergeTargetText(item) }}</span>
                  </td>
                  <td>
                    <span class="risk-pill" :class="riskClass(item.risk_level)">
                      {{ riskLabel(item.risk_level) }}
                    </span>
                    <span class="muted block">{{ item.merge_status || "-" }}</span>
                  </td>
                  <td>
                    <span class="status-pill" :class="statusClass(item)">
                      {{ decisionLabel(item.effective_decision) }}
                    </span>
                    <span class="muted block">{{ item.human_decision ? decisionLabel(item.human_decision) : "默认：" + decisionLabel(item.default_decision) }}</span>
                  </td>
                  <td class="candidate-text">
                    <strong>{{ item.issue_type || "-" }}</strong>
                    <span class="muted block">{{ item.action || "-" }} · {{ item.scope || "-" }}</span>
                    <span class="block">{{ shortText(item.result_excerpt || item.reason || item.decision_reason) }}</span>
                  </td>
                  <td @click.stop>
                    <textarea
                      v-model="comments[item.review_id]"
                      rows="2"
                      :disabled="workingId === item.review_id || batchWorking"
                    />
                  </td>
                  <td class="actions-cell" @click.stop>
                    <button type="button" :disabled="!!workingId || batchWorking" @click="submitDecision(item, 'approve')">通过</button>
                    <button type="button" :disabled="!!workingId || batchWorking" @click="submitDecision(item, 'reject')">拒绝</button>
                    <button type="button" :disabled="!!workingId || batchWorking" @click="submitDecision(item, 'needs_revision')">复查</button>
                    <button type="button" :disabled="!!workingId || batchWorking" @click="submitDecision(item, 'clear')">清空</button>
                  </td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </section>
  </div>
</template>

<style scoped>
.review-mask {
  position: fixed;
  inset: 0;
  z-index: 30;
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 1rem;
  background: rgba(0, 0, 0, 0.58);
}
.review-modal {
  width: min(96vw, 1220px);
  max-height: 88vh;
  display: flex;
  flex-direction: column;
  background: #0f1420;
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 1rem;
}
.review-head {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 1rem;
  margin-bottom: 0.75rem;
}
.review-head h3 {
  margin: 0 0 0.25rem;
  font-size: 1rem;
}
.close-btn {
  border: none;
  background: transparent;
  color: #a8b3c7;
  font-size: 1.45rem;
  cursor: pointer;
}
.review-workspace {
  min-height: 0;
  display: flex;
  flex-direction: column;
  gap: 0.65rem;
}
.review-toolbar,
.batch-bar {
  display: flex;
  align-items: end;
  flex-wrap: wrap;
  gap: 0.5rem;
}
.review-toolbar label {
  display: flex;
  flex-direction: column;
  gap: 0.22rem;
  min-width: 120px;
  color: var(--muted);
  font-size: 0.78rem;
}
.review-toolbar select,
.review-toolbar input,
.batch-bar input {
  min-height: 32px;
  border-radius: 8px;
  border: 1px solid var(--border);
  background: #0d1117;
  color: var(--text);
  padding: 0.35rem 0.5rem;
}
.review-toolbar .search-field {
  min-width: min(260px, 100%);
  flex: 1;
}
.review-toolbar button,
.batch-bar button {
  min-height: 32px;
  padding: 0.35rem 0.55rem;
  border-radius: 8px;
  border: 1px solid var(--border);
  background: #0d1117;
  color: var(--text);
  cursor: pointer;
}
.review-toolbar button:disabled,
.batch-bar button:disabled {
  cursor: not-allowed;
  opacity: 0.55;
}
.batch-bar input {
  min-width: min(280px, 100%);
  flex: 1;
}
.review-body {
  display: grid;
  grid-template-columns: minmax(240px, 340px) minmax(0, 1fr);
  gap: 0.75rem;
  min-height: 0;
}
.detail-panel {
  min-height: 0;
  max-height: 64vh;
  overflow: auto;
  border: 1px solid var(--border);
  border-radius: 8px;
  background: #0d1117;
  padding: 0.7rem;
}
.detail-title {
  display: flex;
  flex-direction: column;
  gap: 0.2rem;
  margin-bottom: 0.55rem;
  overflow-wrap: anywhere;
}
dl {
  margin: 0;
}
dt {
  margin-top: 0.55rem;
  color: var(--muted);
  font-size: 0.76rem;
}
dd {
  margin: 0.15rem 0 0;
  overflow-wrap: anywhere;
}
dd code {
  display: block;
  margin-top: 0.15rem;
  white-space: normal;
}
.excerpt {
  margin: 0.7rem 0 0;
  max-height: 260px;
  overflow: auto;
  white-space: pre-wrap;
  overflow-wrap: anywhere;
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 0.6rem;
  background: #080b11;
  color: var(--text);
  font-size: 0.78rem;
}
.review-scroll {
  min-width: 0;
  overflow: auto;
}
.review-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 0.82rem;
}
.review-table th,
.review-table td {
  text-align: left;
  vertical-align: top;
  padding: 0.45rem 0.4rem;
  border-bottom: 1px solid var(--border);
}
.review-table tbody tr {
  cursor: pointer;
}
.review-table tbody tr.active {
  background: rgba(91, 157, 255, 0.08);
}
.select-col {
  width: 32px;
  min-width: 32px;
}
.select-col input {
  width: 16px;
  height: 16px;
}
.location-cell {
  min-width: 128px;
}
.candidate-text {
  min-width: 170px;
  max-width: 280px;
  overflow-wrap: anywhere;
}
textarea {
  width: 180px;
  max-width: 26vw;
  resize: vertical;
  border-radius: 8px;
  border: 1px solid var(--border);
  background: #0d1117;
  color: var(--text);
  padding: 0.4rem 0.45rem;
}
.actions-cell {
  min-width: 132px;
}
.actions-cell button {
  margin: 0 0.25rem 0.25rem 0;
  padding: 0.26rem 0.45rem;
  border-radius: 8px;
  border: 1px solid var(--border);
  background: #0d1117;
  color: var(--text);
  cursor: pointer;
}
.actions-cell button:disabled {
  cursor: not-allowed;
  opacity: 0.55;
}
.status-pill,
.risk-pill {
  display: inline-block;
  padding: 0.12rem 0.4rem;
  border-radius: 999px;
  border: 1px solid var(--border);
  white-space: nowrap;
}
.st-approved {
  color: var(--ok);
  border-color: rgba(52, 199, 89, 0.45);
}
.st-rejected {
  color: var(--err);
  border-color: rgba(255, 92, 92, 0.45);
}
.st-needs_revision {
  color: #f5c542;
  border-color: rgba(245, 197, 66, 0.45);
}
.risk-critical,
.risk-high {
  color: var(--err);
  border-color: rgba(255, 92, 92, 0.45);
}
.risk-medium {
  color: #f5c542;
  border-color: rgba(245, 197, 66, 0.45);
}
.risk-low {
  color: var(--ok);
  border-color: rgba(52, 199, 89, 0.45);
}
.muted {
  color: var(--muted);
}
.small {
  font-size: 0.82rem;
}
.block {
  display: block;
  margin-top: 0.2rem;
}
.err {
  color: var(--err);
  font-size: 0.88rem;
}
@media (max-width: 900px) {
  .review-body {
    grid-template-columns: 1fr;
  }
  .detail-panel {
    max-height: 36vh;
  }
}
</style>
