<script setup>
import { computed, onMounted, onUnmounted, ref } from "vue";
import { authHeaders } from "../auth";

const props = defineProps({
  jobId: {
    type: String,
    required: true,
  },
  title: {
    type: String,
    default: "表格合并候选审核",
  },
});

const emit = defineEmits(["close", "updated"]);

const loading = ref(false);
const errorText = ref("");
const report = ref(null);
const comments = ref({});
const workingId = ref("");
const selectedReviewId = ref("");
const previewUrl = ref("");
const previewLoading = ref(false);
const previewError = ref("");
const previewSeq = ref(0);
const statusFilter = ref("open");
const evidenceFilter = ref("all");
const searchText = ref("");
const selectedReviewIds = ref({});
const batchComment = ref("");
const batchWorking = ref(false);

function formatErrorPayload(payload) {
  if (!payload) return "请求失败";
  if (typeof payload === "string") return payload;
  const detail = payload.detail && typeof payload.detail === "object" ? payload.detail : payload;
  return detail.user_message || detail.message || detail.detail || detail.error || "请求失败";
}

const reviews = computed(() => {
  const raw = Array.isArray(report.value?.candidate_reviews) ? report.value.candidate_reviews : [];
  const weight = {
    pending_review: 0,
    needs_revision: 1,
    human_confirmed: 2,
    rejected: 3,
  };
  return [...raw].sort((a, b) => {
    const aw = weight[a.confirmation_status] ?? 9;
    const bw = weight[b.confirmation_status] ?? 9;
    if (aw !== bw) return aw - bw;
    return String(a.review_id || "").localeCompare(String(b.review_id || ""));
  });
});

const summary = computed(() => report.value?.summary || {});
const selectedReview = computed(() =>
  reviews.value.find((item) => item.review_id === selectedReviewId.value) || null,
);
const filteredReviews = computed(() => reviews.value.filter(reviewMatchesFilters));
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
  const status = item.confirmation_status || "pending_review";
  if (statusFilter.value === "open" && !["pending_review", "needs_revision"].includes(status)) {
    return false;
  }
  if (statusFilter.value !== "open" && statusFilter.value !== "all" && status !== statusFilter.value) {
    return false;
  }
  const bbox = item.bbox_evidence_status || "missing";
  const visual = item.visual_evidence_level || "none";
  if (evidenceFilter.value === "visual" && !["visual_span_bbox", "manual_verified"].includes(visual)) {
    return false;
  }
  if (evidenceFilter.value === "estimated" && bbox !== "estimated" && visual !== "estimated_bbox") {
    return false;
  }
  if (evidenceFilter.value === "missing" && bbox !== "missing" && visual !== "none") {
    return false;
  }
  const q = searchText.value.trim().toLowerCase();
  if (!q) return true;
  const haystack = [
    item.review_id,
    item.table_id,
    item.block_id,
    item.text,
    item.reason,
    item.default_decision,
    item.span_type,
    item.source,
    item.engine,
  ]
    .map((value) => String(value || "").toLowerCase())
    .join(" ");
  return haystack.includes(q);
}

function statusLabel(status) {
  const labels = {
    pending_review: "待确认",
    needs_revision: "需复查",
    human_confirmed: "已确认",
    rejected: "已拒绝",
  };
  return labels[status] || status || "待确认";
}

function evidenceText(item) {
  const bbox = item.bbox_evidence_status || "missing";
  const visual = item.visual_evidence_level || "none";
  return `${bbox} / ${visual}`;
}

function spanText(item) {
  return `${item.span_type || "unknown"} ${item.row_span || 1}x${item.column_span || 1}`;
}

function anchorText(item) {
  return `p${item.page_no || "-"} · r${item.row_index ?? "-"}c${item.column_index ?? "-"}`;
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

function clearPreviewUrl() {
  if (previewUrl.value) {
    URL.revokeObjectURL(previewUrl.value);
    previewUrl.value = "";
  }
}

async function loadPreview(item) {
  if (!item?.review_id) return;
  const seq = previewSeq.value + 1;
  previewSeq.value = seq;
  selectedReviewId.value = item.review_id;
  previewLoading.value = true;
  previewError.value = "";
  clearPreviewUrl();
  try {
    const r = await fetch(
      `/api/jobs/${props.jobId}/table-merged-cell-review/${encodeURIComponent(item.review_id)}/preview.png`,
      { headers: authHeaders() },
    );
    if (seq !== previewSeq.value) return;
    if (!r.ok) {
      const data = await r.json().catch(() => ({}));
      previewError.value = formatErrorPayload(data);
      return;
    }
    const blob = await r.blob();
    if (seq !== previewSeq.value) return;
    previewUrl.value = URL.createObjectURL(blob);
  } catch (err) {
    if (seq === previewSeq.value) {
      previewError.value = String(err?.message || err);
    }
  } finally {
    if (seq === previewSeq.value) {
      previewLoading.value = false;
    }
  }
}

async function loadReport() {
  loading.value = true;
  errorText.value = "";
  try {
    const r = await fetch(`/api/jobs/${props.jobId}/table-merged-cell-review`, {
      headers: authHeaders(),
    });
    const data = await r.json().catch(() => ({}));
    if (!r.ok) {
      errorText.value = formatErrorPayload(data);
      return;
    }
    report.value = data;
    const next = {};
    for (const item of Array.isArray(data.candidate_reviews) ? data.candidate_reviews : []) {
      next[item.review_id] = item.human_comment || "";
    }
    comments.value = next;
    pruneSelection(reviews.value);
    const visible = filteredReviews.value;
    const nextPreview =
      visible.find((item) => item.review_id === selectedReviewId.value) || visible[0] || reviews.value[0];
    if (nextPreview) {
      await loadPreview(nextPreview);
    } else {
      selectedReviewId.value = "";
      previewError.value = "";
      clearPreviewUrl();
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
      `/api/jobs/${props.jobId}/table-merged-cell-review/${encodeURIComponent(item.review_id)}`,
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
    const r = await fetch(`/api/jobs/${props.jobId}/table-merged-cell-review/batch`, {
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
onUnmounted(clearPreviewUrl);
</script>

<template>
  <div class="review-mask" @click.self="emit('close')">
    <section class="review-modal">
      <header class="review-head">
        <div>
          <h3>{{ title }}</h3>
          <p class="muted small">
            {{ summary.candidate_review_count || 0 }} 项 · 待确认 {{ summary.review_required_count || 0 }} · 已复核 {{ summary.human_reviewed_count || 0 }}
          </p>
        </div>
        <button type="button" class="close-btn" @click="emit('close')">×</button>
      </header>

      <p v-if="errorText" class="err">{{ errorText }}</p>
      <p v-if="loading" class="muted">加载中…</p>
      <p v-else-if="!reviews.length" class="muted">暂无候选。</p>

      <div v-else class="review-workspace">
        <div class="review-toolbar">
          <label>
            <span>状态</span>
            <select v-model="statusFilter">
              <option value="open">待处理</option>
              <option value="pending_review">待确认</option>
              <option value="needs_revision">需复查</option>
              <option value="human_confirmed">已确认</option>
              <option value="rejected">已拒绝</option>
              <option value="all">全部</option>
            </select>
          </label>
          <label>
            <span>证据</span>
            <select v-model="evidenceFilter">
              <option value="all">全部</option>
              <option value="visual">视觉支持</option>
              <option value="estimated">估算 bbox</option>
              <option value="missing">缺证据</option>
            </select>
          </label>
          <label class="search-field">
            <span>搜索</span>
            <input v-model.trim="searchText" type="search" placeholder="ID、文本或原因" />
          </label>
          <button type="button" :disabled="!filteredReviews.length" @click="toggleFilteredSelection">
            {{ allFilteredSelected ? "取消当前筛选" : "选择当前筛选" }}
          </button>
          <button type="button" :disabled="!selectedCount" @click="clearSelection">清空选择</button>
          <span class="muted small">显示 {{ filteredReviews.length }} / 已选 {{ selectedCount }}</span>
        </div>

        <div class="batch-bar">
          <input v-model.trim="batchComment" type="text" placeholder="批量备注（可选）" />
          <button type="button" :disabled="!selectedCount || batchWorking" @click="submitBatchDecision('confirm')">
            批量确认
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

        <p v-if="!filteredReviews.length" class="muted">当前筛选没有候选。</p>

        <div v-else class="review-body">
          <aside class="preview-panel">
            <div class="preview-title">
              <strong>页面预览</strong>
              <span v-if="selectedReview" class="muted small">{{ selectedReview.review_id }}</span>
            </div>
            <p v-if="previewLoading" class="muted small">正在加载预览...</p>
            <p v-else-if="previewError" class="err small">{{ previewError }}</p>
            <img v-else-if="previewUrl" :src="previewUrl" alt="表格候选页面预览" />
            <p v-else class="muted small">选择一条候选查看页面框选。</p>
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
                <th>候选</th>
                <th>位置</th>
                <th>状态</th>
                <th>证据</th>
                <th>文本</th>
                <th>备注</th>
                <th>操作</th>
              </tr>
            </thead>
            <tbody>
              <tr v-for="item in filteredReviews" :key="item.review_id">
                <td class="select-col">
                  <input
                    type="checkbox"
                    :checked="isSelected(item.review_id)"
                    @change="setSelected(item.review_id, $event.target.checked)"
                  />
                </td>
                <td>
                  <code>{{ item.review_id }}</code>
                  <span class="muted block">{{ spanText(item) }}</span>
                </td>
                <td class="nowrap">{{ anchorText(item) }}</td>
                <td>
                  <span class="status-pill" :class="'st-' + (item.confirmation_status || 'pending_review')">
                    {{ statusLabel(item.confirmation_status) }}
                  </span>
                  <span class="muted block">{{ item.default_decision || "-" }}</span>
                </td>
                <td class="evidence">{{ evidenceText(item) }}</td>
                <td class="candidate-text">{{ item.text || "—" }}</td>
                <td>
                  <textarea
                    v-model="comments[item.review_id]"
                    rows="2"
                    :disabled="workingId === item.review_id || batchWorking"
                  />
                </td>
                <td class="actions-cell">
                  <button
                    type="button"
                    :class="{ active: selectedReviewId === item.review_id }"
                    :disabled="previewLoading && selectedReviewId === item.review_id"
                    @click="loadPreview(item)"
                  >
                    预览
                  </button>
                  <button type="button" :disabled="!!workingId || batchWorking" @click="submitDecision(item, 'confirm')">确认</button>
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
  width: min(96vw, 1180px);
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
  min-width: min(240px, 100%);
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
  grid-template-columns: minmax(220px, 320px) minmax(0, 1fr);
  gap: 0.75rem;
  min-height: 0;
}
.preview-panel {
  min-height: 0;
  max-height: 64vh;
  overflow: auto;
  border: 1px solid var(--border);
  border-radius: 8px;
  background: #0d1117;
  padding: 0.7rem;
}
.preview-title {
  display: flex;
  flex-direction: column;
  gap: 0.2rem;
  margin-bottom: 0.55rem;
  overflow-wrap: anywhere;
}
.preview-panel img {
  display: block;
  width: 100%;
  height: auto;
  border-radius: 6px;
  border: 1px solid rgba(255, 255, 255, 0.08);
  background: #fff;
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
.select-col {
  width: 32px;
  min-width: 32px;
}
.select-col input {
  width: 16px;
  height: 16px;
}
.candidate-text {
  min-width: 150px;
  max-width: 240px;
  overflow-wrap: anywhere;
}
.evidence {
  min-width: 118px;
  color: var(--muted);
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
  min-width: 148px;
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
.actions-cell button.active {
  border-color: rgba(91, 157, 255, 0.7);
  color: var(--accent);
}
.status-pill {
  display: inline-block;
  padding: 0.12rem 0.4rem;
  border-radius: 999px;
  border: 1px solid var(--border);
  white-space: nowrap;
}
.st-human_confirmed {
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
.st-pending_review {
  color: var(--accent);
  border-color: rgba(91, 157, 255, 0.45);
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
.nowrap {
  white-space: nowrap;
}
.err {
  color: var(--err);
  font-size: 0.88rem;
}
@media (max-width: 860px) {
  .review-body {
    grid-template-columns: 1fr;
  }
  .preview-panel {
    max-height: 34vh;
  }
}
</style>
