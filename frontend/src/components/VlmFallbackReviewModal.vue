<script setup>
import { computed, onMounted, ref } from "vue";
import { authHeaders } from "../auth";

const props = defineProps({
  jobId: {
    type: String,
    required: true,
  },
});

const emit = defineEmits(["close", "updated"]);

const loading = ref(false);
const workingId = ref("");
const batchWorking = ref(false);
const errorText = ref("");
const report = ref(null);
const selectedReviewId = ref("");
const selectedReviewIds = ref({});
const statusFilter = ref("open");
const typeFilter = ref("all");
const searchText = ref("");
const reviewTexts = ref({});
const confidences = ref({});
const languages = ref({});
const comments = ref({});
const batchComment = ref("");

const reviews = computed(() => {
  const raw = Array.isArray(report.value?.reviews) ? report.value.reviews : [];
  const priorityWeight = { P0: 0, P1: 1, P2: 2, P3: 3 };
  const statusWeight = {
    pending_review: 0,
    needs_revision: 1,
    blocked_missing_visual_evidence: 2,
    accepted_result: 3,
    marked_unusable: 4,
  };
  return [...raw].sort((a, b) => {
    const as = statusWeight[a.effective_status] ?? 9;
    const bs = statusWeight[b.effective_status] ?? 9;
    if (as !== bs) return as - bs;
    const ap = priorityWeight[a.priority] ?? 9;
    const bp = priorityWeight[b.priority] ?? 9;
    if (ap !== bp) return ap - bp;
    return String(a.review_id || "").localeCompare(String(b.review_id || ""));
  });
});

const summary = computed(() => report.value?.summary || {});
const filteredReviews = computed(() => reviews.value.filter(reviewMatchesFilters));
const selectedReview = computed(
  () =>
    reviews.value.find((item) => item.review_id === selectedReviewId.value) ||
    filteredReviews.value[0] ||
    reviews.value[0] ||
    null,
);
const selectedCount = computed(() => Object.values(selectedReviewIds.value).filter(Boolean).length);
const selectedFilteredCount = computed(() =>
  filteredReviews.value.filter((item) => isSelected(item.review_id)).length,
);
const allFilteredSelected = computed(
  () => filteredReviews.value.length > 0 && selectedFilteredCount.value === filteredReviews.value.length,
);

function formatErrorPayload(payload) {
  if (!payload) return "请求失败";
  if (typeof payload === "string") return payload;
  const detail = payload.detail && typeof payload.detail === "object" ? payload.detail : payload;
  return detail.user_message || detail.message || detail.detail || detail.error || "请求失败";
}

function shortText(value, max = 160) {
  const text = String(value || "").replace(/\s+/g, " ").trim();
  if (!text) return "-";
  return text.length > max ? `${text.slice(0, max)}...` : text;
}

function statusLabel(status) {
  const labels = {
    pending_review: "待复核",
    needs_revision: "需复查",
    blocked_missing_visual_evidence: "缺视觉证据",
    accepted_result: "已接受",
    marked_unusable: "不可用",
  };
  return labels[status] || status || "未知";
}

function decisionLabel(decision) {
  const labels = {
    accept_result: "接受结果",
    mark_unusable: "标记不可用",
    needs_revision: "需复查",
  };
  return labels[decision] || decision || "未处理";
}

function typeLabel(item) {
  return item?.target_structure_type || item?.block_type || item?.layout_scope || "unknown";
}

function statusClass(status) {
  return `st-${status || "unknown"}`;
}

function reviewMatchesFilters(item) {
  if (statusFilter.value === "open" && !["pending_review", "needs_revision"].includes(item.effective_status)) {
    return false;
  }
  if (statusFilter.value !== "all" && statusFilter.value !== "open" && item.effective_status !== statusFilter.value) {
    return false;
  }
  if (typeFilter.value !== "all" && typeLabel(item) !== typeFilter.value) return false;
  const q = searchText.value.trim().toLowerCase();
  if (!q) return true;
  const haystack = [
    item.review_id,
    item.source_ocr_task_id,
    item.page_no,
    item.block_id,
    item.priority,
    item.effective_status,
    item.human_decision,
    item.review_text,
    ...(item.trigger_reasons || []),
    ...(item.review_goals || []),
    ...(item.expected_outputs || []),
  ]
    .map((value) => String(value || "").toLowerCase())
    .join(" ");
  return haystack.includes(q);
}

function isSelected(reviewId) {
  return !!selectedReviewIds.value[reviewId];
}

function setSelected(reviewId, checked) {
  if (!reviewId) return;
  const next = { ...selectedReviewIds.value };
  if (checked) next[reviewId] = true;
  else delete next[reviewId];
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

function pruneSelection(items) {
  const valid = new Set(items.map((item) => item.review_id));
  const next = {};
  for (const [reviewId, selected] of Object.entries(selectedReviewIds.value)) {
    if (selected && valid.has(reviewId)) next[reviewId] = true;
  }
  selectedReviewIds.value = next;
}

function clearSelection() {
  selectedReviewIds.value = {};
}

function selectReview(item) {
  if (item?.review_id) selectedReviewId.value = item.review_id;
}

function hydrateDrafts(items) {
  const nextTexts = { ...reviewTexts.value };
  const nextConfidences = { ...confidences.value };
  const nextLanguages = { ...languages.value };
  const nextComments = { ...comments.value };
  for (const item of items) {
    const id = item.review_id;
    if (!id) continue;
    if (nextTexts[id] === undefined) nextTexts[id] = item.review_text || "";
    if (nextConfidences[id] === undefined) nextConfidences[id] = item.review_confidence ?? 0.85;
    if (nextLanguages[id] === undefined) nextLanguages[id] = item.review_language || "unknown";
    if (nextComments[id] === undefined) nextComments[id] = item.human_comment || "";
  }
  reviewTexts.value = nextTexts;
  confidences.value = nextConfidences;
  languages.value = nextLanguages;
  comments.value = nextComments;
}

async function loadReport() {
  loading.value = true;
  errorText.value = "";
  try {
    const r = await fetch(`/api/jobs/${props.jobId}/vlm-fallback-review`, {
      headers: authHeaders(),
    });
    const data = await r.json().catch(() => ({}));
    if (!r.ok) {
      errorText.value = formatErrorPayload(data);
      return;
    }
    report.value = data;
    hydrateDrafts(reviews.value);
    pruneSelection(reviews.value);
    if (!selectedReviewId.value && reviews.value.length) {
      selectedReviewId.value = reviews.value[0].review_id;
    }
  } catch (err) {
    errorText.value = String(err?.message || err);
  } finally {
    loading.value = false;
  }
}

async function submitDecision(item, decision) {
  if (!item?.review_id) return;
  if (decision === "accept_result" && !String(reviewTexts.value[item.review_id] || "").trim()) {
    errorText.value = "接受结果前需要填写识别文本";
    return;
  }
  workingId.value = item.review_id;
  errorText.value = "";
  try {
    const body = {
      decision,
      comment: comments.value[item.review_id] || "",
    };
    if (decision === "accept_result") {
      body.review_text = reviewTexts.value[item.review_id] || "";
      body.review_confidence = Number(confidences.value[item.review_id] ?? 0.85);
      body.review_language = languages.value[item.review_id] || "unknown";
    }
    const r = await fetch(`/api/jobs/${props.jobId}/vlm-fallback-review/${encodeURIComponent(item.review_id)}`, {
      method: "POST",
      headers: authHeaders(true),
      body: JSON.stringify(body),
    });
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

async function submitBatch(decision) {
  const ids = Object.entries(selectedReviewIds.value)
    .filter(([, selected]) => selected)
    .map(([reviewId]) => reviewId);
  if (!ids.length) {
    errorText.value = "请先选择复核项";
    return;
  }
  batchWorking.value = true;
  errorText.value = "";
  try {
    const r = await fetch(`/api/jobs/${props.jobId}/vlm-fallback-review/batch`, {
      method: "POST",
      headers: authHeaders(true),
      body: JSON.stringify({
        review_ids: ids,
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
  <div class="modal-backdrop" @click.self="emit('close')">
    <section class="modal-card vlm-modal" role="dialog" aria-modal="true">
      <header class="modal-head">
        <div>
          <h2>VLM / 人工复核</h2>
          <p class="muted small">
            待审 {{ summary.review_required_count || 0 }} · 可回写 {{ summary.ready_for_writeback_count || 0 }} · 已接受 {{ summary.accepted_result_count || 0 }}
          </p>
        </div>
        <button type="button" class="icon-btn" @click="emit('close')" aria-label="关闭">×</button>
      </header>

      <div v-if="errorText" class="error-box">{{ errorText }}</div>
      <div v-if="loading" class="muted">加载中...</div>

      <div v-else class="review-shell">
        <aside class="review-list">
          <div class="filters">
            <select v-model="statusFilter">
              <option value="open">待处理</option>
              <option value="all">全部</option>
              <option value="pending_review">待复核</option>
              <option value="needs_revision">需复查</option>
              <option value="accepted_result">已接受</option>
              <option value="marked_unusable">不可用</option>
              <option value="blocked_missing_visual_evidence">缺视觉证据</option>
            </select>
            <select v-model="typeFilter">
              <option value="all">全部类型</option>
              <option value="table">表格</option>
              <option value="image">图像</option>
              <option value="formula">公式</option>
              <option value="page">整页</option>
              <option value="caption">图注</option>
            </select>
            <input v-model="searchText" type="search" placeholder="搜索页码、块、原因" />
          </div>

          <div class="batch-row">
            <label><input type="checkbox" :checked="allFilteredSelected" @change="toggleFilteredSelection" /> 本页</label>
            <span>{{ selectedCount }} 已选</span>
          </div>
          <input v-model="batchComment" class="batch-comment" placeholder="批量备注" />
          <div class="batch-actions">
            <button type="button" class="btn linkish" :disabled="batchWorking || !selectedCount" @click="submitBatch('needs_revision')">需复查</button>
            <button type="button" class="btn linkish" :disabled="batchWorking || !selectedCount" @click="submitBatch('mark_unusable')">不可用</button>
            <button type="button" class="btn linkish" :disabled="batchWorking || !selectedCount" @click="submitBatch('clear')">清空</button>
          </div>

          <button
            v-for="item in filteredReviews"
            :key="item.review_id"
            type="button"
            class="review-row"
            :class="{ active: selectedReview?.review_id === item.review_id }"
            @click="selectReview(item)"
          >
            <input type="checkbox" :checked="isSelected(item.review_id)" @click.stop @change="setSelected(item.review_id, $event.target.checked)" />
            <span class="row-main">
              <strong>{{ item.priority || "P?" }} · 第 {{ item.page_no || "-" }} 页 · {{ typeLabel(item) }}</strong>
              <small>{{ shortText(item.trigger_reasons?.join(" / "), 90) }}</small>
            </span>
            <span class="status-pill" :class="statusClass(item.effective_status)">{{ statusLabel(item.effective_status) }}</span>
          </button>
        </aside>

        <main v-if="selectedReview" class="review-detail">
          <div class="detail-title">
            <div>
              <h3>{{ selectedReview.review_id }}</h3>
              <p class="muted small">
                OCR {{ selectedReview.source_ocr_task_id || "-" }} · {{ decisionLabel(selectedReview.human_decision) }}
              </p>
            </div>
            <span class="status-pill" :class="statusClass(selectedReview.effective_status)">{{ statusLabel(selectedReview.effective_status) }}</span>
          </div>

          <div class="meta-grid">
            <span>页码：{{ selectedReview.page_no || "-" }}</span>
            <span>块：{{ selectedReview.block_id || "-" }}</span>
            <span>类型：{{ typeLabel(selectedReview) }}</span>
            <span>bbox：{{ shortText((selectedReview.bbox || []).join(", "), 80) }}</span>
          </div>

          <div class="path-grid">
            <label>
              裁剪图
              <input :value="selectedReview.input_path || '-'" readonly />
            </label>
            <label>
              页面预览
              <input :value="selectedReview.page_preview_path || '-'" readonly />
            </label>
          </div>

          <div class="tag-group">
            <span v-for="reason in selectedReview.trigger_reasons || []" :key="reason" class="tag">{{ reason }}</span>
          </div>

          <label class="field-block">
            识别文本
            <textarea v-model="reviewTexts[selectedReview.review_id]" rows="9" placeholder="填写人工/VLM 复核后恢复的文本、表格文本或公式文本"></textarea>
          </label>

          <div class="inline-fields">
            <label>
              置信度
              <input v-model.number="confidences[selectedReview.review_id]" type="number" min="0" max="1" step="0.01" />
            </label>
            <label>
              语言
              <input v-model="languages[selectedReview.review_id]" placeholder="eng / chi_sim / unknown" />
            </label>
          </div>

          <label class="field-block">
            备注
            <textarea v-model="comments[selectedReview.review_id]" rows="3" placeholder="说明人工判断依据"></textarea>
          </label>

          <div class="detail-actions">
            <button type="button" class="btn" :disabled="workingId === selectedReview.review_id" @click="submitDecision(selectedReview, 'accept_result')">接受并生成回写结果</button>
            <button type="button" class="btn linkish" :disabled="workingId === selectedReview.review_id" @click="submitDecision(selectedReview, 'needs_revision')">需复查</button>
            <button type="button" class="btn linkish" :disabled="workingId === selectedReview.review_id" @click="submitDecision(selectedReview, 'mark_unusable')">不可用</button>
            <button type="button" class="btn linkish" :disabled="workingId === selectedReview.review_id" @click="submitDecision(selectedReview, 'clear')">清空</button>
          </div>
        </main>
        <main v-else class="review-detail empty">暂无 VLM 复核项</main>
      </div>
    </section>
  </div>
</template>

<style scoped>
.modal-backdrop {
  position: fixed;
  inset: 0;
  z-index: 80;
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 1.25rem;
  background: rgba(15, 23, 42, 0.48);
}
.modal-card {
  width: min(1180px, 100%);
  max-height: 92vh;
  overflow: hidden;
  background: #fff;
  border: 1px solid #d8e1ef;
  border-radius: 8px;
  box-shadow: 0 18px 50px rgba(15, 23, 42, 0.24);
}
.modal-head {
  display: flex;
  justify-content: space-between;
  gap: 1rem;
  padding: 1rem 1.1rem;
  border-bottom: 1px solid #e5edf7;
}
.modal-head h2 {
  margin: 0;
  font-size: 1.2rem;
}
.icon-btn {
  width: 32px;
  height: 32px;
  border: 1px solid #cbd5e1;
  border-radius: 8px;
  background: #fff;
  cursor: pointer;
}
.review-shell {
  display: grid;
  grid-template-columns: 390px minmax(0, 1fr);
  min-height: 620px;
}
.review-list {
  overflow: auto;
  padding: 0.9rem;
  border-right: 1px solid #e5edf7;
  background: #f8fafc;
}
.filters {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 0.45rem;
}
.filters input {
  grid-column: 1 / -1;
}
select,
input,
textarea {
  width: 100%;
  box-sizing: border-box;
  border: 1px solid #cbd5e1;
  border-radius: 8px;
  padding: 0.52rem 0.6rem;
  font: inherit;
  background: #fff;
}
.batch-row,
.batch-actions,
.detail-actions,
.inline-fields {
  display: flex;
  gap: 0.5rem;
  align-items: center;
  flex-wrap: wrap;
}
.batch-row {
  justify-content: space-between;
  margin-top: 0.65rem;
  color: #475569;
  font-size: 0.88rem;
}
.batch-comment {
  margin: 0.45rem 0;
}
.review-row {
  width: 100%;
  display: grid;
  grid-template-columns: auto minmax(0, 1fr) auto;
  gap: 0.65rem;
  align-items: center;
  margin-top: 0.55rem;
  padding: 0.65rem;
  text-align: left;
  border: 1px solid #dbe5f2;
  border-radius: 8px;
  background: #fff;
  cursor: pointer;
}
.review-row.active {
  border-color: #2563eb;
  box-shadow: 0 0 0 2px rgba(37, 99, 235, 0.12);
}
.review-row input {
  width: auto;
}
.row-main {
  min-width: 0;
  display: grid;
  gap: 0.2rem;
}
.row-main strong,
.row-main small {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.row-main small {
  color: #64748b;
}
.status-pill,
.tag {
  display: inline-flex;
  align-items: center;
  min-height: 24px;
  border-radius: 999px;
  padding: 0.12rem 0.5rem;
  font-size: 0.78rem;
  background: #e2e8f0;
  color: #334155;
  white-space: nowrap;
}
.st-pending_review,
.st-needs_revision {
  background: #fff7ed;
  color: #9a3412;
}
.st-accepted_result {
  background: #dcfce7;
  color: #166534;
}
.st-marked_unusable,
.st-blocked_missing_visual_evidence {
  background: #fee2e2;
  color: #991b1b;
}
.review-detail {
  overflow: auto;
  padding: 1rem 1.1rem 1.2rem;
}
.detail-title {
  display: flex;
  justify-content: space-between;
  gap: 1rem;
  align-items: flex-start;
}
.detail-title h3 {
  margin: 0;
  font-size: 1.05rem;
}
.meta-grid,
.path-grid,
.inline-fields {
  display: grid;
  gap: 0.6rem;
  margin-top: 0.8rem;
}
.meta-grid {
  grid-template-columns: repeat(4, minmax(0, 1fr));
  color: #475569;
  font-size: 0.9rem;
}
.path-grid,
.inline-fields {
  grid-template-columns: repeat(2, minmax(0, 1fr));
}
.tag-group {
  display: flex;
  gap: 0.4rem;
  flex-wrap: wrap;
  margin-top: 0.8rem;
}
.field-block {
  display: grid;
  gap: 0.35rem;
  margin-top: 0.9rem;
  font-weight: 600;
  color: #334155;
}
.field-block textarea,
.inline-fields input,
.path-grid input {
  font-weight: 400;
}
.detail-actions {
  margin-top: 0.9rem;
}
.btn {
  border: 1px solid #2563eb;
  border-radius: 8px;
  background: #2563eb;
  color: #fff;
  padding: 0.52rem 0.8rem;
  cursor: pointer;
}
.btn:disabled {
  opacity: 0.55;
  cursor: not-allowed;
}
.btn.linkish {
  background: #fff;
  color: #2563eb;
}
.muted {
  color: #64748b;
}
.small {
  font-size: 0.86rem;
}
.error-box {
  margin: 0.8rem 1rem 0;
  padding: 0.7rem 0.85rem;
  border: 1px solid #fecaca;
  border-radius: 8px;
  background: #fef2f2;
  color: #991b1b;
}
.empty {
  display: grid;
  place-items: center;
  color: #64748b;
}
@media (max-width: 900px) {
  .review-shell {
    grid-template-columns: 1fr;
  }
  .review-list {
    max-height: 45vh;
    border-right: 0;
    border-bottom: 1px solid #e5edf7;
  }
  .meta-grid,
  .path-grid,
  .inline-fields {
    grid-template-columns: 1fr;
  }
}
</style>
