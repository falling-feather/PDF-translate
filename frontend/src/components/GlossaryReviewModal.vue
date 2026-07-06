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
    default: "术语确认",
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
const statusFilter = ref("pending");
const typeFilter = ref("all");
const searchText = ref("");
const batchComment = ref("");
const batchWorking = ref(false);

function formatErrorPayload(payload) {
  if (!payload) return "请求失败";
  if (typeof payload === "string") return payload;
  const detail = payload.detail && typeof payload.detail === "object" ? payload.detail : payload;
  return detail.user_message || detail.message || detail.detail || detail.error || "请求失败";
}

const summary = computed(() => report.value?.summary || {});
const reviews = computed(() => {
  const raw = Array.isArray(report.value?.pending_reviews) ? report.value.pending_reviews : [];
  return [...raw].sort((a, b) => {
    const ap = a.status === "pending" ? 0 : 1;
    const bp = b.status === "pending" ? 0 : 1;
    if (ap !== bp) return ap - bp;
    const at = a.type === "glossary_conflict" ? 0 : 1;
    const bt = b.type === "glossary_conflict" ? 0 : 1;
    if (at !== bt) return at - bt;
    return String(a.review_id || "").localeCompare(String(b.review_id || ""));
  });
});
const filteredReviews = computed(() => reviews.value.filter(reviewMatchesFilters));
const selectedReview = computed(() =>
  reviews.value.find((item) => item.review_id === selectedReviewId.value) ||
  filteredReviews.value[0] ||
  reviews.value[0] ||
  null,
);
const actionableFilteredReviews = computed(() => filteredReviews.value.filter(isActionable));
const selectedCount = computed(() => Object.values(selectedReviewIds.value).filter(Boolean).length);
const selectedActionableFilteredCount = computed(() =>
  actionableFilteredReviews.value.filter((item) => isSelected(item.review_id)).length,
);
const allFilteredSelected = computed(
  () =>
    actionableFilteredReviews.value.length > 0 &&
    selectedActionableFilteredCount.value === actionableFilteredReviews.value.length,
);

function reviewMatchesFilters(item) {
  if (statusFilter.value !== "all" && (item.status || "pending") !== statusFilter.value) return false;
  if (typeFilter.value !== "all" && (item.type || "pending") !== typeFilter.value) return false;
  const q = searchText.value.trim().toLowerCase();
  if (!q) return true;
  const haystack = [
    item.review_id,
    item.type,
    item.status,
    item.en,
    item.candidate_zh,
    item.confirmed_zh,
    item.reason,
    item.source,
    item.review_comment,
    ...(Array.isArray(item.existing_zh) ? item.existing_zh : []),
    ...(Array.isArray(item.existing_en) ? item.existing_en : []),
  ]
    .map((value) => String(value || "").toLowerCase())
    .join(" ");
  return haystack.includes(q);
}

function typeLabel(type) {
  const labels = {
    glossary_conflict: "译名冲突",
    shared_translation_review: "共享译名",
  };
  return labels[type] || type || "待确认";
}

function statusLabel(status) {
  const labels = {
    pending: "待处理",
    confirmed: "已采用",
    rejected: "已拒绝",
  };
  return labels[status] || status || "待处理";
}

function statusClass(status) {
  if (status === "confirmed") return "st-confirmed";
  if (status === "rejected") return "st-rejected";
  return "st-pending";
}

function pageText(item) {
  const page = item?.first_page;
  return page === undefined || page === null || page === "" ? "-" : String(page);
}

function existingText(item) {
  const values = Array.isArray(item?.existing_zh) ? item.existing_zh : [];
  return values.length ? values.join(" / ") : "-";
}

function isActionable(item) {
  return item?.type === "glossary_conflict" && (item.status || "pending") === "pending";
}

function shortText(value, max = 120) {
  const text = String(value || "").replace(/\s+/g, " ").trim();
  if (!text) return "-";
  return text.length > max ? `${text.slice(0, max)}...` : text;
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
    for (const item of actionableFilteredReviews.value) delete next[item.review_id];
  } else {
    for (const item of actionableFilteredReviews.value) next[item.review_id] = true;
  }
  selectedReviewIds.value = next;
}

function clearSelection() {
  selectedReviewIds.value = {};
}

function pruneSelection(items) {
  const valid = new Set(items.filter(isActionable).map((item) => item.review_id));
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
    const r = await fetch(`/api/jobs/${props.jobId}/glossary-review`, {
      headers: authHeaders(),
    });
    const data = await r.json().catch(() => ({}));
    if (!r.ok) {
      errorText.value = formatErrorPayload(data);
      return;
    }
    report.value = data;
    const next = {};
    for (const item of Array.isArray(data.pending_reviews) ? data.pending_reviews : []) {
      next[item.review_id] = item.review_comment || "";
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
  if (!item?.review_id || !isActionable(item) || workingId.value || batchWorking.value) return;
  workingId.value = item.review_id;
  errorText.value = "";
  try {
    const r = await fetch(
      `/api/jobs/${props.jobId}/glossary-review/${encodeURIComponent(item.review_id)}`,
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
    const r = await fetch(`/api/jobs/${props.jobId}/glossary-review/batch`, {
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
            术语 {{ summary.term_count || 0 }} 项 · 待处理 {{ summary.pending_count || 0 }} · 冲突 {{ summary.pending_glossary_conflict_count || 0 }} · 已确认 {{ summary.confirmed_count || 0 }}
          </p>
        </div>
        <button type="button" class="close-btn" @click="emit('close')">x</button>
      </header>

      <p v-if="errorText" class="err">{{ errorText }}</p>
      <p v-if="loading" class="muted">加载中...</p>
      <p v-else-if="!reviews.length" class="muted">暂无术语待确认项。</p>

      <div v-else class="review-workspace">
        <div class="review-toolbar">
          <label>
            <span>状态</span>
            <select v-model="statusFilter">
              <option value="pending">待处理</option>
              <option value="confirmed">已采用</option>
              <option value="rejected">已拒绝</option>
              <option value="all">全部</option>
            </select>
          </label>
          <label>
            <span>类型</span>
            <select v-model="typeFilter">
              <option value="all">全部</option>
              <option value="glossary_conflict">译名冲突</option>
              <option value="shared_translation_review">共享译名</option>
            </select>
          </label>
          <label class="search-field">
            <span>搜索</span>
            <input v-model.trim="searchText" type="search" placeholder="英文、候选译名、原因或来源" />
          </label>
          <button type="button" :disabled="!actionableFilteredReviews.length" @click="toggleFilteredSelection">
            {{ allFilteredSelected ? "取消当前选择" : "选择当前可处理项" }}
          </button>
          <button type="button" :disabled="!selectedCount" @click="clearSelection">清空选择</button>
          <span class="muted small">显示 {{ filteredReviews.length }} / 已选 {{ selectedCount }}</span>
        </div>

        <div class="batch-bar">
          <input v-model.trim="batchComment" type="text" placeholder="批量备注，可选" />
          <button type="button" :disabled="!selectedCount || batchWorking" @click="submitBatchDecision('confirm_candidate')">
            批量采用候选
          </button>
          <button type="button" :disabled="!selectedCount || batchWorking" @click="submitBatchDecision('reject_candidate')">
            批量拒绝候选
          </button>
        </div>

        <p v-if="!filteredReviews.length" class="muted">当前筛选没有术语项。</p>

        <div v-else class="review-body">
          <aside class="detail-panel">
            <div class="detail-title">
              <strong>术语详情</strong>
              <span v-if="selectedReview" class="muted small">{{ selectedReview.review_id }}</span>
            </div>
            <template v-if="selectedReview">
              <dl>
                <dt>英文术语</dt>
                <dd>{{ selectedReview.en || "-" }}</dd>
                <dt>当前译名</dt>
                <dd>{{ existingText(selectedReview) }}</dd>
                <dt>候选译名</dt>
                <dd>{{ selectedReview.candidate_zh || selectedReview.confirmed_zh || "-" }}</dd>
                <dt>页面 / 来源</dt>
                <dd>{{ pageText(selectedReview) }} · {{ selectedReview.source || "-" }}</dd>
                <dt>原因</dt>
                <dd>{{ selectedReview.reason || "-" }}</dd>
                <dt>处理记录</dt>
                <dd>
                  {{ statusLabel(selectedReview.status) }}
                  <span v-if="selectedReview.reviewed_by"> · {{ selectedReview.reviewed_by }}</span>
                  <span v-if="selectedReview.reviewed_at"> · {{ selectedReview.reviewed_at }}</span>
                </dd>
              </dl>
              <p v-if="!isActionable(selectedReview)" class="hint">
                该类型暂不支持在此处直接处理。
              </p>
            </template>
            <p v-else class="muted small">选择一条术语查看详情。</p>
          </aside>

          <div class="review-scroll">
            <table class="review-table">
              <thead>
                <tr>
                  <th class="select-col">
                    <input
                      type="checkbox"
                      :checked="allFilteredSelected"
                      :disabled="!actionableFilteredReviews.length"
                      @change="toggleFilteredSelection"
                    />
                  </th>
                  <th>术语</th>
                  <th>译名</th>
                  <th>类型</th>
                  <th>状态</th>
                  <th>来源</th>
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
                      :disabled="!isActionable(item)"
                      @change="setSelected(item.review_id, $event.target.checked)"
                    />
                  </td>
                  <td>
                    <strong>{{ item.en || "-" }}</strong>
                    <span class="muted block">第 {{ pageText(item) }} 页</span>
                  </td>
                  <td class="term-cell">
                    <span class="block">当前：{{ existingText(item) }}</span>
                    <span class="candidate block">候选：{{ item.candidate_zh || item.confirmed_zh || "-" }}</span>
                    <span class="muted block">{{ shortText(item.reason) }}</span>
                  </td>
                  <td>{{ typeLabel(item.type) }}</td>
                  <td>
                    <span class="status-pill" :class="statusClass(item.status)">
                      {{ statusLabel(item.status) }}
                    </span>
                  </td>
                  <td>{{ item.source || "-" }}</td>
                  <td @click.stop>
                    <textarea
                      v-model="comments[item.review_id]"
                      rows="2"
                      :disabled="workingId === item.review_id || batchWorking || !isActionable(item)"
                    />
                  </td>
                  <td class="actions-cell" @click.stop>
                    <button
                      type="button"
                      :disabled="!!workingId || batchWorking || !isActionable(item)"
                      @click="submitDecision(item, 'confirm_candidate')"
                    >
                      采用
                    </button>
                    <button
                      type="button"
                      :disabled="!!workingId || batchWorking || !isActionable(item)"
                      @click="submitDecision(item, 'reject_candidate')"
                    >
                      拒绝
                    </button>
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
  font-size: 1.2rem;
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
  grid-template-columns: minmax(240px, 320px) minmax(0, 1fr);
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
.hint {
  margin: 0.75rem 0 0;
  color: #f0c674;
  font-size: 0.82rem;
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
.term-cell {
  min-width: 180px;
  max-width: 320px;
  overflow-wrap: anywhere;
}
.candidate {
  color: #d7e3ff;
}
textarea {
  width: 170px;
  max-width: 24vw;
  resize: vertical;
  border-radius: 8px;
  border: 1px solid var(--border);
  background: #0d1117;
  color: var(--text);
  padding: 0.35rem;
}
.actions-cell {
  min-width: 112px;
}
.actions-cell button {
  display: inline-block;
  margin: 0 0.25rem 0.25rem 0;
  padding: 0.3rem 0.45rem;
  border-radius: 8px;
  border: 1px solid var(--border);
  background: #151d2e;
  color: var(--text);
  cursor: pointer;
}
.actions-cell button:disabled {
  cursor: not-allowed;
  opacity: 0.55;
}
.status-pill {
  display: inline-flex;
  align-items: center;
  min-height: 24px;
  border-radius: 999px;
  padding: 0.1rem 0.45rem;
  border: 1px solid var(--border);
  white-space: nowrap;
}
.st-pending {
  color: #f0c674;
}
.st-confirmed {
  color: #5fd18b;
}
.st-rejected {
  color: #ff8f8f;
}
.muted {
  color: var(--muted);
}
.small {
  font-size: 0.8rem;
}
.block {
  display: block;
  margin-top: 0.16rem;
}
.err {
  color: #ff8f8f;
}
@media (max-width: 780px) {
  .review-modal {
    max-height: 94vh;
  }
  .review-body {
    grid-template-columns: 1fr;
  }
  textarea {
    max-width: 48vw;
  }
}
</style>
