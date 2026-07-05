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
    const nextPreview = reviews.value.find((item) => item.review_id === selectedReviewId.value) || reviews.value[0];
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
  if (!item?.review_id || workingId.value) return;
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
            <tr v-for="item in reviews" :key="item.review_id">
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
                  :disabled="workingId === item.review_id"
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
                <button type="button" :disabled="!!workingId" @click="submitDecision(item, 'confirm')">确认</button>
                <button type="button" :disabled="!!workingId" @click="submitDecision(item, 'reject')">拒绝</button>
                <button type="button" :disabled="!!workingId" @click="submitDecision(item, 'needs_revision')">复查</button>
                <button type="button" :disabled="!!workingId" @click="submitDecision(item, 'clear')">清空</button>
              </td>
            </tr>
          </tbody>
          </table>
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
