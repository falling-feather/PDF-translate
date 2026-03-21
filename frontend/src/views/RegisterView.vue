<script setup>
import { ref } from "vue";
import { useRouter } from "vue-router";
import { setSession } from "../auth";

const router = useRouter();
const username = ref("");
const password = ref("");
const err = ref("");
const loading = ref(false);

async function submit() {
  err.value = "";
  loading.value = true;
  try {
    const fd = new FormData();
    fd.append("username", username.value.trim());
    fd.append("password", password.value);
    const r = await fetch("/api/auth/register", { method: "POST", body: fd });
    const data = await r.json().catch(() => ({}));
    if (!r.ok) {
      err.value = typeof data.detail === "string" ? data.detail : JSON.stringify(data.detail || data);
      return;
    }
    setSession(data);
    router.replace("/translate");
  } catch (e) {
    err.value = String(e.message || e);
  } finally {
    loading.value = false;
  }
}
</script>

<template>
  <div class="page">
    <div class="card narrow">
      <h1>注册</h1>
      <p class="muted small">注册后仅可使用翻译工作台；管理员账号不可在此注册。</p>
      <form @submit.prevent="submit">
        <label class="field">
          <span>用户名</span>
          <input v-model="username" autocomplete="username" required />
        </label>
        <label class="field">
          <span>密码（至少 6 位）</span>
          <input v-model="password" type="password" autocomplete="new-password" minlength="6" required />
        </label>
        <p v-if="err" class="err">{{ err }}</p>
        <button class="btn primary" type="submit" :disabled="loading">{{ loading ? "提交中…" : "注册" }}</button>
      </form>
      <p class="muted small" style="margin-top: 1rem">
        已有账号？<router-link to="/login">登录</router-link>
      </p>
    </div>
  </div>
</template>

<style scoped>
.page {
  min-height: 100vh;
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 1.5rem;
}
.card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 1.5rem;
}
.narrow {
  width: 100%;
  max-width: 400px;
}
h1 {
  margin: 0 0 0.5rem;
  font-size: 1.35rem;
}
.field {
  display: flex;
  flex-direction: column;
  gap: 0.35rem;
  margin-bottom: 1rem;
}
.field span {
  font-size: 0.85rem;
  color: var(--muted);
}
.field input {
  padding: 0.55rem 0.65rem;
  border-radius: 8px;
  border: 1px solid var(--border);
  background: #0d1117;
  color: var(--text);
}
.btn {
  width: 100%;
  padding: 0.6rem;
  border-radius: 8px;
  border: 1px solid var(--border);
  font-weight: 600;
  cursor: pointer;
}
.btn.primary {
  background: var(--accent);
  color: #fff;
  border-color: #2563c9;
}
.btn:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}
.err {
  color: var(--err);
  font-size: 0.9rem;
}
</style>
