import { createRouter, createWebHistory } from "vue-router";
import { getRole, getToken } from "./auth";
import LoginView from "./views/LoginView.vue";
import RegisterView from "./views/RegisterView.vue";
import UserView from "./views/UserView.vue";
import AdminView from "./views/AdminView.vue";

const router = createRouter({
  history: createWebHistory(),
  routes: [
    { path: "/", redirect: "/login" },
    { path: "/login", component: LoginView, meta: { guest: true } },
    { path: "/register", component: RegisterView, meta: { guest: true } },
    { path: "/translate", component: UserView, meta: { requiresAuth: true } },
    { path: "/admin", component: AdminView, meta: { requiresAuth: true, requiresAdmin: true } },
  ],
});

router.beforeEach((to, _from, next) => {
  const token = getToken();
  const role = getRole();
  if (to.meta.requiresAuth && !token) {
    next("/login");
    return;
  }
  if (to.meta.requiresAdmin && role !== "admin") {
    next("/translate");
    return;
  }
  if (to.meta.guest && token) {
    next(role === "admin" ? "/admin" : "/translate");
    return;
  }
  next();
});

export default router;
