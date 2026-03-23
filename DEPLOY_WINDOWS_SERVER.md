# Windows 服务器部署（精简版 · 暂不接域名）

按顺序做即可。做完后：**用浏览器打开 `http://服务器公网IP:901`** 访问（阿里云安全组须放行 **901** 端口）。

---

## 一、要装什么（一次性）

| 软件 | 用途 | 获取方式 |
|------|------|----------|
| **Git** | 拉代码 | 你已安装 |
| **Python 3.10+** | 运行后端 | https://www.python.org/downloads/ 安装时勾选 **Add Python to PATH** |
| **Node.js LTS** | 构建前端（可选：也可在本地 build 后拷贝 static） | https://nodejs.org/ |

无需单独装「拓展」；Python 依赖由下面 `pip install -e .` 自动装齐。

---

## 二、用 Git 下载代码（一条线）

打开 **PowerShell** 或 **cmd**，执行（路径可改，下面用 `C:\apps\PDF-translate`）：

```bat
mkdir C:\apps 2>nul
cd C:\apps
git clone https://github.com/falling-feather/PDF-translate.git
cd PDF-translate
```

私有仓库则需先配置 Git 登录（Token 或 SSH）。

---

## 三、Python 环境与依赖

```bat
cd C:\apps\PDF-translate
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install -U pip
pip install -e .
```

以后每次开新终端要先执行：`cd C:\apps\PDF-translate` 再 `.\.venv\Scripts\activate`。

---

## 四、构建前端（生成网页静态文件）

```bat
cd C:\apps\PDF-translate\frontend
npm ci
npm run build
cd ..
```

---

## 五、数据目录 + 环境变量（最少两项）

```bat
mkdir C:\pdf-translate-data
```

**设置环境变量**（任选一种）：

- **图形界面**：Win + R → `sysdm.cpl` → **高级** → **环境变量** → 「用户变量」新建：  
  - 名称：`PDF_TRANSLATE_DATA`  
  - 值：`C:\pdf-translate-data`  
- **仅当前窗口临时生效**（测试用）：

```bat
set PDF_TRANSLATE_DATA=C:\pdf-translate-data
```

**首次创建管理员**（仅数据库里还没有任何用户时有效）：可再建一个用户变量：

- `PDF_TRANSLATE_BOOTSTRAP_ADMIN_PASSWORD` = 你自己定的强密码  

（管理员用户名默认见项目 `SETUP_MANUAL.md` / 代码中的 `PDF_TRANSLATE_ADMIN_USERNAME`。）

翻译用的 **API Key** 可之后在网页 **管理后台** 里填；也可在环境变量里设 `DEEPSEEK_API_KEY` 等（见 `SETUP_MANUAL.md`）。

---

## 六、阿里云安全组（必做，否则外网打不开）

登录 **阿里云 ECS 控制台** → 本实例 **安全组** → **入方向规则** → 添加：

- 端口：**901**，协议 **TCP**，源 **0.0.0.0/0**（或只允许公司网段更安全）。

**本机防火墙**：「Windows Defender 防火墙」→ **高级设置** → **入站规则** → 新建 → 端口 **TCP 901** → 允许连接。

---

## 七、启动命令（傻瓜式）

在已 `activate` 的 venv 下，项目根目录执行：

```bat
cd C:\apps\PDF-translate
.\.venv\Scripts\activate
set PDF_TRANSLATE_DATA=C:\pdf-translate-data
set PDF_TRANSLATE_WEB_HOST=0.0.0.0
set PDF_TRANSLATE_WEB_PORT=901
python -m pdf_translate.server
```

若已在「系统环境变量」里永久设置了 `PDF_TRANSLATE_DATA`，可省略 `set PDF_TRANSLATE_DATA`。

**访问链接：**

- 在服务器本机浏览器：`http://127.0.0.1:901`
- 同事电脑浏览器：`http://你的ECS公网IP:901`

---

## 八、可选：做成一键启动 bat

在 `C:\apps\PDF-translate` 下自建 `run_server.bat`（内容示例）：

```bat
@echo off
chcp 65001 >nul
cd /d C:\apps\PDF-translate
call .venv\Scripts\activate.bat
set PDF_TRANSLATE_DATA=C:\pdf-translate-data
set PDF_TRANSLATE_WEB_HOST=0.0.0.0
set PDF_TRANSLATE_WEB_PORT=901
python -m pdf_translate.server
pause
```

双击运行即可（窗口不要关，关了服务就停）。

---

## 九、以后要更新代码

```bat
cd C:\apps\PDF-translate
git pull
.\.venv\Scripts\activate
pip install -e .
cd frontend
npm ci
npm run build
```

然后重新执行第七节或第八节启动。

---

## 十、和域名的关系（你说过后续再做）

当前方案是 **IP + 端口 901**。以后接 **fallingfeather.cn** 时，再改为：域名解析 + 本机装 **Caddy** 或 **IIS 反代**，把 **443** 转到本机 **901**，那时可把安全组里的 **901 对公网**关掉，只留 **80/443**。与本文步骤独立，可后补。
