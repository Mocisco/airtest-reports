import os; print("APP_PATH=", os.path.abspath(__file__))

import os
import json
import datetime
import shutil
import airtest_service
import threading
import ctypes
import time
import socket
import subprocess

from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_socketio import SocketIO
try:
    from playwright_cloud import CloudAutoConnector
except ImportError:
    CloudAutoConnector = None

GLOBAL_CLOUD_CONNECTOR = None
from werkzeug.utils import secure_filename

app = Flask(__name__, template_folder='templates', static_folder='static')
app.config['SECRET_KEY'] = 'secret_key_for_session'  # 或沿用原来的 'secret!'
socketio = SocketIO(app, async_mode='threading', cors_allowed_origins='*')


# UI explore (bulk) in-flight guard
UI_EXPLORE_LOCK = threading.Lock()
UI_EXPLORE_INFLIGHT = set()  # serial set
# ================= 配置区域 =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SCREENSHOT_ROOT_DIR = os.path.join(BASE_DIR, 'screenshots')
SCREENSHOT_ARCHIVE_DIR = os.path.join(BASE_DIR, 'screenshots_archive')
SCREENSHOT_ARCHIVE_ZIP_DIR = os.path.join(SCREENSHOT_ARCHIVE_DIR, 'zips')
ARCHIVE_INDEX_FILE = os.path.join(SCREENSHOT_ARCHIVE_ZIP_DIR, 'archives_index.json')
META_DATA_FILE = os.path.join(BASE_DIR, 'data', 'reports_meta.json')
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
EDGE_PATH = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
CDP_PORT = 9222
CDP_PROFILE = r"C:\pw_edge_cdp_profile"

# 云真机账号（用于自动占用/兜底释放�?
CLOUD_USERNAME = "YOUR_USERNAME"
CLOUD_PASSWORD = "YOUR_PASSWORD"

# ================= UI 标注台配�?=================
UI_KB_FILE = os.path.join(BASE_DIR, 'data', 'ui_knowledge_base.json')
IMAGE_TEMPLATES_FILE = os.path.join(BASE_DIR, 'data', 'image_templates.json')

# ================= 报告对比模板（基�?Report�?=================
# 需求：在某个报告上“一键设为模板”，后续对比页默认用该模板作为左侧基准�?
COMPARE_TEMPLATE_FILE = os.path.join(BASE_DIR, 'data', 'compare_template.json')

def _load_compare_template() -> dict:
    try:
        if os.path.exists(COMPARE_TEMPLATE_FILE):
            with open(COMPARE_TEMPLATE_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
    except Exception:
        pass
    return {}

def _save_compare_template(data: dict) -> None:
    try:
        os.makedirs(os.path.dirname(COMPARE_TEMPLATE_FILE), exist_ok=True)
        tmp = COMPARE_TEMPLATE_FILE + '.tmp'
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(data or {}, f, ensure_ascii=False, indent=2)
        os.replace(tmp, COMPARE_TEMPLATE_FILE)
    except Exception:
        pass

def get_baseline_rid() -> str:
    data = _load_compare_template()
    rid = (data.get('baseline_rid') or '').strip()
    if not rid:
        return ''
    # 避免指向已删除报�?
    if not manager.get(rid):
        return ''
    return rid

def _read_json(path, default=None):
    if default is None:
        default = {}
    try:
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception:
        pass
    return default

def _write_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def _sync_external_ui_assets():
    """可选：�?UI_ASSET_DIRS 中的图片同步�?scripts/_external_import/，以�?Web 预览和标注�?

    说明：浏览器预览只能从项目目�?send_from_directory，所以这里采用“复制同步”策略�?
    """
    raw = os.environ.get('UI_ASSET_DIRS', '')
    dirs = [d.strip() for d in raw.split(';') if d.strip()]
    if not dirs:
        return 0
    scripts_dir = os.path.join(BASE_DIR, 'scripts')
    target_root = os.path.join(scripts_dir, '_external_import')
    os.makedirs(target_root, exist_ok=True)
    exts = {'.png', '.jpg', '.jpeg'}
    copied = 0
    for d in dirs:
        if not os.path.isdir(d):
            continue
        for root, _dirs, files in os.walk(d):
            for name in files:
                if os.path.splitext(name)[1].lower() not in exts:
                    continue
                src = os.path.join(root, name)
                dst = os.path.join(target_root, name)
                if os.path.exists(dst):
                    continue
                try:
                    shutil.copy2(src, dst)
                    copied += 1
                except Exception:
                    pass
    return copied


# ================= 任务状态机（结构化�?=================
# 保持向后兼容：依然会 emit('task_status', {'running': bool})
# 新增：emit('task_state', {'state':..., 'mode':..., 'message':..., 'ts':...})

TASK_STATE_LOCK = threading.Lock()
TASK_STATE = "IDLE"      # IDLE | INITING | RUNNING | STOPPING | FINISHED | ERROR
TASK_MODE = None         # init | capture | install | None
TASK_MESSAGE = ""


def _now_ts() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def set_task_state(state: str, *, mode: str = None, message: str = ""):
    """
    设置任务状态并推送给前端�?
    """
    global TASK_STATE, TASK_MODE, TASK_MESSAGE, is_running

    with TASK_STATE_LOCK:
        TASK_STATE = str(state or "IDLE").upper()
        if mode is not None:
            TASK_MODE = mode
        if message is not None:
            TASK_MESSAGE = message

        running = TASK_STATE in ("INITING", "RUNNING", "STOPPING")
        is_running = running

        payload = {
            "state": TASK_STATE,
            "mode": TASK_MODE,
            "message": TASK_MESSAGE,
            "ts": _now_ts()
        }

    # 兼容旧前端：仍然�?running 布尔�?
    try:
        socketio.emit('task_status', {'running': running})
    except Exception:
        pass

    # 新事件：结构化任务状�?
    try:
        socketio.emit('task_state', payload)
    except Exception:
        pass

    # 同时写一条日志（便于追踪�?
    if message:
        logger_callback(f"🧭 TASK_STATE={TASK_STATE} mode={TASK_MODE} | {message}")
    else:
        logger_callback(f"🧭 TASK_STATE={TASK_STATE} mode={TASK_MODE}")


# ================= 云真机工具函�?=================
def is_cdp_alive(port=CDP_PORT):
    try:
        s = socket.create_connection(("127.0.0.1", port), timeout=1)
        s.close()
        return True
    except Exception:
        return False


def launch_edge_cdp():
    print("🚀 自动启动 Edge（CDP 模式�?..")
    subprocess.Popen(
        [
            EDGE_PATH,
            f"--remote-debugging-port={CDP_PORT}",
            f"--user-data-dir={CDP_PROFILE}"
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )


if not os.path.exists(SCREENSHOT_ROOT_DIR):
    os.makedirs(SCREENSHOT_ROOT_DIR)
if not os.path.exists(os.path.dirname(META_DATA_FILE)):
    os.makedirs(os.path.dirname(META_DATA_FILE))

QUALITY_MAPPING = {
    "extreme": ["极致", "精致", "high", "ultra", "extreme", "精美"],
    "balanced": ["均衡", "标准", "middle", "balanced", "medium"],
    "power": ["省电", "流畅", "�?, "low", "power"]
}
QUALITY_DISPLAY = {"power": "省电模式", "balanced": "均衡模式", "extreme": "极致画质"}

def load_archive_index():
    """读取截图归档索引：date(YYYY-MM-DD) -> zip_name"""
    try:
        if os.path.exists(ARCHIVE_INDEX_FILE):
            with open(ARCHIVE_INDEX_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
    except Exception:
        pass
    return {}

def _zip_exists(zip_name: str) -> bool:
    if not zip_name:
        return False
    return os.path.exists(os.path.join(SCREENSHOT_ARCHIVE_ZIP_DIR, zip_name))



# ================= 云真机进度推�?=================
def emit_cloud_progress(step, total, name):
    """
    给前端实时推送云真机占用进度
    """
    msg = f"📡 云真机进度：{step}/{total} {name}"
    print(msg)
    try:
        socketio.emit("cloud_progress", {
            "step": int(step),
            "total": int(total) if total else 0,
            "name": name
        })
    except Exception as e:
        print(f"Socket emit cloud_progress error: {e}")


# ================= 核心逻辑�?=================
class ReportGenerator:
    def __init__(self, root_dir):
        self.root_dir = root_dir

    def _parse_mode(self, filename):
        filename_lower = filename.lower()
        for mode, keywords in QUALITY_MAPPING.items():
            for kw in keywords:
                if kw in filename_lower:
                    return mode
        return "balanced"

    def generate_report_for_folder(self, date_folder_name):
        date_path = os.path.join(self.root_dir, date_folder_name)
        if not os.path.exists(date_path):
            return None
        report_data = {
            "date": date_folder_name,
            "devices": {},
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        device_folders = [d for d in os.listdir(date_path) if os.path.isdir(os.path.join(date_path, d))]
        for device in device_folders:
            device_path = os.path.join(date_path, device)
            images = []
            all_files = sorted(os.listdir(device_path))
            originals = [f for f in all_files if f.lower().endswith(('.png', '.jpg', '.jpeg')) and '_thumb' not in f]
            for img_name in originals:
                thumb_name = img_name.replace(".png", "_thumb.jpg")
                has_thumb = thumb_name in all_files
                full_path = os.path.join(device_path, img_name)
                meta_path = full_path + ".meta.json"
                meta = None
                flags = []
                if os.path.exists(meta_path):
                    try:
                        with open(meta_path, "r", encoding="utf-8") as mf:
                            meta = json.load(mf)
                        flags = meta.get("flags") or []
                    except Exception:
                        meta = None
                        flags = []
                web_path_original = f"/files/{date_folder_name}/{device}/{img_name}"
                web_path_thumb = f"/files/{date_folder_name}/{device}/{thumb_name}" if has_thumb else web_path_original
                mode = self._parse_mode(img_name)
                images.append({
                    "filename": img_name,
                    "mode": mode,
                    "path": web_path_original,
                    "thumb_path": web_path_thumb,
                    "full_path": full_path,
                    "meta": meta,
                    "flags": flags
                })
            if images:
                grouped = {"power": [], "balanced": [], "extreme": []}
                for img in images:
                    grouped[img['mode']].append(img)
                report_data["devices"][device] = grouped
        if not report_data["devices"]:
            return None
        return report_data

    def scan_all_new(self, existing_dates):
        if not os.path.exists(self.root_dir):
            return []
        all_folders = [f for f in os.listdir(self.root_dir) if os.path.isdir(os.path.join(self.root_dir, f))]
        new_reports = []
        for folder in all_folders:
            if folder in existing_dates:
                continue
            print(f"🔎 发现新文件夹，正在生成报�? {folder}")
            report = self.generate_report_for_folder(folder)
            if report:
                new_reports.append(report)
        return new_reports


class ReportManager:
    def __init__(self):
        self.load()

    def load(self):
        if os.path.exists(META_DATA_FILE):
            try:
                with open(META_DATA_FILE, 'r', encoding='utf-8') as f:
                    self.data = json.load(f)
            except:
                self.data = {}
        else:
            self.data = {}

        # 自动同步归档状态（避免页面点进去后才发现目录被打包删除�?
        try:
            self.sync_archive_status()
        except Exception:
            pass

    def save(self):
        with open(META_DATA_FILE, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)


    def sync_archive_status(self):
        """根据磁盘�?archives_index.json 自动标记 archived 状态�?""
        idx = load_archive_index()
        changed = False

        for rid, report in (self.data or {}).items():
            date = report.get("date")
            if not date:
                continue

            date_dir = os.path.join(SCREENSHOT_ROOT_DIR, date)
            is_local = os.path.exists(date_dir)

            zip_name = idx.get(date)
            if (not is_local) and zip_name and _zip_exists(zip_name):
                if not report.get("archived") or report.get("archive_zip") != zip_name:
                    report["archived"] = True
                    report["archive_zip"] = zip_name
                    report["local_available"] = False
                    changed = True
            else:
                if is_local and report.get("archived"):
                    report["archived"] = False
                    report["archive_zip"] = ""
                    report["local_available"] = True
                    changed = True
                else:
                    report["local_available"] = bool(is_local)

        if changed:
            self.save()
    def add_report(self, report_data):
        rid = f"{report_data['date']}_{datetime.datetime.now().strftime('%H%M%S')}"
        self.data[rid] = report_data
        self.save()
        return rid

    def get_all(self):
        try:
            self.sync_archive_status()
        except Exception:
            pass
        return dict(sorted(self.data.items(), key=lambda x: x[1].get('date', ''), reverse=True))

    def get_existing_dates(self):
        dates = set()
        for rid, report in self.data.items():
            if 'date' in report:
                dates.add(report['date'])
        return dates

    def get(self, rid):
        return self.data.get(rid)

    def delete(self, rid):
        if rid in self.data:
            del self.data[rid]
            self.save()
            return True
        return False


generator = ReportGenerator(SCREENSHOT_ROOT_DIR)
manager = ReportManager()


# ================= 页面路由 =================
@app.route('/get_devices')
def get_devices():
    """获取设备列表"""
    try:
        output = subprocess.check_output(['adb', 'devices']).decode()
        devices = []
        for line in output.split('\n')[1:]:
            if 'device' in line and 'unauthorized' not in line:
                serial = line.split('\t')[0]
                model = airtest_service.get_device_model_name(serial)
                devices.append({'ip': serial, 'model': model, 'status': 'online', 'msg': '�?就绪'})
        return jsonify({"status": "success", "data": devices})
    except:
        return jsonify({"status": "error", "data": []})


@app.route('/connect', methods=['POST'])
def connect():
    data = request.json
    input_text = data.get('target', '')
    results = airtest_service.connect_devices_batch(input_text)
    return jsonify(results)


@app.route('/disconnect', methods=['POST'])
def disconnect():
    data = request.json
    ip = data.get('ip')
    success = airtest_service.disconnect_device(ip)
    return jsonify({"status": "success" if success else "error"})


@app.route('/stop_task', methods=['POST'])
def stop_task():
    airtest_service.stop_task()
    return jsonify({"status": "success"})


@app.route('/upload_apk', methods=['POST'])
def upload_apk():
    if 'file' not in request.files:
        return jsonify({"status": "error", "msg": "未找到文�?})
    file = request.files['file']
    if file.filename == '':
        return jsonify({"status": "error", "msg": "未选择文件"})

    if file and file.filename.endswith('.apk'):
        filename = secure_filename(file.filename)
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        try:
            file.save(save_path)
            print(f"📦 APK 上传成功: {save_path}")
            return jsonify({"status": "success", "apk_path": save_path})
        except Exception as e:
            return jsonify({"status": "error", "msg": str(e)})
    else:
        return jsonify({"status": "error", "msg": "请上�?.apk 格式文件"})


# ================= 自动化控�?API（日�?& 任务线程�?=================
task_logs = []
is_running = False
task_thread = None


def logger_callback(msg):
    global task_logs
    task_logs.append(msg)
    if len(task_logs) > 500:
        task_logs.pop(0)
    print(f"[Socket] Sending: {msg}")
    try:
        socketio.emit('new_log', {'data': msg})
    except Exception as e:
        print(f"Socket emit error: {e}")


def stop_thread(thread):
    if thread is None or not thread.is_alive():
        return
    tid = thread.ident
    exctype = SystemExit
    tid = ctypes.c_long(tid)
    if not isinstance(exctype, type):
        exctype = type(exctype)
    res = ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, ctypes.py_object(exctype))
    if res > 1:
        ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, None)


def _start_task_common(*, mode: str, socket_logger, custom_tasks=None, selected_devices=None, apk_path=None):
    global task_thread

    if is_running:
        socket_logger("⚠️ 已有任务在运行中，请稍后再试")
        return False

    set_task_state("INITING", mode=mode, message="任务初始�?)

    def task_wrapper():
        try:
            set_task_state("RUNNING", mode=mode, message="任务运行�?)
            airtest_service.run_airtest_task(
                socket_logger,
                mode,
                custom_tasks=custom_tasks or [],
                target_devices=selected_devices or [],
                apk_path=apk_path
            )
            set_task_state("FINISHED", mode=mode, message="任务正常结束")
        except SystemExit:
            set_task_state("FINISHED", mode=mode, message="🛑 任务已被强制中止（SystemExit�?)
        except Exception as e:
            set_task_state("ERROR", mode=mode, message=f"�?脚本异常崩溃: {e}")
        finally:
            time.sleep(0.1)
            set_task_state("IDLE", mode=None, message="空闲")

    task_thread = threading.Thread(target=task_wrapper, daemon=True)
    task_thread.start()
    return True


@socketio.on('start_task')
def handle_start_task(data):
    mode = data.get('mode')
    custom_tasks = data.get('custom_tasks', []) or []
    selected_devices = data.get('selected_devices', []) or []
    apk_path = data.get('apk_path')

    def socket_logger(msg: str):
        logger_callback(msg)

    _start_task_common(
        mode=mode,
        socket_logger=socket_logger,
        custom_tasks=custom_tasks,
        selected_devices=selected_devices,
        apk_path=apk_path
    )


# ================= 云真�?API =================
@app.route('/api/cloud/auto_connect_fav', methods=['POST'])
def cloud_auto_favorites():
    global GLOBAL_CLOUD_CONNECTOR

    if CloudAutoConnector is None:
        return jsonify({"success": False, "msg": "playwright_cloud 未加�?})

    if not is_cdp_alive():
        launch_edge_cdp()
        time.sleep(3)

    def on_progress(step, total, name):
        emit_cloud_progress(step, total, name)

    GLOBAL_CLOUD_CONNECTOR = CloudAutoConnector(
        CLOUD_USERNAME,
        CLOUD_PASSWORD,
        progress_cb=on_progress
    )

    max_devices = 5
    res = GLOBAL_CLOUD_CONNECTOR.auto_from_favorites(max_devices=max_devices)

    if not res.get("success"):
        return jsonify(res)

    adbs = res.get("adbs", [])

    results = []
    if adbs:
        targets = "\n".join(adbs)
        try:
            results = airtest_service.connect_devices_batch(targets) or []
        except Exception as e:
            print(f"�?auto connect adb failed: {e}")
            results = [{"ip": a, "model": "Unknown", "status": "error", "msg": "�?ADB 连接失败"} for a in adbs]

    return jsonify({
        "success": True,
        "adbs": adbs,
        "count": len(adbs),
        "results": results
    })


@app.route('/api/cloud/release', methods=['POST'])
def cloud_release():
    global GLOBAL_CLOUD_CONNECTOR

    if CloudAutoConnector is None:
        return jsonify({"success": False, "msg": "playwright_cloud 未加�?})

    if not is_cdp_alive():
        try:
            airtest_service.disconnect_all_devices()
        except Exception:
            pass
        try:
            subprocess.run(["adb", "disconnect"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            pass
        return jsonify({"success": False, "msg": "未检测到 Edge CDP�?222），已断开本地 ADB，请先启动云真机页面后再�?})

    connector = GLOBAL_CLOUD_CONNECTOR or CloudAutoConnector(
        CLOUD_USERNAME,
        CLOUD_PASSWORD,
        progress_cb=None
    )

    try:
        try:
            connector.recover_used_devices_from_disk()
        except Exception:
            pass

        if GLOBAL_CLOUD_CONNECTOR is not None:
            try:
                connector.used_devices = (GLOBAL_CLOUD_CONNECTOR.used_devices or []) + (connector.used_devices or [])
                connector.recover_used_devices_from_disk()
            except Exception:
                pass

        if connector.used_devices:
            connector.release_all_connected()
            release_mode = "精准释放（记�?磁盘恢复�?
        else:
            n = connector.release_all_mine_from_favorites()
            release_mode = f"兜底释放（收藏页自己在使用）{n} �?

        print("🧹 正在断开本地 ADB 连接...")
        try:
            airtest_service.disconnect_all_devices()
        except Exception:
            pass
        try:
            subprocess.run(["adb", "disconnect"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            pass

        GLOBAL_CLOUD_CONNECTOR = None
        return jsonify({"success": True, "msg": f"�?云真机释放完成：{release_mode}"})
    except Exception as e:
        return jsonify({"success": False, "msg": str(e)})


# ================= 页面渲染 =================
@app.route('/')
def index():
    history = manager.get_all()
    all_devices = set()
    for rid, data in history.items():
        devices = data.get('devices', {})
        all_devices.update(devices.keys())
    baseline_rid = get_baseline_rid()
    return render_template('index.html', history=history, all_devices=sorted(list(all_devices)), baseline_rid=baseline_rid)


@app.route('/ui_label')
def ui_label_page():
    """UI 语义标注台�?""
    return render_template('ui_label.html')


@app.route('/ui_explore')
def ui_explore_page():
    """自动探索 + Pending 标注（状态转移）�?""
    return render_template('ui_explore.html')


@app.route('/api/ui/list')
def api_ui_list():
    """列出所有可标注 UI 图片 + 已有 label�?

    - 优先读取 image_templates.json 中的 filename（这些是 Airtest 已登记的模板�?
    - 再补充扫�?scripts/ 目录下所�?png/jpg/jpeg（覆盖“新增候�?UI 候选裁剪”等�?
    - label 来源�?
        1) ui_knowledge_base.json �?filename->label
        2) 若不存在，则�?image_templates.json �?key 当作默认 label（例�?key=游戏加载完成�?
    """
    img_cfg = _read_json(IMAGE_TEMPLATES_FILE, default={})
    kb = _read_json(UI_KB_FILE, default={})

    # 可选：同步外部模板目录（解决“图片太多无法上传”的场景�?
    _sync_external_ui_assets()

    # filename -> [template_key,...]
    f2k = {}
    for k, v in img_cfg.items():
        fn = str(v.get('filename', '')).strip() if isinstance(v, dict) else ''
        if fn:
            f2k.setdefault(fn, []).append(k)

    # 收集 scripts 下的图片（用相对路径，保证子目录�?ui_candidates/ 可被正确识别�?
    scripts_dir = os.path.join(BASE_DIR, 'scripts')
    exts = {'.png', '.jpg', '.jpeg'}
    found_files = set(f2k.keys())
    if os.path.isdir(scripts_dir):
        for root, _dirs, files in os.walk(scripts_dir):
            for name in files:
                if os.path.splitext(name)[1].lower() in exts:
                    rel = os.path.relpath(os.path.join(root, name), scripts_dir)
                    rel = rel.replace('\\', '/')
                    found_files.add(rel)

    items = []
    for fn in sorted(found_files):
        meta = kb.get(fn, {}) if isinstance(kb, dict) else {}
        label = str(meta.get('label', '')).strip()
        # 没标注时，用 image_templates.json �?key 当默�?label（如果该文件只有一�?key 且看起来是中�?语义名）
        if not label:
            keys = f2k.get(fn, [])
            if len(keys) == 1:
                label = str(keys[0])
        items.append({
            'filename': fn,
            'template_keys': f2k.get(fn, []),
            'label': label,
            'tags': meta.get('tags', []) if isinstance(meta, dict) else [],
            'notes': str(meta.get('notes', '')).strip() if isinstance(meta, dict) else '',
        })
    return jsonify({'success': True, 'items': items})


@app.route('/api/ui/save_label', methods=['POST'])
def api_ui_save_label():
    """保存某个 filename 的标注。可选：将该模板“入库”到 image_templates.json�?""
    payload = request.get_json(force=True, silent=True) or {}
    filename = str(payload.get('filename', '')).strip()
    label = str(payload.get('label', '')).strip()
    tags = payload.get('tags', [])
    notes = str(payload.get('notes', '')).strip()
    register = bool(payload.get('register', False))

    if not filename:
        return jsonify({'success': False, 'msg': 'filename required'}), 400

    kb = _read_json(UI_KB_FILE, default={})
    if not isinstance(kb, dict):
        kb = {}
    kb[filename] = {
        'label': label,
        'tags': tags if isinstance(tags, list) else [],
        'notes': notes,
    }
    _write_json(UI_KB_FILE, kb)

    if register and label:
        # 把该 filename 最小化写入 image_templates.json（不覆盖已有同名 key�?
        airtest_service.ensure_template_registered(filename, template_key=label)

    return jsonify({'success': True})


@app.route('/api/ui/extract_candidates', methods=['POST'])
def api_ui_extract_candidates():
    """从指定设备截图并提取 UI 候选图标�?""
    payload = request.get_json(force=True, silent=True) or {}
    serial = str(payload.get('serial', '')).strip()
    if not serial:
        return jsonify({'success': False, 'msg': 'serial required'}), 400
    max_items = int(payload.get('max_items', 120))
    res = airtest_service.capture_and_extract_ui_candidates(serial, max_items=max_items)
    return jsonify({'success': True, 'result': res})


@app.route('/api/ui/explore_step', methods=['POST'])
def api_ui_explore_step():
    payload = request.get_json(force=True, silent=True) or {}
    serial = str(payload.get('serial', '')).strip()
    from_state = str(payload.get('from_state', '')).strip() or None
    max_candidates = int(payload.get('max_candidates', 60))
    settle_wait = float(payload.get('settle_wait', 1.0))
    diff_threshold = float(payload.get('diff_threshold', 2.2))
    if not serial:
        return jsonify({'success': False, 'msg': 'serial required'}), 400
    rec = airtest_service.explore_step(
        serial,
        from_state=from_state,
        max_candidates=max_candidates,
        settle_wait=settle_wait,
        diff_threshold=diff_threshold,
    )
    return jsonify({'success': True, 'record': rec})


@app.route('/api/ui/explore_all', methods=['POST'])
def api_ui_explore_all():
    payload = request.get_json(force=True, silent=True) or {}
    serial = str(payload.get('serial', '')).strip()
    from_state = str(payload.get('from_state', '')).strip() or None
    max_candidates = int(payload.get('max_candidates', 180))
    settle_wait = float(payload.get('settle_wait', 1.0))
    diff_threshold = float(payload.get('diff_threshold', 2.2))
    max_steps = int(payload.get('max_steps', 120))

    if not serial:
        return jsonify({'success': False, 'msg': 'serial required'}), 400

    # Prevent accidental multi-click / double request
    with UI_EXPLORE_LOCK:
        if serial in UI_EXPLORE_INFLIGHT:
            return jsonify({'success': False, 'msg': f'explore_all already running for {serial}'}), 429
        UI_EXPLORE_INFLIGHT.add(serial)

    try:
        res = airtest_service.explore_all(
            serial,
            from_state=from_state,
            max_candidates=max_candidates,
            settle_wait=settle_wait,
            diff_threshold=diff_threshold,
            max_steps=max_steps,
        )
        # normalize
        ok = bool(res.get('ok', True)) if isinstance(res, dict) else True
        if not ok:
            return jsonify({'success': False, 'msg': res.get('msg', 'explore_all failed'), 'result': res})
        return jsonify({'success': True, 'result': res})
    finally:
        with UI_EXPLORE_LOCK:
            UI_EXPLORE_INFLIGHT.discard(serial)


@app.route('/api/ui/pending_list')
def api_ui_pending_list():
    arr = airtest_service.load_ui_pending()
    # 前端按未提交优先
    arr = [r for r in arr if isinstance(r, dict)]
    arr.sort(key=lambda r: (0 if r.get('status') == 'pending' else 1, -int(r.get('ts', 0))))
    return jsonify({'success': True, 'items': arr})


@app.route('/api/ui/pending_commit', methods=['POST'])
def api_ui_pending_commit():
    payload = request.get_json(force=True, silent=True) or {}
    pending_id = str(payload.get('pending_id', '')).strip()
    ui_label = str(payload.get('ui_label', '')).strip()
    to_state = str(payload.get('to_state', '')).strip()
    anchors = payload.get('anchors', [])
    if isinstance(anchors, str):
        anchors = [a.strip() for a in anchors.split(',') if a.strip()]
    register = bool(payload.get('register', False))
    rollback = payload.get('rollback', None)

    res = airtest_service.commit_pending_transition(
        pending_id,
        ui_label=ui_label,
        to_state=to_state,
        anchors=anchors,
        register=register,
        rollback=rollback,
    )
    if not res.get('ok'):
        return jsonify({'success': False, 'msg': res.get('msg', 'commit failed')}), 400
    return jsonify({'success': True, 'record': res.get('record')})


@app.route('/api/ui/click', methods=['POST'])
def api_ui_click():
    """执行层语义点击：label -> 模板匹配 -> touch�?
    说明：这是“标注接入执行层”的最小闭环接口�?
    """
    payload = request.get_json(force=True, silent=True) or {}
    serial = str(payload.get('serial', '')).strip()
    label = str(payload.get('label', '')).strip()
    if not serial or not label:
        return jsonify({'success': False, 'msg': 'serial and label required'}), 400

    # 连接设备（短连接：用于交互测试）
    from airtest.core.api import connect_device
    dev = None
    try:
        dev = connect_device(f"android:///{serial}?cap_method=ADBCAP&touch_method=ADBTOUCH")
        model = airtest_service.get_device_model_name(serial)
        coords = airtest_service.get_device_coords_by_model(model)
        ok = airtest_service.ui_click_semantic(dev, label, coords, retries=2)
        return jsonify({'success': True, 'clicked': bool(ok)})
    except Exception as e:
        return jsonify({'success': False, 'msg': str(e)}), 500
    finally:
        if dev:
            try:
                dev.stop_app(airtest_service.PACKAGE_NAME)
            except Exception:
                pass


@app.route('/scan', methods=['POST'])
def scan_and_save():
    existing_dates = manager.get_existing_dates()
    new_reports = generator.scan_all_new(existing_dates)
    count = 0
    for report in new_reports:
        manager.add_report(report)
        count += 1
    return jsonify({"success": True, "message": f"成功添加 {count} 个新报告" if count > 0 else "没有发现新截图文件夹"})




@app.route('/download_archive/<path:zip_name>')
def download_archive(zip_name):
    # 安全：只允许下载 .zip
    if not zip_name or (not zip_name.lower().endswith('.zip')):
        return jsonify({"success": False, "msg": "invalid zip"}), 400
    zpath = os.path.join(SCREENSHOT_ARCHIVE_ZIP_DIR, zip_name)
    if not os.path.exists(zpath):
        return jsonify({"success": False, "msg": "zip not found"}), 404
    return send_from_directory(SCREENSHOT_ARCHIVE_ZIP_DIR, zip_name, as_attachment=True)

@app.route('/view/<rid>')
def view_report(rid):
    report = manager.get(rid)
    if not report:
        return "记录不存�?, 404
    # 若该日期已被归档（本地目录不存在），则仅提示�?zip 解压查看
    if report.get("archived") and (not report.get("local_available", True)):
        report_min = {
            "date": report.get("date"),
            "timestamp": report.get("timestamp"),
            "archived": True,
            "archive_zip": report.get("archive_zip"),
            "devices": {}
        }
        return render_template('view_report.html', report=report_min, display=QUALITY_DISPLAY, rid=rid, baseline_rid=get_baseline_rid())

    return render_template('view_report.html', report=report, display=QUALITY_DISPLAY, rid=rid, baseline_rid=get_baseline_rid())


@app.route('/compare_page')
def compare_page():
    history = manager.get_all()
    baseline_rid = get_baseline_rid()
    return render_template('compare.html', history=history, display=QUALITY_DISPLAY, baseline_rid=baseline_rid)


@app.route('/api/template/get')
def api_get_template():
    """获取当前基准模板（report rid）�?""
    baseline_rid = get_baseline_rid()
    return jsonify({"success": True, "baseline_rid": baseline_rid})


@app.route('/api/template/set', methods=['POST'])
def api_set_template():
    """设置某个报告为基准模板（report rid）�?""
    data = request.get_json(silent=True) or {}
    rid = (data.get('rid') or '').strip()
    if not rid:
        return jsonify({"success": False, "msg": "rid is required"}), 400
    if not manager.get(rid):
        return jsonify({"success": False, "msg": "report not found"}), 404
    payload = {
        "baseline_rid": rid,
        "updated_at": _now_ts(),
    }
    _save_compare_template(payload)
    return jsonify({"success": True, "baseline_rid": rid})


@app.route('/api/report_data/<rid>')
def get_report_json(rid):
    report = manager.get(rid)
    if report:
        return jsonify({"success": True, "data": report})
    return jsonify({"success": False, "message": "Not Found"})


@app.route('/delete_report/<rid>', methods=['POST'])
def delete_report(rid):
    delete_local = False
    try:
        data = request.get_json(silent=True) or {}
        delete_local = bool(data.get('delete_local', False))
    except Exception:
        delete_local = False

    report = manager.get(rid)
    if delete_local and report:
        folder_name = report.get('date')
        if folder_name:
            folder_path = os.path.join(SCREENSHOT_ROOT_DIR, folder_name)
            if os.path.exists(folder_path):
                try:
                    shutil.rmtree(folder_path)
                except Exception as e:
                    print(f"�?删除文件夹失�? {e}")

    manager.delete(rid)
    return jsonify({"success": True})


@app.route('/files/<path:filename>')
def serve_files(filename):
    return send_from_directory(SCREENSHOT_ROOT_DIR, filename)


@app.route('/tpl/<path:filename>')
def serve_templates(filename):
    """提供 scripts/ 下的模板图片�?UI 标注台预览�?""
    scripts_dir = os.path.join(BASE_DIR, 'scripts')
    # send_from_directory 自带路径安全校验（防�?.. 逃逸）
    return send_from_directory(scripts_dir, filename)


# ================= HTTP：启动任务（兼容原接口） =================
@app.route('/api/run_test', methods=['POST'])
def run_test_api():
    data = request.json or {}
    custom_tasks = data.get('tasks', [])
    target_devices = data.get('target_devices', [])

    if not custom_tasks:
        return jsonify({"success": False, "message": "任务列表为空"})

    def http_logger(msg: str):
        logger_callback(msg)

    ok = _start_task_common(
        mode="capture",
        socket_logger=http_logger,
        custom_tasks=custom_tasks,
        selected_devices=target_devices,
        apk_path=None
    )
    if not ok:
        return jsonify({"success": False, "message": "任务正在运行�?})
    return jsonify({"success": True, "message": "截图任务已启�?})


@app.route('/api/run_login_only', methods=['POST'])
def run_login_only_api():
    data = request.json or {}
    target_devices = data.get('target_devices', [])

    def http_logger(msg: str):
        logger_callback(msg)

    ok = _start_task_common(
        mode="init",
        socket_logger=http_logger,
        custom_tasks=[],
        selected_devices=target_devices,
        apk_path=None
    )
    if not ok:
        return jsonify({"success": False, "message": "任务正在运行�?})
    return jsonify({"success": True, "message": "新包登录流程已启�?})


@app.route('/api/stop_test', methods=['POST'])
def stop_test_api():
    global task_thread

    if not is_running or task_thread is None:
        set_task_state("IDLE", mode=None, message="当前无运行任�?)
        return jsonify({"success": True, "message": "当前无运行任�?})

    set_task_state("STOPPING", mode=TASK_MODE, message="收到停止指令，进入优雅停止阶�?)

    airtest_service.stop_task()
    logger_callback("⚠️ 已发送停止指令：STOP_FLAG=1 + ADB 强停游戏（优雅停止阶段）")

    grace_s = 4.0
    start = time.time()
    while is_running and task_thread and task_thread.is_alive() and (time.time() - start) < grace_s:
        time.sleep(0.2)

    if task_thread and (not task_thread.is_alive()):
        set_task_state("FINISHED", mode=TASK_MODE, message="任务已优雅停止（线程已退出）")
        time.sleep(0.1)
        set_task_state("IDLE", mode=None, message="空闲")
        return jsonify({"success": True, "message": "已停�?})

    if is_running and task_thread and task_thread.is_alive():
        logger_callback("🛑 宽限期内线程未退出，执行强制中止（兜底）")
        stop_thread(task_thread)
        time.sleep(0.5)
        set_task_state("FINISHED", mode=TASK_MODE, message="已强制中止（兜底�?)
        logger_callback("⚠️ 已强制中止：如后续出�?OCR/ADB 异常，建议重启服务以清理残留状�?)
        time.sleep(0.1)
        set_task_state("IDLE", mode=None, message="空闲")

    return jsonify({"success": True, "message": "中止指令已发�?})


@app.route('/api/logs')
def get_logs():
    with TASK_STATE_LOCK:
        state_payload = {
            "state": TASK_STATE,
            "mode": TASK_MODE,
            "message": TASK_MESSAGE,
            "ts": _now_ts()
        }
    return jsonify({"running": is_running, "logs": task_logs, "task_state": state_payload})


# �?新增：更轻量的任务状态查询（前端重连/断线可用；不影响原接口）
@app.route('/api/task_state')
def api_task_state():
    with TASK_STATE_LOCK:
        state_payload = {
            "state": TASK_STATE,
            "mode": TASK_MODE,
            "message": TASK_MESSAGE,
            "ts": _now_ts()
        }
    return jsonify({"success": True, "running": is_running, "task_state": state_payload})


@app.route('/api/connect_devices', methods=['POST'])
def api_connect_devices():
    data = request.json
    input_text = data.get('targets', '')
    if not input_text.strip():
        return jsonify({"success": False, "message": "输入内容为空"})
    results = airtest_service.connect_devices_batch(input_text)
    return jsonify({"success": True, "results": results})


@app.route('/api/update_device_cap', methods=['POST'])
def api_update_device_cap():
    data = request.json
    model = data.get('model')
    quality = data.get('quality')
    if model and quality:
        airtest_service.save_device_quality(model, quality)
        return jsonify({"success": True})
    return jsonify({"success": False})


@app.route('/api/disconnect_device', methods=['POST'])
def api_disconnect_device():
    data = request.json
    ip = data.get('ip')
    if ip:
        airtest_service.disconnect_device(ip)
        return jsonify({"success": True})
    return jsonify({"success": False})


@app.route('/api/disconnect_all', methods=['POST'])
def api_disconnect_all():
    airtest_service.disconnect_all_devices()
    return jsonify({"success": True, "message": "已断开所有设备连�?})


if __name__ == '__main__':
    import socket as _socket
    try:
        s = _socket.socket(_socket.AF_INET, _socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        host_ip = s.getsockname()[0]
        s.close()
    except:
        host_ip = "127.0.0.1"

    print("=" * 50)
    print(f"🚀 服务已启�?(WebSocket Mode)")
    print(f"📂 截图路径: {SCREENSHOT_ROOT_DIR}")
    print(f"🏠 本机访问: http://localhost:5000")
    print(f"📡 局域网访问: http://{host_ip}:5000")
    print("=" * 50)

    set_task_state("IDLE", mode=None, message="服务启动完成，空�?)
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)