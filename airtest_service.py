# -*- coding: utf-8 -*-
import os
import re
import time
import json
import datetime

# UI explore (bulk)
import ui_explore
import logging
import threading
import traceback
import subprocess
import zipfile
import shutil
import math
import functools
from concurrent.futures import ThreadPoolExecutor


from collections import deque
import numpy as np
from PIL import Image

from airtest.core.api import *
from airtest.core.error import *
from airtest.core.settings import Settings as ST

# ================= 0. 自定义异常与停止检�?=================

class TaskStoppedError(Exception):
    """当任务被强制停止时抛出的专用异常，用于打断所有流�?""
    pass

def check_stop():
    """全局检查停止标志，若停止则抛出异常，打断当前堆�?""
    if STOP_FLAG:
        raise TaskStoppedError("任务已强制停�?)

# ================= 1. OCR 初始�?(含兼容性补�? =================

# OCR 级别说明�?
# 0 = 关闭 OCR（默认，强制“模�?坐标优先”，提升多设备并发吞吐）
# 1 = 仅在“网络重�?安装器兜�?诊断”场景允�?OCR（建议值）
# 2 = 全量 OCR（旧行为，吞吐最低）
OCR_LEVEL = int(os.environ.get("AIRTEST_OCR_LEVEL", "0"))

OCR_AVAILABLE = False
OCR_MODEL = None
OCR_LOCK = threading.Lock()

# 仅在需�?OCR 时才初始化（避免多设备并发时 OCR 成为全局串行瓶颈�?
if OCR_LEVEL <= 0:
    print("ℹ️ [System] OCR 已禁用（AIRTEST_OCR_LEVEL=0），仅使用模板匹�?坐标")
else:
    try:
        # 1. 屏蔽日志
        logging.getLogger("ppocr").setLevel(logging.WARNING)

        import paddle

        # 定义一个空函数，用来过 OCR 库，解决 disable_mkldnn / set_optimization_level 缺失问题
        def mock_method(*args, **kwargs):
            return None

        # 1. 针对 Paddle 新版接口
        if hasattr(paddle, "inference") and hasattr(paddle.inference, "Config"):
            cfg = paddle.inference.Config
            if not hasattr(cfg, "disable_mkldnn"):
                cfg.disable_mkldnn = mock_method
            if not hasattr(cfg, "set_optimization_level"):
                cfg.set_optimization_level = mock_method

        # 2. 针对 Paddle 旧版接口 (AnalysisConfig)
        if hasattr(paddle, "fluid") and hasattr(paddle.fluid, "libpaddle"):
            try:
                AnalysisConfig = paddle.fluid.libpaddle.AnalysisConfig
                if not hasattr(AnalysisConfig, "disable_mkldnn"):
                    AnalysisConfig.disable_mkldnn = mock_method
                if not hasattr(AnalysisConfig, "set_optimization_level"):
                    AnalysisConfig.set_optimization_level = mock_method
            except Exception:
                pass

        from paddleocr import PaddleOCR

        # 初始�?(�?paddle 自动检�?GPU)
        # 注意：不要开�?show_log；多设备并发�?OCR 仅作为兜底使�?
        OCR_MODEL = PaddleOCR(use_angle_cls=False, lang="ch")
        OCR_AVAILABLE = True

        try:
            device = paddle.device.get_device()
            print(f"�?[System] PaddleOCR 加载成功 | 当前运行设备: {device} | OCR_LEVEL={OCR_LEVEL}")
        except Exception:
            print(f"�?[System] PaddleOCR 加载成功 | OCR_LEVEL={OCR_LEVEL}")

    except ImportError:
        print("⚠️ [System] 未检测到 paddleocr 库，OCR 功能将禁�?)
    except Exception as e:
        print(f"⚠️ [System] OCR 初始化失�? {e}")

# OCR 降级策略：在 OCR_LEVEL=1 时，仅允许处理“重�?网络异常/安装器”等少量关键场景
OCR_FALLBACK_TAGS = {
    "前往征战", "点击选服", "开始游�?, "进入游戏",
    "重新连接", "重试", "网络异常", "连接失败", "登录失败", "请重�?, "重新登录", "返回登录",
    # 安装器相关（如你不跑装包流程，也可以忽略�?
    "安装", "继续安装", "立即安装", "开始安�?, "同意并安�?, "仍要安装", "继续", "下一�?,
    "允许", "允许安装", "设置", "去设�?, "打开", "完成"
}
# 入口按钮专用 OCR 放行（仅用于“入口阶段”的 click�?
ENTRY_OCR_TAGS = {"前往征战", "点击选服", "开始游�?, "进入游戏"}

def ocr_allowed(tag: str) -> bool:
    if not tag:
        return False
    if OCR_LEVEL >= 2:
        return True
    if OCR_LEVEL == 1:
        return str(tag) in OCR_FALLBACK_TAGS
    return False

def ocr_allowed_entry(tag: str) -> bool:
    if not tag:
        return False
    # 仍然尊重全量模式
    if OCR_LEVEL >= 2:
        return True
    # 入口阶段：只允许入口关键�?
    return str(tag) in ENTRY_OCR_TAGS
# ================= 2. 基础配置 =================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR = os.path.join(BASE_DIR, "scripts")
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "screenshots")
ARCHIVE_DIR = os.path.join(BASE_DIR, "screenshots_archive")
CONFIG_FILE = os.path.join(DATA_DIR, "device_caps.json")

if not os.path.exists(TEMPLATE_DIR): os.makedirs(TEMPLATE_DIR)
if not os.path.exists(DATA_DIR): os.makedirs(DATA_DIR)
if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
if not os.path.exists(ARCHIVE_DIR): os.makedirs(ARCHIVE_DIR)

ST.PROJECT_ROOT = TEMPLATE_DIR
ST.OPDELAY = 0.5
ST.FIND_TIMEOUT = 10
ST.FIND_THRESHOLD = 0.7

STOP_FLAG = False
CURRENT_LOGGER = None
CONFIG_LOCK = threading.Lock()
MAX_RETRIES = 3

PACKAGE_NAME = "com.example.game"

# ================= 3. 工具函数 =================

# 全局变量定义（防�?Pylance 报错�?
_OCR_CHECK_COUNTER = 0
_BAD_SCREEN_CACHE = {}

def _crop_region(image, region=None):
    """辅助函数：裁剪图片，用于加�?OCR"""
    if region is None:
        return image
    try:
        if not hasattr(image, 'crop'):
            return image
        return image.crop(region)
    except Exception:
        return image

def log_proxy(msg: str, logger=None):
    target = logger if logger else CURRENT_LOGGER
    timestr = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{timestr}] {msg}")
    if target:
        try:
            target(msg)
        except Exception:
            pass

def simple_wait(sec: float):
    """可中断的 sleep"""
    check_stop()
    end_t = time.time() + sec
    while time.time() < end_t:
        check_stop()
        time.sleep(0.1)

def retry_action(retries=3, delay=1.0, exception_type=Exception):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            check_stop()
            last_err = None
            for i in range(retries):
                try:
                    check_stop()
                    return func(*args, **kwargs)
                except TaskStoppedError:
                    raise
                except exception_type as e:
                    last_err = e
                    if i < retries - 1:
                        time.sleep(delay)
            raise last_err
        return wrapper
    return decorator

def save_error_snapshot(dev, device_alias: str, step_name: str):
    try:
        if STOP_FLAG: return
        timestamp = int(time.time())
        save_dir = os.path.join(BASE_DIR, "screenshots_error")
        os.makedirs(save_dir, exist_ok=True)
        filename = f"error_{device_alias}_{step_name}_{timestamp}.jpg"
        filepath = os.path.join(save_dir, filename)
        dev.snapshot(filename=filepath)
        log_proxy(f"[{device_alias}] 📸 异常截图已保�? {filename}")
    except Exception:
        pass

def diagnose_screen(dev, device_alias: str, step_context: str):
    """诊断模式：保存截图并 OCR"""
    try:
        if STOP_FLAG: return
        target_dir = os.path.join(ARCHIVE_DIR, step_context)
        os.makedirs(target_dir, exist_ok=True)

        existing_files = [f for f in os.listdir(target_dir) if f.lower().endswith(('.png', '.jpg'))]
        should_save = len(existing_files) < 10

        timestamp = int(time.time())
        filename = f"{device_alias}_{timestamp}.jpg"
        filepath = os.path.join(target_dir, filename)

        if should_save:
            dev.snapshot(filename=filepath)
            log_proxy(f"[{device_alias}] 📂 [诊断存档] 已保存至: {step_context}/{filename}")
        else:
            filepath = "temp_diagnose.jpg"
            dev.snapshot(filename=filepath)

        if OCR_AVAILABLE:
            with OCR_LOCK:
                if os.path.exists(filepath):
                    res = OCR_MODEL.ocr(filepath)
                    texts = []
                    if res and res[0]:
                        for line in res[0]:
                            if line and line[1]:
                                texts.append(line[1][0])
                    log_text = " | ".join(texts[:15])
                    log_proxy(f"[{device_alias}] 🔍 [OCR诊断] 当前屏幕包含: {log_text}")
    except Exception as e:
        log_proxy(f"[{device_alias}] ⚠️ 诊断过程异常: {e}")

@retry_action(retries=2, delay=0.5)
def safe_snapshot(dev, filepath: str):
    check_stop()
    dev.snapshot(filename=filepath)
    try:
        if os.path.exists(filepath):
            thumb_path = filepath.replace(".png", "_thumb.jpg")
            with Image.open(filepath) as img:
                if img.mode in ("RGBA", "P"):
                    img = img.convert("RGB")
                img.thumbnail((300, 300))
                img.save(thumb_path, "JPEG", quality=70)
    except Exception:
        pass
# ================= 3.5 截图质量检�?& 归档清理（保守策略） =================

ARCHIVE_ZIP_DIR = os.path.join(ARCHIVE_DIR, "zips")
ARCHIVE_INDEX_FILE = os.path.join(ARCHIVE_ZIP_DIR, "archives_index.json")

os.makedirs(ARCHIVE_ZIP_DIR, exist_ok=True)

def _parse_date_folder(name: str):
    try:
        return datetime.datetime.strptime(name, "%Y-%m-%d").date()
    except Exception:
        return None

def _fmt_md(d: datetime.date) -> str:
    # 11.2 这种格式（不补零），更方便肉眼定�?
    return f"{d.month}.{d.day}"

def _load_archive_index() -> dict:
    try:
        if os.path.exists(ARCHIVE_INDEX_FILE):
            with open(ARCHIVE_INDEX_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
    except Exception:
        pass
    return {}

def _save_archive_index(idx: dict) -> None:
    try:
        os.makedirs(os.path.dirname(ARCHIVE_INDEX_FILE), exist_ok=True)
        with open(ARCHIVE_INDEX_FILE, "w", encoding="utf-8") as f:
            json.dump(idx, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

def archive_old_screenshots(keep_days: int = 14, logger=None) -> dict:
    '''
    归档 screenshots/ 下超�?keep_days 的日期目录：
      - 旧目录打包到 screenshots_archive/zips/ �?
      - zip 命名：M.D-M.D（如 11.2-12.5�?
      - 写入 archives_index.json：date(YYYY-MM-DD)->zip_name
      - 打包成功后删除原目录

    返回归档摘要：{"success":bool,"zip":str|None,"start":str|None,"end":str|None,"count":int,"kept_days":int}
    '''
    try:
        today = datetime.date.today()
        cutoff = today - datetime.timedelta(days=int(keep_days))

        if not os.path.exists(OUTPUT_DIR):
            return {"success": True, "zip": None, "start": None, "end": None, "count": 0, "kept_days": keep_days}

        candidates = []
        for name in os.listdir(OUTPUT_DIR):
            p = os.path.join(OUTPUT_DIR, name)
            if not os.path.isdir(p):
                continue
            d = _parse_date_folder(name)
            if not d:
                continue
            # 只归档严格早�?cutoff 的日期（保留 keep_days 天）
            if d < cutoff:
                candidates.append((d, name, p))

        candidates.sort(key=lambda x: x[0])
        if not candidates:
            return {"success": True, "zip": None, "start": None, "end": None, "count": 0, "kept_days": keep_days}

        start_d, start_name, _ = candidates[0]
        end_d, end_name, _ = candidates[-1]

        zip_name = f"{_fmt_md(start_d)}-{_fmt_md(end_d)}.zip"
        zip_path = os.path.join(ARCHIVE_ZIP_DIR, zip_name)

        # 避免覆盖：若同名已存在，则追加序�?
        if os.path.exists(zip_path):
            base = zip_name[:-4]
            k = 2
            while True:
                alt = f"{base}_{k}.zip"
                alt_path = os.path.join(ARCHIVE_ZIP_DIR, alt)
                if not os.path.exists(alt_path):
                    zip_name, zip_path = alt, alt_path
                    break
                k += 1

        if logger:
            logger(f"🗜�?归档旧截图：{start_name} ~ {end_name} -> {zip_name} (�?{len(candidates)} �?")
        else:
            log_proxy(f"🗜�?归档旧截图：{start_name} ~ {end_name} -> {zip_name} (�?{len(candidates)} �?")

        idx = _load_archive_index()

        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
            for d, folder_name, folder_path in candidates:
                # 以日期目录为顶层，保持原结构：YYYY-MM-DD/device/xxx.png
                for root, _dirs, files in os.walk(folder_path):
                    for fn in files:
                        src = os.path.join(root, fn)
                        rel = os.path.relpath(src, OUTPUT_DIR)  # 形如 YYYY-MM-DD/Device/xxx.png
                        zf.write(src, arcname=rel)
                idx[folder_name] = zip_name

        _save_archive_index(idx)

        # 删除已归档目�?
        for _d, folder_name, folder_path in candidates:
            try:
                shutil.rmtree(folder_path, ignore_errors=True)
            except Exception:
                pass

        return {
            "success": True,
            "zip": zip_name,
            "start": start_name,
            "end": end_name,
            "count": len(candidates),
            "kept_days": keep_days
        }
    except Exception as e:
        if logger:
            logger(f"⚠️ 归档失败: {e}")
        else:
            log_proxy(f"⚠️ 归档失败: {e}")
        return {"success": False, "zip": None, "start": None, "end": None, "count": 0, "kept_days": keep_days, "error": str(e)}

def analyze_image_quality(filepath: str) -> dict:
    '''
    保守型截图异常检测（避免“飞�?小差异”误判）�?
      - 黑屏/近黑�?
      - 大面积黑�?
      - 明显色偏（单通道强占比）
      - 低对�?疑似花屏（仅标记�?suspect，不用于自动清理�?

    返回 {"flags":[...], "metrics":{...}}
    '''
    out = {"flags": [], "metrics": {}}
    try:
        if not filepath or (not os.path.exists(filepath)):
            out["flags"].append("missing_file")
            return out

        with Image.open(filepath) as im:
            if im.mode not in ("RGB", "RGBA"):
                im = im.convert("RGB")
            if im.mode == "RGBA":
                im = im.convert("RGB")
            # 降采样加�?
            im_small = im.resize((max(64, im.size[0]//8), max(64, im.size[1]//8)))
            arr = np.asarray(im_small).astype(np.float32)
            if arr.ndim != 3 or arr.shape[2] < 3:
                return out

        r, g, b = arr[...,0], arr[...,1], arr[...,2]
        luma = 0.2126*r + 0.7152*g + 0.0722*b

        mean_luma = float(np.mean(luma))
        std_luma  = float(np.std(luma))
        black_ratio = float(np.mean((r < 20) & (g < 20) & (b < 20)))
        dark_ratio  = float(np.mean(luma < 15))
        gray_ratio = float(np.mean((np.abs(r-g) < 8) & (np.abs(g-b) < 8)))

        mr, mg, mb = float(np.mean(r)), float(np.mean(g)), float(np.mean(b))
        ch_max = max(mr, mg, mb)
        ch_min = min(mr, mg, mb)
        cast_ratio = float((ch_max + 1.0) / (ch_min + 1.0))

        mx = np.maximum(np.maximum(r, g), b)
        mn = np.minimum(np.minimum(r, g), b)
        sat = np.where(mx <= 0.0, 0.0, (mx - mn) / (mx + 1e-6))
        sat_mean = float(np.mean(sat))

        out["metrics"] = {
            "mean_luma": round(mean_luma, 3),
            "std_luma": round(std_luma, 3),
            "black_ratio": round(black_ratio, 4),
            "dark_ratio": round(dark_ratio, 4),
            "gray_ratio": round(gray_ratio, 4),
            "mean_r": round(mr, 3),
            "mean_g": round(mg, 3),
            "mean_b": round(mb, 3),
            "cast_ratio": round(cast_ratio, 3),
            "sat_mean": round(sat_mean, 4),
        }

        if mean_luma < 18 or dark_ratio > 0.90 or black_ratio > 0.92:
            out["flags"].append("black_screen")

        if black_ratio > 0.28 and mean_luma > 25:
            out["flags"].append("large_black_block")

        if mean_luma > 30 and cast_ratio > 1.7 and sat_mean > 0.18:
            out["flags"].append("color_cast")

        if mean_luma > 35 and std_luma > 35 and sat_mean > 0.45 and gray_ratio < 0.25:
            out["flags"].append("suspect_noise_or_glitch")

        if mean_luma > 25 and std_luma < 8 and gray_ratio > 0.75:
            out["flags"].append("low_contrast_or_overlay")

    except Exception as e:
        out["flags"].append("analyze_error")
        out["metrics"]["error"] = str(e)

    return out

def write_image_meta(image_path: str, meta: dict) -> str:
    '''
    将分析结果写入旁路文件：<image>.meta.json
    '''
    try:
        meta_path = image_path + ".meta.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta or {}, f, ensure_ascii=False, indent=2)
        return meta_path
    except Exception:
        return ""


# ================= 4. 账号�?=================

class AccountPool:
    """线程安全账号池：同一轮任务内一个账号只能被占用一次，释放后才回到池中�?""

    def __init__(self, filename="accounts.txt"):
        self.filename = os.path.join(BASE_DIR, filename)
        self._lock = threading.Lock()
        self._cv = threading.Condition(self._lock)

        self.accounts = self._load_accounts()
        self.available = deque(self.accounts)   # 可用队列（FIFO�?
        self.in_use = {}                        # account -> {device, ts}

    def _load_accounts(self):
        loaded = []
        if os.path.exists(self.filename):
            try:
                with open(self.filename, "r", encoding="utf-8") as f:
                    loaded = [line.strip() for line in f.readlines() if line.strip()]
            except Exception:
                pass
        # 默认账号池：1211-1221（含 1221�?
        return loaded if loaded else [str(i) for i in range(1211, 1222)]

    def reset(self):
        """每轮 capture 开始前调用：清空占用并重建可用队列�?""
        with self._cv:
            self.available = deque(self.accounts)
            self.in_use = {}
            self._cv.notify_all()
            print("🔄 [AccountPool] 账号池已重置（available=全量，in_use=清空�?)

    def acquire(self, device_alias: str = "", timeout: float = 180.0):
        """获取一个账号（阻塞等待直到超时）。超时返�?None�?""
        end_t = time.time() + float(timeout)
        with self._cv:
            # 若同设备已占用过账号（极少见），直接复用
            for acc, meta in self.in_use.items():
                if meta.get("device") == device_alias and device_alias:
                    return acc

            while not self.available:
                remaining = end_t - time.time()
                if remaining <= 0:
                    return None
                self._cv.wait(timeout=min(1.0, remaining))

            acc = self.available.popleft()
            self.in_use[acc] = {"device": device_alias or "", "ts": time.time()}
            return acc

    def release(self, account: str):
        """释放账号回池。重复释放不会抛错�?""
        if not account:
            return
        with self._cv:
            if account in self.in_use:
                self.in_use.pop(account, None)
                self.available.append(account)
                self._cv.notify_all()

    def snapshot_state(self):
        """便于日志/排障�?""
        with self._lock:
            return {
                "total": len(self.accounts),
                "available": len(self.available),
                "in_use": len(self.in_use),
                "in_use_detail": dict(self.in_use),
            }

account_manager = AccountPool()

# ================= 5. 配置管理 =================

class DeviceConfigManager:
    def __init__(self):
        self.file_path = CONFIG_FILE
        os.makedirs(os.path.dirname(self.file_path), exist_ok=True)
        self.load()

    def load(self):
        with CONFIG_LOCK:
            if os.path.exists(self.file_path):
                try:
                    with open(self.file_path, "r", encoding="utf-8") as f:
                        self.data = json.load(f)
                except:
                    self.data = {}
            else:
                self.data = {}

    def save(self):
        with CONFIG_LOCK:
            with open(self.file_path, "w", encoding="utf-8") as f:
                json.dump(self.data, f, ensure_ascii=False, indent=2)

    def get_quality(self, name: str):
        return self.data.get(name.strip()) if name else None

    def set_quality(self, name: str, quality: str):
        if not name: return
        self.data[name.strip()] = quality
        self.save()

config_manager = DeviceConfigManager()

def save_device_quality(model: str, quality: str):
    config_manager.set_quality(model, quality)
    log_proxy(f"[Config] 已更新设备画�? {model} -> {quality}")

def load_json_config(filename, default=None):
    if default is None: default = {}
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path): return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        return default

_img_config = load_json_config("image_templates.json")
_coord_config = load_json_config("game_coords.json")
_app_config = load_json_config("app_config.json")

# ================= 5.1 UI 语义库（前端标注接入执行层） =================
# 目标：让 Web 标注结果能直接驱动执行层�?ui_click（语�?>模板）�?
# - image_templates.json 仍是“模板参数权威来源”（key=模板�?步骤名，value.filename=图片�?
# - ui_knowledge_base.json 记录“filename -> label(语义�?/tags/notes”等可人工维护信�?

UI_KB_FILE = os.path.join(DATA_DIR, "ui_knowledge_base.json")
UI_STATES_FILE = os.path.join(DATA_DIR, "ui_states.json")
UI_GRAPH_FILE = os.path.join(DATA_DIR, "ui_graph.json")
UI_PENDING_FILE = os.path.join(DATA_DIR, "ui_pending.json")

def _read_json(path: str, default=None):
    if default is None:
        default = {}
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return default

def _write_json(path: str, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)

def load_ui_kb() -> dict:
    """返回：{filename: {label:str, tags:[...], notes:str}}"""
    kb = _read_json(UI_KB_FILE, default={})
    if not isinstance(kb, dict):
        return {}
    # 规范�?
    out = {}
    for fn, meta in kb.items():
        if not fn:
            continue
        if not isinstance(meta, dict):
            meta = {"label": str(meta)}
        out[fn] = {
            "label": str(meta.get("label", "")).strip(),
            "tags": meta.get("tags", []) if isinstance(meta.get("tags", []), list) else [],
            "notes": str(meta.get("notes", "")).strip(),
        }
    return out

def save_ui_kb(kb: dict):
    _write_json(UI_KB_FILE, kb)


def load_ui_states() -> dict:
    """返回：{state_name: {anchors:[...], optional_anchors:[...], notes:str}}"""
    db = _read_json(UI_STATES_FILE, default={})
    return db if isinstance(db, dict) else {}


def save_ui_states(db: dict):
    _write_json(UI_STATES_FILE, db if isinstance(db, dict) else {})


def load_ui_graph() -> dict:
    """返回：{from_state: {actions:[{...}]}}"""
    g = _read_json(UI_GRAPH_FILE, default={})
    return g if isinstance(g, dict) else {}


def save_ui_graph(g: dict):
    _write_json(UI_GRAPH_FILE, g if isinstance(g, dict) else {})


def load_ui_pending() -> list:
    """返回 pending 列表（按时间顺序）�?""
    arr = _read_json(UI_PENDING_FILE, default=[])
    return arr if isinstance(arr, list) else []


def save_ui_pending(arr: list):
    _write_json(UI_PENDING_FILE, arr if isinstance(arr, list) else [])


def _append_pending(rec: dict):
    arr = load_ui_pending()
    arr.append(rec)
    # 控制体积：只保留最�?300 �?
    if len(arr) > 300:
        arr = arr[-300:]
    save_ui_pending(arr)
    return rec


def _img_mean_abs_diff(p1: str, p2: str) -> float:
    """快速判断两张截图是否变化明显（用于过滤“点击无反应”）�?""
    try:
        im1 = Image.open(p1).convert("RGB")
        im2 = Image.open(p2).convert("RGB")
        if im1.size != im2.size:
            im2 = im2.resize(im1.size)
        a1 = np.asarray(im1, dtype=np.int16)
        a2 = np.asarray(im2, dtype=np.int16)
        d = np.abs(a1 - a2)
        return float(d.mean())
    except Exception:
        return 0.0

def _filename_to_template_keys() -> dict:
    """�?image_templates.json 得到 filename -> [template_key,...]"""
    mapping = {}
    for k, v in _img_config.items():
        try:
            fn = str(v.get("filename", "")).strip()
        except Exception:
            fn = ""
        if not fn:
            continue
        mapping.setdefault(fn, []).append(k)
    return mapping

def resolve_template_keys(query: str) -> list:
    """语义解析�?
    1) query 恰好�?IMAGE_TEMPLATES �?key -> [query]
    2) query 匹配 ui_kb 中任�?filename �?label -> 返回�?filename 关联的模�?key 列表
       - 若该 filename 尚未�?image_templates.json 中登记，则返回空（执行层可提示“先入库模板”）
    """
    if not query:
        return []
    q = str(query).strip()
    if not q:
        return []
    if q in IMAGE_TEMPLATES:
        return [q]

    kb = load_ui_kb()
    f2k = _filename_to_template_keys()
    matched = []
    for fn, meta in kb.items():
        if str(meta.get("label", "")).strip() == q:
            matched.extend(f2k.get(fn, []))
    # 去重但保�?
    seen = set()
    out = []
    for k in matched:
        if k not in seen:
            seen.add(k)
            out.append(k)
    return out

def ui_click_semantic(dev, label: str, coords, *, retries: int = 2) -> bool:
    """语义点击：label -> 模板 key 列表 -> ui_click�?
    - 允许一�?label 对应多个模板（不同分辨率/不同状态），逐个尝试�?
    """
    check_stop()
    keys = resolve_template_keys(label)
    if not keys:
        # 兼容：直接按 label 当作模板 key
        keys = [str(label).strip()]

    for _ in range(max(1, retries)):
        for k in keys:
            try:
                if ui_click(dev, k, coords, image=True, ocr=False, fallback=False):
                    return True
            except Exception:
                pass
    return False

def ensure_template_registered(filename: str, *, template_key: str = None):
    """把一�?filename 补进 image_templates.json（最小字段：filename）�?
    - template_key 为空时，使用 filename 作为 key（不会覆盖已�?key�?
    """
    if not filename:
        return False
    fn = str(filename).strip()
    if not fn:
        return False

    # 已存在则不动
    for k, v in _img_config.items():
        if str(v.get("filename", "")).strip() == fn:
            return True

    key = str(template_key).strip() if template_key else fn
    if key in _img_config:
        # 避免覆盖
        key = f"{key}__{int(time.time())}"
    _img_config[key] = {"filename": fn}
    # 写回 data/image_templates.json
    _write_json(os.path.join(DATA_DIR, "image_templates.json"), _img_config)
    # 立即更新 IMAGE_TEMPLATES
    try:
        tpl_path = os.path.join(TEMPLATE_DIR, fn)
        IMAGE_TEMPLATES[key] = Template(tpl_path)
    except Exception:
        pass
    return True

DEFAULT_COORDS = _coord_config.get("default", {})
COORD_OVERRIDES = _coord_config.get("overrides", [])
MODEL_ALIAS = _app_config.get("model_alias", {})

IMAGE_TEMPLATES = {}
for name, params in _img_config.items():
    tpl_path = os.path.join(TEMPLATE_DIR, params.get("filename", ""))
    kwargs = params.copy()
    kwargs.pop("filename", None)
    if "record_pos" in kwargs: kwargs["record_pos"] = tuple(kwargs["record_pos"])
    if "resolution" in kwargs: kwargs["resolution"] = tuple(kwargs["resolution"])
    IMAGE_TEMPLATES[name] = Template(tpl_path, **kwargs)

def get_device_model_name(ip_or_serial: str) -> str:
    try:
        cmd_bt = f"adb -s {ip_or_serial} shell settings get secure bluetooth_name"
        res_bt = subprocess.run(cmd_bt, shell=True, capture_output=True, text=True, timeout=3)
        raw_bt = res_bt.stdout.strip()
        if not raw_bt or raw_bt == "null":
            cmd_model = f"adb -s {ip_or_serial} shell getprop ro.product.model"
            res_model = subprocess.run(cmd_model, shell=True, capture_output=True, text=True, timeout=3)
            raw_name = res_model.stdout.strip()
        else:
            raw_name = raw_bt
        clean_name = re.sub(r"[^\w\s\-]", "", raw_name).strip()
        final_name = re.sub(r"\s+", "_", clean_name)
        if final_name in MODEL_ALIAS: final_name = MODEL_ALIAS[final_name]
        return final_name if final_name else f"Device_{ip_or_serial.replace(':', '_')}"
    except:
        return f"Unknown_{ip_or_serial.replace(':', '_')}"

def get_device_coords_by_model(model_name: str):
    coords = DEFAULT_COORDS.copy()
    model_upper = model_name.upper()
    for override in COORD_OVERRIDES:
        keywords = override.get("keywords", [])
        if any(k.upper() in model_upper for k in keywords):
            coords.update(override.get("coords", {}))
    return coords

def get_device_quality_modes(max_quality: str):
    q = str(max_quality)
    if "极致" in q: return ["极致", "均衡", "省电"]
    if "精致" in q: return ["精致", "均衡", "省电"]
    return ["均衡", "省电"]

# ================= 6. 图像匹配 & 更新检�?=================

def cv_find(dev, template_name: str):
    try:
        if template_name not in IMAGE_TEMPLATES: return None
        screen = dev.snapshot(filename=None)
        if screen is None: return None
        res = IMAGE_TEMPLATES[template_name].match_in(screen)
        if isinstance(res, dict) and "confidence" in res: return res
        return res
    except:
        return None

def is_update_screen(dev) -> bool:
    """
    智能检测当前是否处于：更新、加载、补丁下�?界面
    (GPU 加速版 - 移除复杂缓存逻辑，直接检�?
    """
    try:
        # 1. 纯色黑屏/白屏检�?(游戏启动瞬间)
        try:
            screen = dev.snapshot(filename=None)
            if screen is None: return False
            img_np = np.array(screen)
            if np.mean(img_np) < 20:
                return True
        except:
            pass

        # 2. 图片模板检�?(如果配置�?update 相关图片)
        candidates = ["预加载数�?, "补丁下载�?, "补丁下载中大�?, "加载�?]
        for key in candidates:
            if key in IMAGE_TEMPLATES:
                if cv_find(dev, key): return True

        # 3. OCR 关键词检测（�?OCR_LEVEL>=1，默认关闭以提升吞吐�?
        if OCR_AVAILABLE and OCR_LEVEL >= 1:
            loading_keywords = [
                "正在加载", "加载�?, "Loading", "资源校验",
                "更新�?, "下载�?, "解压�?, "正在连接",
                "正在初始�?
            ]
            try:
                hit = ocr_find_text(dev, loading_keywords)
                if hit:
                    return True
            except Exception:
                pass

    except Exception as e:
        print(f"⚠️ [Check] UpdateScreen 检测异�? {e}")
        return False

    return False

# ================= 7. 文本归一�?/ OCR / UI 抽象�?=================

def normalize_text(s: str) -> str:
    if not s: return ""
    s2 = s.replace(" ", "").replace("\u3000", "").replace("\xa0", "")
    s2 = re.sub(r"[\.\·\…]", "", s2)
    return s2.lower()

def is_subsequence(short: str, long: str) -> bool:
    it = iter(long)
    return all(c in it for c in short)

def _to_pil_image(img):
    """兼容 Airtest snapshot 的返回类型（PIL / numpy�?""
    try:
        if isinstance(img, Image.Image):
            return img
    except Exception:
        pass
    try:
        return Image.fromarray(img)
    except Exception:
        return None

def _normalize_roi(roi, w: int, h: int):
    """
    roi 支持两种格式�?
    - 绝对像素: (x1, y1, x2, y2) �?x2>1 �?y2>1
    - 相对比例: (x1, y1, x2, y2) 且都�?0~1
    返回绝对像素并做边界裁剪�?
    """
    if not roi:
        return None
    x1, y1, x2, y2 = roi
    if 0 <= x1 <= 1 and 0 <= y1 <= 1 and 0 <= x2 <= 1 and 0 <= y2 <= 1:
        x1, y1, x2, y2 = int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h)
    x1 = max(0, min(w - 1, int(x1)))
    y1 = max(0, min(h - 1, int(y1)))
    x2 = max(1, min(w, int(x2)))
    y2 = max(1, min(h, int(y2)))
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)

def entry_ocr_click(dev, tag: str, *, roi=None,
                    min_conf=0.70, stable_frames=2, frame_gap=0.25,
                    verify_fn=None, verify_wait=1.0) -> bool:
    """
    入口专用 OCR 点击�?
    - 只在 ocr_allowed_entry(tag) 时工�?
    - �?ROI 扫描
    - 需要连�?stable_frames 帧都命中（降低误点）
    - 点击后可�?verify_fn 验证是否推进
    """
    if not OCR_AVAILABLE or OCR_MODEL is None:
        return False
    if not ocr_allowed_entry(tag):
        return False

    roi = roi or get_ocr_roi_for_tag(tag)  # 入口 ROI 映射（见 OCR_ROI_BY_TAG�?

    last_pos = None
    hit_count = 0

    for _ in range(stable_frames):
        hit = ocr_find_text(dev, [tag], roi=roi, min_conf=min_conf, min_area=160)
        if not hit:
            hit_count = 0
            simple_wait(frame_gap)
            continue

        _txt, pos, conf = hit

        # 可选：位置抖动保护（两帧位置要接近�?
        if last_pos is not None:
            if abs(pos[0] - last_pos[0]) > 30 or abs(pos[1] - last_pos[1]) > 30:
                hit_count = 0
                last_pos = pos
                simple_wait(frame_gap)
                continue

        last_pos = pos
        hit_count += 1
        simple_wait(frame_gap)

    if hit_count < stable_frames or last_pos is None:
        return False

    dev.touch(last_pos)
    simple_wait(verify_wait)

    if verify_fn:
        try:
            return bool(verify_fn())
        except Exception:
            return False

    return True

def ocr_find_text(dev, keywords, *, roi=None, min_conf: float = 0.68, min_area: int = 160, max_area: int = 999999):
    """
    OCR 搜索文本（带 ROI/置信�?面积过滤）�?
    返回 (text_raw, (cx, cy), conf) �?None
    """
    if not OCR_AVAILABLE or OCR_MODEL is None:
        return None

    check_stop()
    with OCR_LOCK:
        try:
            screen_raw = dev.snapshot(filename=None)
            if screen_raw is None:
                return None

            pil = _to_pil_image(screen_raw)
            if pil is None:
                screen = screen_raw
                result = OCR_MODEL.ocr(screen)
                roi_px = None
            else:
                w, h = pil.size
                roi_px = _normalize_roi(roi, w, h)
                if roi_px:
                    crop = pil.crop(roi_px)
                    screen = np.array(crop)
                else:
                    screen = np.array(pil)
                result = OCR_MODEL.ocr(screen)
        except Exception:
            return None

    if not result or not result[0]:
        return None

    kws_norm = [normalize_text(k) for k in keywords]

    for line in result[0]:
        box = line[0]
        text_raw = line[1][0]
        conf = float(line[1][1]) if len(line[1]) > 1 else 1.0
        if conf < float(min_conf):
            continue

        try:
            xs = [p[0] for p in box]
            ys = [p[1] for p in box]
            area = (max(xs) - min(xs)) * (max(ys) - min(ys))
        except Exception:
            area = 0
        if area < int(min_area) or area > int(max_area):
            continue

        text_norm = normalize_text(text_raw)
        for kw_norm in kws_norm:
            if kw_norm in text_norm or is_subsequence(kw_norm, text_norm):
                cx = (box[0][0] + box[2][0]) / 2
                cy = (box[0][1] + box[2][1]) / 2
                if roi_px:
                    cx += roi_px[0]
                    cy += roi_px[1]
                return text_raw, (cx, cy), conf
    return None

def ocr_find_group(screen, groups):
    if not OCR_AVAILABLE or OCR_MODEL is None: return None
    check_stop()
    with OCR_LOCK:
        try:
            result = OCR_MODEL.ocr(screen)
        except Exception:
            return None

    if not result or not result[0]: return None
    for line in result[0]:
        box = line[0]
        text_raw = line[1][0]
        text_norm = normalize_text(text_raw)
        for group_name, kw_list in groups:
            for kw in kw_list:
                kw_norm = normalize_text(kw)
                if kw_norm in text_norm or is_subsequence(kw_norm, text_norm):
                    cx = (box[0][0] + box[2][0]) / 2
                    cy = (box[0][1] + box[2][1]) / 2
                    return group_name, text_raw, (cx, cy)
    return None

# ================= 7.4 OCR ROI 策略（流程B稳定性核心） =================

OCR_ROI_BY_TAG = {
    "确定": (0.30, 0.60, 0.70, 0.92),
    "确认": (0.30, 0.60, 0.70, 0.92),
    "登录": (0.25, 0.55, 0.75, 0.92),
    "进入": (0.25, 0.55, 0.75, 0.92),
    "进入游戏": (0.25, 0.55, 0.75, 0.92),
    "开始游�?: (0.25, 0.55, 0.75, 0.92),
    "前往征战": (0.25, 0.68, 0.75, 0.92),
    "征战":     (0.25, 0.68, 0.75, 0.92),
    "公告关闭": (0.32, 0.72, 0.68, 0.96),
    "关闭": (0.60, 0.00, 1.00, 0.40),
    "�?�?: (0.60, 0.00, 1.00, 0.40),
    "X": (0.60, 0.00, 1.00, 0.40),
    "x": (0.60, 0.00, 1.00, 0.40),
    "我知道了": (0.20, 0.55, 0.80, 0.90),
    "知道�?: (0.20, 0.55, 0.80, 0.90),
    "取消": (0.20, 0.55, 0.80, 0.90),

    "重新连接": (0.15, 0.35, 0.85, 0.90),
    "重试": (0.15, 0.35, 0.85, 0.90),
    "网络异常": (0.05, 0.25, 0.95, 0.70),
    "连接失败": (0.05, 0.25, 0.95, 0.70),
    "请重�?: (0.05, 0.25, 0.95, 0.70),
}

def get_ocr_roi_for_tag(tag: str):
    if not tag:
        return None
    if tag in OCR_ROI_BY_TAG:
        return OCR_ROI_BY_TAG[tag]
    t = str(tag)
    if any(k in t for k in ["确定", "确认", "登录", "进入"]):
        return (0.25, 0.55, 0.75, 0.92)
    if any(k in t for k in ["关闭", "活动", "提示"]) or t in ["X", "x", "close", "Close"]:
        return (0.60, 0.00, 1.00, 0.45)
    if any(k in t for k in ["重连", "重新连接", "重试", "网络", "失败"]):
        return (0.10, 0.30, 0.90, 0.92)
    return None

def ui_has(dev, tag: str) -> bool:
    check_stop()
    if tag in IMAGE_TEMPLATES:
        res = cv_find(dev, tag)
        if res and isinstance(res, dict): return res.get("confidence", 0) >= 0.7
        return bool(res)
    if OCR_AVAILABLE and ocr_allowed_entry(str(tag)):
        hit = ocr_find_text(dev, [tag], roi=get_ocr_roi_for_tag(tag))
        return hit is not None
    if OCR_AVAILABLE and ocr_allowed(tag):
        hit = ocr_find_text(dev, [tag], roi=get_ocr_roi_for_tag(tag))
        return hit is not None
    return False

    check_stop()
    if tag in IMAGE_TEMPLATES:
        res = cv_find(dev, tag)
        if res and isinstance(res, dict): return res.get("confidence", 0) >= 0.7
        return bool(res)
    if OCR_AVAILABLE and ocr_allowed(tag):
        hit = ocr_find_text(dev, [tag], roi=get_ocr_roi_for_tag(tag))
        return hit is not None
    return False

@retry_action(retries=2, delay=0.5)
def ui_click(dev, tag: str, coords, *, image=True, ocr=True, fallback=True) -> bool:
    check_stop()

    if image and tag in IMAGE_TEMPLATES:
        res = cv_find(dev, tag)
        if res:
            pos = res if isinstance(res, tuple) else res.get("result")
            if pos:
                dev.touch(pos)
                simple_wait(0.5)
                return True

    if ocr and OCR_AVAILABLE and ocr_allowed(tag):
        hit = ocr_find_text(dev, [tag], roi=get_ocr_roi_for_tag(tag))
        if hit:
            _, pos, _conf = hit
            dev.touch(pos)
            simple_wait(0.5)
            return True

    if fallback and tag in coords:
        dev.touch(coords[tag])
        simple_wait(0.5)
        return True

    return False


def is_announcement_popup(dev) -> bool:
    """
    公告弹窗判定（仅用于关闭公告，避免误点其他界面）
    依据�?
      - 顶部标题区域出现“公告�?
      - 且弹窗区域下方存在“关闭”按钮（弱条件）
    """
    if (not OCR_AVAILABLE) or (OCR_LEVEL < 2):
        return False
    try:
        # 标题栏（左上�?
        hit_title = ocr_find_text(dev, ["公告"], roi=(0.05, 0.03, 0.35, 0.18), min_conf=0.55, min_area=200)
        if not hit_title:
            return False
        # 底部按钮（中下）——不强制必须命中，但命中可提高确定�?
        hit_btn = ocr_find_text(dev, ["关闭"], roi=(0.32, 0.72, 0.68, 0.96), min_conf=0.50, min_area=300)
        return True if hit_title else False
    except Exception:
        return False


def close_announcement_popup(dev, device_alias: str = "") -> bool:
    """
    安全关闭公告�?
      1) 仅在 is_announcement_popup==True 时执�?
      2) 优先点中下部“关闭”按钮（更稳定）
      3) 失败再点右上�?X 的近似坐标（避免 OCR 识别不到 X�?
      4) 带验证：公告消失才算成功
    """
    check_stop()
    if (not OCR_AVAILABLE) or (OCR_LEVEL < 2):
        return False
    if not is_announcement_popup(dev):
        return False

    # 1) 优先点“关闭”按�?
    hit = ocr_find_text(dev, ["关闭"], roi=(0.32, 0.72, 0.68, 0.96), min_conf=0.50, min_area=300)
    if hit:
        _txt, pos, conf = hit
        dev.touch(pos)
        simple_wait(0.8)
        if not is_announcement_popup(dev):
            if device_alias:
                log_proxy(f"[{device_alias}] 📣 已关闭公告（按钮�?conf={conf:.2f}")
            return True

    # 2) 兜底点右上角 X（截图中在标题栏右侧�?
    try:
        w, h = dev.get_current_resolution()
        # 右上�?X 大致位置（在弹窗标题栏内，而非屏幕最右上角）
        dev.touch((w * 0.93, h * 0.11))
        simple_wait(0.8)
        if not is_announcement_popup(dev):
            if device_alias:
                log_proxy(f"[{device_alias}] 📣 已关闭公告（右上角X坐标兜底�?)
            return True
    except Exception:
        pass

    return False


# ================= 7.5 弹窗/公告/网络错误统一治理�?=================

COMMON_POPUP_CLOSE_TEXT = [
    "关闭", "�?�?, "取消", "�?�?, "知道�?, "我知道了", "确认", "确定", "OK", "Ok", "ok",
    "X", "x", "Close", "close",
    "活动", "提示",
    "重新连接", "重试", "网络异常", "连接失败", "登录失败", "请重�?, "重新登录", "返回登录",
]

COMMON_POPUP_TEMPLATES = [
    "new_step7_close_notice",
    "new_step8_close_net_error",
]

POPUP_TEXT_RECONNECT = [
    "重新连接", "重试", "重新登录", "返回登录", "请重�?, "网络异常", "连接失败", "登录失败"
]
POPUP_TEXT_CLOSE = [
    "关闭", "�?�?, "取消", "�?�?, "我知道了", "知道�?, "确定", "确认", "OK", "Ok", "ok", "Close", "close", "X", "x"
]
POPUP_TEXT_ANNOUNCE = []  # 已禁用公告类 OCR

def clear_all_popups(dev, coords, device_alias: str = "", max_rounds: int = 6) -> int:
    check_stop()
    closed = 0

    # Try to close announcement popup first if OCR is enabled
    if OCR_AVAILABLE and OCR_LEVEL >= 2:
        if close_announcement_popup(dev, device_alias=device_alias):
            closed += 1
            simple_wait(0.8)

    """
    循环清理可能阻塞流程的弹�?网络错误�?

    �?Image-first：优先模�?坐标
    �?OCR 降级�?
       - OCR_LEVEL=0：完全不使用 OCR（默认，吞吐最高）
       - OCR_LEVEL=1：仅用于“重�?网络异常”等少量关键兜底
       - OCR_LEVEL=2：允许通用关闭�?OCR（旧行为，吞吐最低）

    返回：本次调用累计关闭次数�?
    """
    check_stop()
    closed = 0

    for _ in range(max_rounds):
        hit_this_round = False

        # 1) 模板类弹窗优先（最稳定、最快）
        for tpl in COMMON_POPUP_TEMPLATES:
            use_fallback = tpl in coords
            if ui_click(dev, tpl, coords, image=True, ocr=False, fallback=use_fallback):
                hit_this_round = True
                closed += 1
                simple_wait(0.8)

        # 2) OCR 仅做“重�?网络异常”兜底（默认不开�?
        if OCR_AVAILABLE and OCR_LEVEL >= 1:
            for t in POPUP_TEXT_RECONNECT:
                # 注意：ui_click 内部会再�?ocr_allowed(tag) 过滤
                if ui_click(dev, t, coords, image=False, ocr=True, fallback=False):
                    hit_this_round = True
                    closed += 1
                    simple_wait(1.0)

            # 全量 OCR 模式下才允许通用关闭词（吞吐最低）
            if OCR_LEVEL >= 2:
                for t in POPUP_TEXT_CLOSE:
                    if ui_click(dev, t, coords, image=False, ocr=True, fallback=False):
                        hit_this_round = True
                        closed += 1
                        simple_wait(0.7)

        # 3) 坐标兜底（仅在“可确认命中”时点击，避免误触）
        for key in ("关闭", "取消", "确定"):
            if key in coords:
                if ui_has(dev, key):
                    dev.touch(coords[key])
                    hit_this_round = True
                    closed += 1
                    simple_wait(0.6)

        if not hit_this_round:
            break

    if closed and device_alias:
        log_proxy(f"[{device_alias}] 🧹 弹窗清理完成，本次关�?{closed} �?)
    return closed

def click_with_verify(dev, tag: str, coords, verify_fn, *,
                      image=True, ocr=True, fallback=True,
                      retries: int = 2, post_wait: float = 1.0) -> bool:
    """
    对“关键按钮”使用：点击 -> 等待 -> 验证界面是否发生预期变化�?
    verify_fn 返回 True 代表成功�?
    """
    check_stop()
    for _ in range(max(1, retries)):
        if ui_click(dev, tag, coords, image=image, ocr=ocr, fallback=fallback):
            simple_wait(post_wait)
            try:
                if verify_fn():
                    return True
            except Exception:
                pass
    return False

def try_enter_key(dev, verify_fn=None, post_wait: float = 1.0) -> bool:
    """
    ENTER 键兜底：只在必要时使用，并可选“验证”避免误操作带来的假进展�?
    """
    check_stop()
    try:
        dev.keyevent("ENTER")
        simple_wait(post_wait)
        if verify_fn:
            try:
                return bool(verify_fn())
            except Exception:
                return False
        return True
    except Exception:
        return False

def fast_enter_battle_after_select_server(dev, coords, device_alias: str = "", timeout: float = 35.0) -> bool:
    """
    选服并点“确定”之后：快速识别入口并点“前往征战�?
    组合策略：清弹窗 -> 模板优先 -> 入口专用 ROI OCR 兜底 -> 坐标兜底（带验证�?
    说明�?
      - 不提升全局 OCR_LEVEL（避免多设备并发�?OCR_LOCK 串行拖慢�?
      - 仅在入口阶段对少量关键词启用 OCR（见 ENTRY_OCR_TAGS / ocr_allowed_entry�?
    """
    check_stop()

    ready_entry = ["前往征战", "点击选服", "开始游�?, "进入游戏"]

    # OCR 词库仅在全量 OCR 模式下启用（避免并发吞吐下降�?
    if OCR_AVAILABLE and OCR_LEVEL >= 2:
        kb = load_json_config("ocr_knowledge_base.json", default={})
        ready_entry = kb.get("Ready_Entry", ready_entry)

    def _after_entry_progress() -> bool:
        # 点击入口后，出现这些任一现象都视为“推进成功�?
        if ui_has(dev, "出城"):
            return True
        if is_update_screen(dev):
            return True
        # 入口按钮消失（或界面切换）也可视作推�?
        if (not ui_has(dev, "前往征战")) and (not ui_has(dev, "点击选服")):
            return True
        return False

    last_ocr_t = 0.0

    def _has_entry_any():
        nonlocal last_ocr_t

        if clear_all_popups(dev, coords, device_alias=device_alias) > 0:
            return "progress"

        # �?保护：部分账�?设备会在选服确认后直接进入主�?
        if ui_has(dev, "出城"):
            return "done"

        # �?模板/坐标路径（最快）
        if ui_has(dev, "前往征战") or ui_has(dev, "点击选服"):
            return "done"

        # �?入口专用 OCR：仅�?ROI，且做节流（避免 wait-loop 每秒都跑 OCR�?
        if OCR_AVAILABLE and (time.time() - last_ocr_t) >= 2.0:
            last_ocr_t = time.time()
            try:
                # 优先找“前往征战”（点击目标），其次“点击选服”（说明仍在入口态）
                hit = ocr_find_text(dev, ["前往征战"], roi=get_ocr_roi_for_tag("前往征战"), min_conf=0.68, min_area=180)
                if hit:
                    return "done"
                hit2 = ocr_find_text(dev, ["点击选服"], roi=(0.10, 0.50, 0.90, 0.75), min_conf=0.65, min_area=180)
                if hit2:
                    return "done"
                # 全量模式才扫扩展词库
                if OCR_LEVEL >= 2:
                    hit3 = ocr_find_text(dev, ready_entry, roi=(0.10, 0.55, 0.90, 0.95), min_conf=0.55, min_area=200)
                    if hit3:
                        return "done"
            except Exception:
                pass

        if is_update_screen(dev):
            return "progress"

        return None

    ok = smart_wait_until(
        desc="选服后等待入口页（前往征战/点击选服�?,
        timeout=timeout,
        check_fn=_has_entry_any,
        on_idle=lambda: clear_all_popups(dev, coords, device_alias=device_alias)
    )
    if not ok:
        return False

    # 1) 模板优先（不走通用 OCR�?
    if click_with_verify(
        dev,
        "前往征战",
        coords,
        verify_fn=_after_entry_progress,
        image=True,
        ocr=False,
        fallback=False,
        retries=2,
        post_wait=1.0,
    ):
        return True

    # 2) 入口专用 OCR（两帧稳�?+ ROI + 验证�?
    if entry_ocr_click(
        dev,
        "前往征战",
        stable_frames=2,
        min_conf=0.70,
        verify_fn=_after_entry_progress,
        verify_wait=1.0,
    ):
        return True

    # 3) 坐标兜底（带验证�?
    if "前往征战" in coords:
        try:
            dev.touch(coords["前往征战"])
            simple_wait(1.0)
            if _after_entry_progress():
                return True
        except Exception:
            pass
    return False
# ================= 8. 智能等待控制�?=================

class WaitController:
    def __init__(self, timeout: float, interval: float = 1.0, max_idle: float = None):
        self.timeout = timeout
        self.interval = interval
        self.max_idle = max_idle or max(timeout / 2, 5)
        self.start_t = time.time()
        self.last_progress_t = time.time()

    def expired(self) -> bool:
        return time.time() - self.start_t > self.timeout

    def idle_too_long(self) -> bool:
        return time.time() - self.last_progress_t > self.max_idle

    def record_progress(self):
        self.last_progress_t = time.time()

    def wait(self):
        simple_wait(self.interval)

def smart_wait_until(desc: str, timeout: float, check_fn, on_idle=None):
    w = WaitController(timeout=timeout, interval=1.0)
    log_proxy(f"�?等待: {desc} (最�?{timeout}s)")

    while not w.expired():
        check_stop()

        state = None
        try:
            state = check_fn()
        except TaskStoppedError:
            raise
        except Exception as e:
            log_proxy(f"⚠️ {desc} 检查异�? {e}")
            state = None

        if state == "done":
            log_proxy(f"�?条件达成: {desc}")
            return True
        elif state == "progress":
            w.record_progress()
        else:
            if w.idle_too_long() and on_idle:
                log_proxy(f"⚠️ {desc} 长时间无进展，执�?idle 回调")
                try:
                    on_idle()
                except TaskStoppedError:
                    raise
                except Exception:
                    pass
                w.record_progress()
        w.wait()

    log_proxy(f"�?等待超时: {desc} (> {timeout}s)")
    return False

def wait_ui(dev, tag: str, timeout: float) -> bool:
    return smart_wait_until(desc=f"等待 UI: {tag}", timeout=timeout, check_fn=lambda: "done" if ui_has(dev, tag) else None)


def is_login_main_page(dev) -> bool:
    """
    Giant 登录主界面判定（终态）�?

    �?Image-first：优先使用模板（如你�?image_templates.json 中补充了登录主界面相关模板，可在此生效）
    �?OCR 降级：仅�?OCR_LEVEL>=2 时启用文字判�?
    """
    check_stop()

    # 1) 优先模板（可按需�?image_templates.json 中补充这些模板名�?
    for tpl in ("giant_login_main", "giant_login_phone", "giant_login_title"):
        if tpl in IMAGE_TEMPLATES:
            if cv_find(dev, tpl):
                return True

    # 2) 全量 OCR 模式才允许用“请输入手机�?下一�?巨人账号”做终态判�?
    if OCR_AVAILABLE and OCR_LEVEL >= 2:
        keywords = ["请输入手机号", "下一�?, "巨人账号"]
        roi = (0.20, 0.30, 0.80, 0.85)
        try:
            hit = ocr_find_text(dev, keywords, roi=roi, min_conf=0.65, min_area=300)
            return hit is not None
        except Exception:
            return False

    return False


# ================= 9. 业务流程：新包登�?=================

def _flow_new_package_init(dev, coords, device_alias, serial, retry_count=0):
    check_stop()
    if retry_count > MAX_RETRIES:
        log_proxy(f"[{device_alias}] �?[流程A] 连续失败超限，放�?)
        save_error_snapshot(dev, device_alias, "flowA_fatal_error")
        return

    prefix = f"[{device_alias}] [流程A:新包|第{retry_count + 1}次]"
    fixed_acc_user = "YOUR_ACCOUNT"
    fixed_acc_pwd = "YOUR_PASSWORD"

    try:
        log_proxy(f"{prefix} 🚀 启动游戏...")
        dev.start_app(PACKAGE_NAME)
        simple_wait(5)

        def check_agree():
            if ui_has(dev, "new_step1_agree") or (OCR_AVAILABLE and OCR_LEVEL >= 2 and ui_has(dev, "同意")): return "done"
            if is_update_screen(dev): return "progress"
            return None
        smart_wait_until(desc="协议【同意】弹�?, timeout=20, check_fn=check_agree)
        ui_click(dev, "new_step1_agree", coords, image=True, ocr=False) or (OCR_AVAILABLE and OCR_LEVEL >= 2 and ui_click(dev, "同意", coords))
        simple_wait(2)

        def check_login_env():
            # 登录主界面直接视�?done，禁止清公告
            if is_login_main_page(dev):
                return "done"
            if clear_all_popups(dev, coords, device_alias=device_alias) > 0:
                return "progress"
            if ui_has(dev, "前往征战") or ui_has(dev, "点击选服"):
                return "done"
            if ui_has(dev, "new_step3_enter_acc_input"):
                return "done"
            if is_update_screen(dev):
                return "progress"
            return None

        ok_env = smart_wait_until(desc="等待登录界面(含弹窗清�?", timeout=90, check_fn=check_login_env)

        if not ok_env:
            log_proxy(f"{prefix} ⚠️ 无法找到登录界面，开始诊�?..")
            diagnose_screen(dev, device_alias, "Fail_Login_Env")
            save_error_snapshot(dev, device_alias, f"flowA_env_fail_{retry_count}")
            dev.stop_app(PACKAGE_NAME)
            simple_wait(3)
            return _flow_new_package_init(dev, coords, device_alias, serial, retry_count + 1)

        #if ui_has(dev, "前往征战") or ui_has(dev, "点击选服"):
        #    log_proxy(f"{prefix} 🔎 检测到已在入口界面，跳过账号登录步�?)
        else:
            ui_click(dev, "new_step2_confirm_protocol", coords, image=True) or ui_click(dev, "确定", coords)
            simple_wait(2)
            ui_click(dev, "new_step3_enter_acc_input", coords, image=True, ocr=True)
            simple_wait(1)
            ui_click(dev, "new_step4_click_acc_bar", coords, image=True, ocr=True)
            simple_wait(1)
            log_proxy(f"{prefix} ⌨️ 输入账号")
            dev.text(fixed_acc_user)
            simple_wait(1)
            ui_click(dev, "new_step5_enter_pwd_input", coords, image=True, ocr=True)
            simple_wait(1)
            log_proxy(f"{prefix} ⌨️ 输入密码")
            dev.text(fixed_acc_pwd)
            simple_wait(1)

        log_proxy(f"{prefix} 🖱�?点击登录按钮")
        clicked_login = False
        login_candidates = [
            ("new_step6_confirm_login", True, False),
            ("确定", True, False),
        ]
        if OCR_AVAILABLE and OCR_LEVEL >= 2:
            login_candidates.insert(1, ("登录", False, True))
        for tag, use_img, use_ocr in login_candidates:
            if ui_click(dev, tag, coords, image=use_img, ocr=use_ocr, fallback=False):
                clicked_login = True
                break

        if not clicked_login:
            try:
                log_proxy(f"{prefix} ⌨️ 尝试通过回车键确认登�?)
                dev.keyevent("ENTER")
                clicked_login = True
            except Exception:
                pass

        if not clicked_login:
            log_proxy(f"{prefix} ⚠️ 无法找到登录按钮")
            diagnose_screen(dev, device_alias, "Fail_Click_Login")

        simple_wait(5)

        login_retry = {"count": 0}
        def check_entry_active():
            if clear_all_popups(dev, coords, device_alias=device_alias) > 0:
                return "progress"

            if ui_has(dev, "前往征战") or ui_has(dev, "点击选服"):
                return "done"

            if ui_has(dev, "new_step3_enter_acc_input"):
                if login_retry["count"] < 3:
                    log_proxy(f"{prefix} �?仍在登录界面，第 {login_retry['count']+1} 次尝试补点登录按�?)
                    def _login_progress():
                        return (not ui_has(dev, "new_step3_enter_acc_input")) or is_update_screen(dev) or ui_has(dev, "前往征战") or ui_has(dev, "点击选服")
                    for tag in ["new_step6_confirm_login", "登录", "确认登录", "确定", "进入游戏", "开始游�?]:
                        if click_with_verify(dev, tag, coords, _login_progress, image=(tag == "new_step6_confirm_login"), ocr=True, fallback=False, retries=1, post_wait=2.0):
                            login_retry["count"] += 1
                            return "progress"
                return "progress"

            if is_update_screen(dev):
                return "progress"
            return None

        ok_entry = smart_wait_until(desc="等待游戏入口(自动处理公告/报错)", timeout=60, check_fn=check_entry_active)

        if not ok_entry:
            log_proxy(f"{prefix} ⚠️ 未检测到入口，重�?..")
            diagnose_screen(dev, device_alias, "Fail_Entry")
            save_error_snapshot(dev, device_alias, f"flowA_entry_fail_{retry_count}")
            dev.stop_app(PACKAGE_NAME)
            return _flow_new_package_init(dev, coords, device_alias, serial, retry_count + 1)

        log_proxy(f"{prefix} �?发现入口，开始选服")
        ui_click(dev, "点击选服", coords, image=True, ocr=True)
        simple_wait(1)
        ui_click(dev, "预发�?, coords, image=True)
        simple_wait(1)
        ui_click(dev, "TEST_SERVER", coords, image=True)
        simple_wait(1)
        ui_click(dev, "确定", coords)
        simple_wait(0.8)

        if not fast_enter_battle_after_select_server(dev, coords, device_alias=device_alias, timeout=35):
            log_proxy(f"{prefix} ⚠️ 选服后未能稳定进入征战入口，开始诊�?)
            diagnose_screen(dev, device_alias, "Fail_EnterBattle_A")

        # 入口点击兜底：部分机型模板不稳（如华为高 DPI），此处使用入口专用 OCR（不提升全局 OCR_LEVEL�?
        if not ui_has(dev, "出城"):
            def _flowA_after_entry_progress():
                if ui_has(dev, "出城"):
                    return True
                if is_update_screen(dev):
                    return True
                if (not ui_has(dev, "前往征战")) and (not ui_has(dev, "点击选服")):
                    return True
                return False

            # 模板优先
            click_with_verify(dev, "前往征战", coords, _flowA_after_entry_progress, image=True, ocr=False, fallback=False, retries=1, post_wait=1.0)
            # 入口专用 OCR 兜底
            entry_ocr_click(dev, "前往征战", stable_frames=2, min_conf=0.70, verify_fn=_flowA_after_entry_progress, verify_wait=1.0)
        simple_wait(3)

        dev.stop_app(PACKAGE_NAME)
        log_proxy(f"{prefix} �?流程A执行完毕")

    except TaskStoppedError:
        log_proxy(f"{prefix} 🛑 任务已强制停�?)
        raise
    except Exception as e:
        log_proxy(f"{prefix} ⚠️ 异常: {e}")
        save_error_snapshot(dev, device_alias, f"flowA_exception_{retry_count}")
        try:
            dev.stop_app(PACKAGE_NAME)
        except:
            pass
        return _flow_new_package_init(dev, coords, device_alias, serial, retry_count + 1)

# ================= 10. 业务流程：截图任�?(含优�? =================

def _flow_screenshot_task(dev, coords, device_alias, custom_tasks, save_root, serial, fixed_account=None, retry_count=0):
    check_stop()
    if retry_count > MAX_RETRIES:
        log_proxy(f"[{device_alias}] �?[流程B] 失败次数超限，跳�?)
        save_error_snapshot(dev, device_alias, "flowB_fatal_error")
        return

    prefix = f"[{device_alias}] [流程B:截图|第{retry_count + 1}次]"
    current_account = fixed_account  # 若为 None，则在入口就绪后再从账号�?acquire（避免卡住设备提前占用账号）
    device_save_dir = os.path.join(save_root, device_alias)
    os.makedirs(device_save_dir, exist_ok=True)

    try:
        log_proxy(f"{prefix} 🚀 启动游戏...")
        dev.start_app(PACKAGE_NAME)
        simple_wait(5)

        def check_start_env():
            if clear_all_popups(dev, coords, device_alias=device_alias) > 0:
                return "progress"
            if ui_has(dev, "前往征战") or ui_has(dev, "账号"):
                return "done"
            if is_update_screen(dev):
                return "progress"
            return None

        def handle_start_idle():
            clear_all_popups(dev, coords, device_alias=device_alias)

        ok_env = smart_wait_until(desc="启动环境准备", timeout=90, check_fn=check_start_env, on_idle=handle_start_idle)

        if not ok_env:
            save_error_snapshot(dev, device_alias, f"flowB_start_fail_{retry_count}")
            try:
                dev.stop_app(PACKAGE_NAME)
            except:
                pass
            simple_wait(3)
            return _flow_screenshot_task(dev, coords, device_alias, custom_tasks, save_root, serial, current_account, retry_count + 1)

        def check_flowB_entry_active():
            if clear_all_popups(dev, coords, device_alias=device_alias) > 0:
                return "progress"
            if ui_has(dev, "前往征战") or ui_has(dev, "点击选服") or ui_has(dev, "账号"):
                return "done"
            if is_update_screen(dev):
                return "progress"
            return None

        ok_entry_b = smart_wait_until(desc="入口状态确�?, timeout=60, check_fn=check_flowB_entry_active)

        if not ok_entry_b:
            log_proxy(f"{prefix} ⚠️  入口未就绪，重试")
            save_error_snapshot(dev, device_alias, f"flowB_step7_fail_{retry_count}")
            dev.stop_app(PACKAGE_NAME)
            simple_wait(3)
            return _flow_screenshot_task(dev, coords, device_alias, custom_tasks, save_root, serial, current_account, retry_count + 1)

        # �?关键修复：账号“临用临取�?
        # 只有当设备真正进入流程B入口（准备输入账号）时才占用账号，避免卡住设备提前耗尽账号池�?
        acquired_by_this_call = False
        if not current_account:
            current_account = account_manager.acquire(device_alias=device_alias, timeout=180.0)
            acquired_by_this_call = True
            if not current_account:
                st = account_manager.snapshot_state()
                log_proxy(f"{prefix} �?账号池已空且等待超时，跳过该设备（state={st})")
                try:
                    dev.stop_app(PACKAGE_NAME)
                except Exception:
                    pass
                return
        else:
            # 如果是重试递归带入�?fixed_account，则不再次占�?
            acquired_by_this_call = False

        log_proxy(f"{prefix} 🔑 登录账号: {current_account}")
        ui_click(dev, "账号", coords, image=True, ocr=True)
        simple_wait(1)

        try:
            if hasattr(dev, "clear_text"): dev.clear_text()
            else:
                try:
                    dev.keyevent("CTRL+A"); dev.keyevent("DEL")
                except:
                    pass
            dev.text(str(current_account))
            simple_wait(0.5)
        except Exception:
            save_error_snapshot(dev, device_alias, f"flowB_input_account_fail_{retry_count}")
            raise

        log_proxy(f"{prefix} 🌍 选服 TEST_SERVER")
        ui_click(dev, "点击选服", coords, image=True, ocr=True)
        simple_wait(1)
        ui_click(dev, "预发�?, coords, image=True)
        simple_wait(1)
        ui_click(dev, "TEST_SERVER", coords, image=True)
        simple_wait(0.5)
        ui_click(dev, "TEST_SERVER", coords, image=True, ocr=True)
        simple_wait(1)

        log_proxy(f"{prefix} 🖱�?点击『确定』并验证跳转（稳登录�?)
        clear_all_popups(dev, coords, device_alias=device_alias)

        def _after_confirm_progress():
            if is_update_screen(dev):
                return True
            if (not ui_has(dev, "TEST_SERVER")) and (not ui_has(dev, "预发�?)) and (not ui_has(dev, "点击选服")):
                return True
            if ui_has(dev, "前往征战"):
                return True
            return False

        confirm_clicked = click_with_verify(
            dev,
            "确定",
            coords,
            verify_fn=_after_confirm_progress,
            image=True,
            ocr=True,
            fallback=True,
            retries=2,
            post_wait=1.2,
        )

        if not confirm_clicked:
            log_proxy(f"{prefix} ⌨️ ENTER 兜底（带验证�?)
            try_enter_key(dev, verify_fn=_after_confirm_progress, post_wait=1.0)

        simple_wait(2.5)

        # �?保护：若选服确认后已直接进入主城（出现『出城』）�?
        # 则跳过入口页识别/点击『前往征战』阶段，直接开始后续任务�?
        if ui_has(dev, "出城"):
            log_proxy(f"{prefix} �?已在主城（检测到『出城』），跳过入口页等待")
        else:
            if not fast_enter_battle_after_select_server(dev, coords, device_alias=device_alias, timeout=35):
                log_proxy(f"{prefix} ⚠️ 选服后未能稳定识别并点击『前往征战』，开始诊�?)
                diagnose_screen(dev, device_alias, "Fail_EnterBattle_B")
            else:
                simple_wait(6)

        if ui_has(dev, "预发�?) or ui_has(dev, "TEST_SERVER"):
            log_proxy(f"{prefix} ⚠️ 未跳过选服，尝试盲�?..")
            try:
                w, h = dev.get_current_resolution()
                dev.touch((w * 0.5, h * 0.85))
            except:
                pass
            simple_wait(2)

        # 若尚未进入主城，则尝试点『前往征战』推进；否则跳过�?
        if not ui_has(dev, "出城"):
            def _flowB_after_entry_progress():
                if ui_has(dev, "出城"):
                    return True
                if is_update_screen(dev):
                    return True
                if (not ui_has(dev, "前往征战")) and (not ui_has(dev, "点击选服")):
                    return True
                return False

            click_with_verify(dev, "前往征战", coords, _flowB_after_entry_progress, image=True, ocr=False, fallback=False, retries=1, post_wait=1.0)
            entry_ocr_click(dev, "前往征战", stable_frames=2, min_conf=0.70, verify_fn=_flowB_after_entry_progress, verify_wait=1.0)
            simple_wait(8)

        ok_city = smart_wait_until(
            desc="等待主城【出城�?,
            timeout=60,
            check_fn=lambda: "done" if ui_has(dev, "出城") else None,
            on_idle=lambda: clear_all_popups(dev, coords, device_alias=device_alias)
        )
        if not ok_city:
            log_proxy(f"{prefix} ⚠️ 未能进入主城")
            save_error_snapshot(dev, device_alias, f"flowB_enter_city_fail_{retry_count}")
            dev.stop_app(PACKAGE_NAME)
            return _flow_screenshot_task(dev, coords, device_alias, custom_tasks, save_root, serial, current_account, retry_count + 1)

        ui_click(dev, "收起任务�?, coords, image=True)
        ui_click(dev, "出城", coords, image=True)
        simple_wait(3)

        for task in custom_tasks or []:
            check_stop()
            loc_name = task.get("name", "未命�?)
            x, y = task.get("x"), task.get("y")
            log_proxy(f"{prefix} 📸 执行截图: {loc_name} ({x}, {y})")

            ui_click(dev, "快速查�?, coords, image=True)
            simple_wait(1)
            ui_click(dev, "横坐�?, coords, image=True)
            simple_wait(0.5)
            dev.text(str(x))
            ui_click(dev, "纵坐�?, coords, image=True)
            simple_wait(0.5)
            dev.text(str(y))
            ui_click(dev, "前往", coords, image=True)
            simple_wait(3)

            max_q = config_manager.get_quality(device_alias) or "极致"
            for q in get_device_quality_modes(max_q):
                check_stop()
                try:
                    ui_click(dev, "设置�?, coords, image=True)
                    simple_wait(0.5)
                    ui_click(dev, "设置", coords, image=True)
                    simple_wait(0.5)
                    ui_click(dev, "画面设置", coords, image=True)
                    simple_wait(0.5)
                    ui_click(dev, q, coords, image=True)
                    simple_wait(0.5)
                    ui_click(dev, "返回", coords)
                    simple_wait(0.5)
                    ui_click(dev, "返回", coords)
                    simple_wait(1.5)

                    fname = f"{loc_name}_{q}.png"
                    img_path = os.path.join(device_save_dir, fname)
                    safe_snapshot(dev, img_path)

                    # �?质量检测：仅做“坏图标记”，不做相似�?智能差异判断
                    try:
                        meta = analyze_image_quality(img_path)
                        meta["device"] = device_alias
                        meta["loc_name"] = loc_name
                        meta["quality_mode"] = q
                        meta["ts"] = int(time.time())
                        meta_path = write_image_meta(img_path, meta)

                        # 若命中强异常，额外复制到集中目录，方便快速查看（不移动原图，避免报告缺图�?
                        if meta.get("flags"):
                            bad_root = os.path.join(save_root, "_bad", device_alias)
                            os.makedirs(bad_root, exist_ok=True)
                            try:
                                shutil.copy2(img_path, os.path.join(bad_root, fname))
                                if meta_path and os.path.exists(meta_path):
                                    shutil.copy2(meta_path, os.path.join(bad_root, os.path.basename(meta_path)))
                            except Exception:
                                pass
                    except Exception:
                        pass

                    log_proxy(f"{prefix} 💾 已保�? {fname}")
                except TaskStoppedError:
                    raise
                except Exception as e:
                    log_proxy(f"{prefix} ⚠️ 画质切换/截图失败: {e}")
                    save_error_snapshot(dev, device_alias, f"capture_fail_{q}")

        dev.stop_app(PACKAGE_NAME)
        log_proxy(f"{prefix} �?截图任务完成")

    except TaskStoppedError:
        log_proxy(f"{prefix} 🛑 任务已停�?)
        raise
    except Exception as e:
        log_proxy(f"{prefix} ⚠️ 流程B异常: {e}")
        save_error_snapshot(dev, device_alias, f"flowB_exception_{retry_count}")
        try:
            dev.stop_app(PACKAGE_NAME)
        except:
            pass
        return _flow_screenshot_task(dev, coords, device_alias, custom_tasks, save_root, serial, current_account, retry_count + 1)

    finally:
        # �?确保账号释放：一次截图流程内账号只占用一次，只有设备结束/退出后才回到池�?
        try:
            if 'acquired_by_this_call' in locals() and acquired_by_this_call and current_account:
                account_manager.release(str(current_account))
                log_proxy(f"{prefix} ♻️ 已释放账�? {current_account}")
        except Exception:
            pass

# ================= 11. 业务流程：装�?=================

def _adb_shell(serial_or_dev, cmd: str, timeout=15) -> str:
    try:
        if hasattr(serial_or_dev, "adb"):
            return serial_or_dev.adb.shell(cmd, timeout=timeout) or ""
    except Exception:
        pass
    try:
        return subprocess.check_output(cmd, shell=True, timeout=timeout).decode("utf-8", errors="ignore")
    except Exception:
        return ""

def verify_package_installed(dev, package: str) -> bool:
    out = _adb_shell(dev, f"pm path {package}", timeout=10)
    return bool(out and "package:" in out)

def get_install_error_hint(output: str) -> str:
    s = (output or "").upper()
    if "INSTALL_FAILED_UPDATE_INCOMPATIBLE" in s or "SIGNATURE" in s:
        return "signature_incompatible"
    if "INSTALL_FAILED_VERSION_DOWNGRADE" in s:
        return "version_downgrade"
    if "INSTALL_PARSE_FAILED" in s:
        return "parse_failed"
    if "INSTALL_FAILED_USER_RESTRICTED" in s or "USER_RESTRICTED" in s:
        return "user_restricted"
    return "unknown"

def install_split_apks(dev, package_name: str, apk_dir: str, device_alias: str):
    """
    apk_dir 内包�?base.apk + 若干 split*.apk
    """
    base = os.path.join(apk_dir, "base.apk")
    if not os.path.exists(base):
        cands = [os.path.join(apk_dir, f) for f in os.listdir(apk_dir) if f.lower().endswith(".apk")]
        if not cands:
            raise RuntimeError("split 目录内找不到任何 apk")
        base_like = [p for p in cands if "base" in os.path.basename(p).lower()]
        base = base_like[0] if base_like else cands[0]

    apks = [os.path.join(apk_dir, f) for f in os.listdir(apk_dir) if f.lower().endswith(".apk")]
    apks = sorted(list(set(apks)))

    remote_paths = []
    for p in apks:
        rp = f"/data/local/tmp/{os.path.basename(p)}"
        dev.adb.push(p, rp)
        remote_paths.append(rp)

    cmd = "pm install-multiple -r -g " + " ".join([f"'{p}'" for p in remote_paths])
    out = dev.adb.shell(cmd, timeout=240) or ""
    log_proxy(f"[{device_alias}] 📦 split install output: {out}")
    if "Success" not in out:
        raise RuntimeError(out)

    if not verify_package_installed(dev, package_name):
        raise RuntimeError("split install success but verify failed")

def _flow_install_task(dev, device_alias, apk_path):
    check_stop()
    prefix = f"[{device_alias}] [流程:装包]"

    if not apk_path or not os.path.isfile(apk_path):
        log_proxy(f"{prefix} �?APK 无效: {apk_path}")
        return False

    # -------------------- 小工具：校验闭环 --------------------
    def _adb_shell_local(cmd: str, timeout: int = 20) -> str:
        check_stop()

        # 1) Airtest wrapper（兼容旧版，不传 timeout�?
        try:
            try:
                # 新版 Airtest
                out = dev.adb.shell(cmd, timeout=timeout)
            except TypeError:
                # 旧版 Airtest（不支持 timeout�?
                out = dev.adb.shell(cmd)

            if out:
                out_s = str(out).strip()
                if out_s:
                    return out_s

        except Exception as e:
            log_proxy(f"{prefix} ⚠️ dev.adb.shell 异常: {type(e).__name__}: {e}")

        # 2) subprocess 兜底（保持你原来的逻辑�?
        try:
            adb_path = getattr(dev.adb, "adb_path", "adb")
            serial = getattr(dev.adb, "serialno", None) or getattr(dev, "serialno", None)
            if not serial:
                serial = getattr(dev, "uuid", "")

            p = subprocess.run(
                [adb_path, "-s", serial, "shell", cmd],
                capture_output=True,
                text=True,
                timeout=timeout
            )
            return (p.stdout or p.stderr or "").strip()
        except Exception as e:
            log_proxy(f"{prefix} �?subprocess adb shell 失败: {e}")
            return ""
        except Exception as e:
            log_proxy(f"{prefix} �?subprocess adb shell 也失�? {type(e).__name__}: {e}")
            return ""
    def _ensure_device_online():
        adb_path = getattr(dev.adb, "adb_path", "adb")
        serial = getattr(dev.adb, "serialno", None) or getattr(dev, "serialno", None) or getattr(dev, "uuid", "")
        if not serial:
            return False

        # 如果�?IP:PORT 形式，先 adb connect 一次（无害，可重复�?
        if ":" in serial:
            try:
                subprocess.run([adb_path, "connect", serial], timeout=5, capture_output=True, text=True)
            except Exception:
                pass

        try:
            out = subprocess.check_output([adb_path, "devices"], timeout=5).decode("utf-8", errors="ignore")
            return f"{serial}\tdevice" in out
        except Exception:
            return False

    def _get_apk_pkg_name_from_path(apk_local: str) -> str:
        try:
            cmd = f'aapt dump badging "{apk_local}"'
            p = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=10)
            txt = (p.stdout or "") + "\n" + (p.stderr or "")
            m = re.search(r"package:\s+name='([^']+)'", txt)
            if m:
                return m.group(1).strip()
        except Exception:
            pass
        return PACKAGE_NAME

    target_pkg = _get_apk_pkg_name_from_path(apk_path)
    if target_pkg != PACKAGE_NAME:
        log_proxy(f"{prefix} ℹ️ APK 解析包名: {target_pkg}（默认脚本包�? {PACKAGE_NAME}�?)

    # �?修复点：包名候选列表（避免 aapt 不可�?解析失败导致装完误判�?
    pkg_candidates = []
    if target_pkg:
        pkg_candidates.append(target_pkg)
    if PACKAGE_NAME not in pkg_candidates:
        pkg_candidates.append(PACKAGE_NAME)

    # �?修复点：多策略校�?+ 可观测日�?
    def _verify_installed(pkg: str, timeout_s: float = 25.0, log_every: float = 5.0) -> bool:
        check_stop()
        end = time.time() + float(timeout_s)
        last_log = 0.0
        while time.time() < end:
            out_path = _adb_shell_local(f"pm path {pkg}", timeout=8) or ""
            ok_path = ("package:" in out_path)

            out_list = _adb_shell_local(f"pm list packages | grep {pkg}", timeout=8)
            ok_list = (pkg in (out_list or ""))

            now = time.time()
            if (now - last_log) >= log_every:
                log_proxy(f"{prefix} 🔎 verify pkg={pkg} | pm path='{out_path.strip()[:120]}' | pm list hit={ok_list}")
                last_log = now

            if ok_path or ok_list:
                return True

            time.sleep(1.0)
        return False

    def _verify_any_installed(timeout_s: float = 25.0) -> bool:
        for p in pkg_candidates:
            if _verify_installed(p, timeout_s=timeout_s):
                return True
        return False

    # -------------------- 0) 清理旧包（不强制成功�?--------------------
    log_proxy(f"{prefix} 📦 准备安装: {os.path.basename(apk_path)}")
    try:
        dev.stop_app(target_pkg)
        simple_wait(0.5)
    except Exception:
        pass
    try:
        dev.uninstall_app(target_pkg)
        simple_wait(0.5)
    except Exception:
        pass

    # -------------------- 1) ADB push + pm install（优先） --------------------
    apk_name = os.path.basename(apk_path)
    remote_tmp = f"/data/local/tmp/{apk_name}"               # �?pm install 读取该路径（system_server 有权限）
    remote_sd  = f"/sdcard/Download/{apk_name}"              # �?UI 安装器兜底路径（file:// 更稳�?
    remote_apk = remote_sd                                   # 兼容旧变量名（后�?UI 兜底复用�?

    # 1.0) WiFi ADB 容易掉线：push / shell 前先确保设备在线
    if not _ensure_device_online():
        log_proxy(f"{prefix} ⚠️ 设备离线/找不到，已尝试重连但未就绪（可能会走后续兜底流程�?)
    else:
        # 1.1) push �?/data/local/tmp（修复华�?高版本安卓对 /sdcard fuse �?SELinux 读权限问题）
        try:
            check_stop()
            dev.adb.push(apk_path, remote_tmp)
            log_proxy(f"{prefix} �?push 成功(tmp): {remote_tmp}")
        except Exception as e:
            log_proxy(f"{prefix} �?push 失败(tmp): {e}")
        else:
            # 1.2) pm install 直接�?/data/local/tmp
            check_stop()
            pm_out = _adb_shell_local(f"pm install -r -t -d '{remote_tmp}'", timeout=240)

            if "Success" in (pm_out or ""):
                log_proxy(f"{prefix} �?pm install 成功，开始校验安装状�?..")
                if _verify_any_installed(timeout_s=25):
                    log_proxy(f"{prefix} �?校验通过：pm 已确认安�?)
                    # 清理临时文件（不强制�?
                    try:
                        _adb_shell_local(f"rm -f '{remote_tmp}'", timeout=8)
                    except Exception:
                        pass
                    return True
                else:
                    log_proxy(f"{prefix} ⚠️ pm install �?Success，但校验未确认（继续兜底流程�?)
            else:
                if (pm_out or "").strip():
                    log_proxy(f"{prefix} ⚠️ pm install 返回: {(pm_out or '').strip()[:180]}")
                else:
                    log_proxy(f"{prefix} ⚠️ pm install 无输出，继续兜底流程")

            # 1.3) �?UI 安装器兜底准�?sdcard 副本（安装器�?file:// 访问 /data/local/tmp 不一定可用）
            try:
                _adb_shell_local("mkdir -p /sdcard/Download", timeout=8)
                _adb_shell_local(f"cp -f '{remote_tmp}' '{remote_sd}'", timeout=12)
                log_proxy(f"{prefix} �?兜底副本已准�?sd): {remote_sd}")
            except Exception as e:
                log_proxy(f"{prefix} ⚠️ 准备 sdcard 兜底副本失败: {e}")
                # 退一步：直接 push 一份到 sdcard（可能更慢，但兜底）
                try:
                    dev.adb.push(apk_path, remote_sd)
                    log_proxy(f"{prefix} �?push 成功(sd): {remote_sd}")
                except Exception as e2:
                    log_proxy(f"{prefix} �?push 失败(sd): {e2}")


    # -------------------- 2) dev.install_app（第二优先） --------------------
    try:
        check_stop()
        dev.install_app(apk_path)
        log_proxy(f"{prefix} �?dev.install_app 调用完成，开始校验安装状�?..")
        if _verify_any_installed(timeout_s=25):
            log_proxy(f"{prefix} �?校验通过：pm 已确认安�?)
            return True
        else:
            log_proxy(f"{prefix} ⚠️ dev.install_app 后仍未通过校验，继�?UI 安装器兜�?)
    except Exception as e:
        log_proxy(f"{prefix} ⚠️ dev.install_app 失败: {e}，进�?UI 安装器兜�?)

    # -------------------- 3) UI 安装器兜底：拉起安装�?+ ROI OCR 状态机 --------------------
    try:
        check_stop()
        am_cmd = (
            f'am start -a android.intent.action.VIEW '
            f'-d "file://{remote_apk}" '
            f'-t "application/vnd.android.package-archive"'
        )
        _adb_shell_local(am_cmd, timeout=12)
        simple_wait(1.2)
        log_proxy(f"{prefix} 🧩 已拉起系统安装器 UI（开�?OCR 状态机�?)
    except Exception as e:
        log_proxy(f"{prefix} �?拉起安装器失�? {e}")
        return False

    def _install_roi_from_snapshot(img_pil: Image.Image):
        w, h = img_pil.size
        return (int(w * 0.10), int(h * 0.45), int(w * 0.90), int(h * 0.97))

    kw_install = [
        "安装", "继续安装", "立即安装", "开始安�?, "同意并安�?,
        "仍要安装", "继续", "下一�?, "允许", "允许安装",
        "允许来自此来�?, "允许来自此来源安�?, "允许此来�?, "允许此应�?
    ]
    kw_finish = ["完成", "打开", "安装完成", "已安�?, "已完�?]
    kw_block = [
        "已阻�?, "出于安全原因", "无法安装", "解析软件包时出现问题",
        "应用未安�?, "安装失败", "与设备不兼容", "签名不一�?, "存在风险"
    ]
    kw_go_settings = ["设置", "去设�?, "允许此来�?, "允许来自此来�?, "允许安装", "允许"]

    state = "waiting_install"
    last_action_t = time.time()

    def _ocr_tap_any(keywords, roi):
        check_stop()
        if not OCR_AVAILABLE:
            return False
        hit = ocr_find_text(dev, keywords, roi=roi, min_conf=0.62, min_area=140, max_area=250000)
        if hit:
            _txt, pos, _conf = hit
            dev.touch(pos)
            simple_wait(0.9)
            return True
        return False

    def _check_install_ui():
        nonlocal state, last_action_t
        check_stop()

        # �?修复�?：最可靠的早退——只�?pm 校验通过�?done
        if _verify_any_installed(timeout_s=1.5):
            return "done"

        # �?修复�?：入�?UI 早退（装完安装器退�?被覆盖时，避�?OCR 空转�?
        if ui_has(dev, "前往征战") or ui_has(dev, "点击选服"):
            return "done"

        try:
            raw = dev.snapshot(filename=None)
            if raw is None:
                return None
            pil = _to_pil_image(raw)
            if pil is None:
                return None
            roi_px = _install_roi_from_snapshot(pil)
        except Exception:
            return None

        if _ocr_tap_any(kw_block, roi_px):
            if _ocr_tap_any(kw_go_settings, roi_px):
                state = "waiting_install"
                return "progress"
            if _ocr_tap_any(kw_go_settings, (0.05, 0.05, 0.95, 0.55)):
                state = "waiting_install"
                return "progress"
            return "progress"

        if _ocr_tap_any(kw_install, roi_px):
            state = "installing"
            last_action_t = time.time()
            return "progress"

        if _ocr_tap_any(kw_finish, roi_px):
            last_action_t = time.time()
            if _verify_any_installed(timeout_s=6.0):
                return "done"
            return "progress"

        if state == "installing":
            if time.time() - last_action_t > 8.0:
                if _ocr_tap_any(kw_finish, roi_px):
                    if _verify_any_installed(timeout_s=6.0):
                        return "done"
                    return "progress"
                last_action_t = time.time()
                return "progress"
            return "progress"

        return None

    ok = smart_wait_until(desc="安装�?UI（ROI OCR + 校验闭环�?, timeout=600, check_fn=_check_install_ui)

    if ok:
        log_proxy(f"{prefix} �?UI 装包成功（已通过 pm 校验�?)
        return True

    log_proxy(f"{prefix} �?UI 装包超时/失败：未通过安装校验（pm�?)
    return False

# ================= 12. 多设备入�?=================

def connect_devices_batch(input_text: str):
    results = []
    pattern = r"(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}:\d+)"
    targets = list(set(re.findall(pattern, input_text)))
    if not targets:
        return [{"ip": "无效输入", "status": "error", "msg": "格式错误"}]
    config_manager.load()
    for target in targets:
        try:
            subprocess.run(["adb", "connect", target], timeout=5, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if f"{target}\tdevice" in subprocess.check_output(["adb", "devices"]).decode("utf-8"):
                model = get_device_model_name(target)
                q = config_manager.get_quality(model)
                results.append({
                    "ip": target,
                    "model": model,
                    "status": "success" if q else "warning",
                    "msg": f"�?就绪 ({q})" if q else "⚠️ 需配置画质",
                    "need_config": not q
                })
            else:
                results.append({"ip": target, "status": "warning", "msg": "⚠️ 离线", "need_config": False})
        except Exception as e:
            results.append({"ip": target, "status": "error", "msg": str(e), "need_config": False})
    return results

def disconnect_device(serial):
    try:
        subprocess.run(["adb", "disconnect", serial], timeout=3, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except Exception as e:
        print(f"⚠️ disconnect_device error: {e}")
        return False

def disconnect_all_devices():
    try:
        subprocess.run(["adb", "disconnect"], timeout=3, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except Exception as e:
        print(f"⚠️ disconnect_all_devices error: {e}")
        return False


# ================= 13. UI 候选提取（用于前端标注扩容�?=================
# 说明：SLG �?UI 图标往往是“中小尺寸的�?矩形图块”�?
# 我们用一个保守的 CV 轮廓提取策略�?
# - 先做灰度 + 轻微模糊 + Canny
# - 提取外轮廓，按尺�?长宽比过�?
# - bbox 内再做轻�?padding 进行裁剪
# 输出：保存为 scripts/ui_candidates/tpl_ui_<ts>_<n>.png

def extract_ui_candidates_from_image(image_path: str, out_dir: str, *,
                                     min_size: int = 36, max_size: int = 220,
                                     min_area: int = 36 * 36,
                                     max_area: int = 260 * 260,
                                     aspect_min: float = 0.55,
                                     aspect_max: float = 1.80,
                                     padding: int = 6,
                                     max_items: int = 120) -> list:
    """从一张截图里自动裁剪“疑�?UI 图标”候选�?

    返回：生成的候选图片绝对路径列表�?
    """
    try:
        import cv2
    except Exception:
        return []

    if not image_path or (not os.path.exists(image_path)):
        return []
    os.makedirs(out_dir, exist_ok=True)

    img = cv2.imread(image_path)
    if img is None:
        return []

    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    edges = cv2.Canny(gray, 40, 120)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for cnt in contours:
        x, y, bw, bh = cv2.boundingRect(cnt)
        area = bw * bh
        if area < min_area or area > max_area:
            continue
        if bw < min_size or bh < min_size or bw > max_size or bh > max_size:
            continue
        ar = bw / float(bh + 1e-6)
        if ar < aspect_min or ar > aspect_max:
            continue
        # 过滤太靠边的大块（通常是黑�?遮罩�?
        if x <= 1 or y <= 1 or (x + bw) >= (w - 1) or (y + bh) >= (h - 1):
            continue
        boxes.append((x, y, bw, bh))

    # 去重：按面积从大到小，做一个简单的 IoU 抑制
    boxes.sort(key=lambda b: b[2] * b[3], reverse=True)
    kept = []
    def _iou(a, b):
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        x1 = max(ax, bx); y1 = max(ay, by)
        x2 = min(ax + aw, bx + bw); y2 = min(ay + ah, by + bh)
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        if inter <= 0:
            return 0.0
        ua = aw * ah + bw * bh - inter
        return inter / float(ua + 1e-6)

    for b in boxes:
        if len(kept) >= max_items:
            break
        overlapped = False
        for k in kept:
            if _iou(b, k) > 0.35:
                overlapped = True
                break
        if not overlapped:
            kept.append(b)

    ts = int(time.time() * 1000)
    saved = []
    for idx, (x, y, bw, bh) in enumerate(kept):
        px1 = max(0, x - padding)
        py1 = max(0, y - padding)
        px2 = min(w, x + bw + padding)
        py2 = min(h, y + bh + padding)
        crop = img[py1:py2, px1:px2]
        if crop is None or crop.size == 0:
            continue
        fn = f"tpl_ui_{ts}_{idx:03d}.png"
        out_path = os.path.join(out_dir, fn)
        cv2.imwrite(out_path, crop)
        saved.append(out_path)
    return saved


def capture_and_extract_ui_candidates(serial: str, *,
                                      out_subdir: str = "ui_candidates",
                                      max_items: int = 120) -> dict:
    """连接设备 -> 截图 -> 提取 UI 候�?-> 保存�?scripts/ui_candidates/�?
    返回：{"screenshot": path, "candidates": [paths...] }
    """
    result = {"screenshot": None, "candidates": []}
    if not serial:
        return result

    dev = None
    try:
        dev = connect_device(f"android:///{serial}?cap_method=ADBCAP&touch_method=ADBTOUCH")
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        shot_name = f"ui_capture_{serial.replace(':','_')}_{ts}.png"
        shot_path = os.path.join(OUTPUT_DIR, shot_name)
        dev.snapshot(filename=shot_path)
        result["screenshot"] = shot_path

        out_dir = os.path.join(TEMPLATE_DIR, out_subdir)
        paths = extract_ui_candidates_from_image(shot_path, out_dir, max_items=max_items)
        result["candidates"] = paths

        # 自动把“新文件名”写�?ui_kb（空 label），方便前端立即显示为“待标注�?
        kb = load_ui_kb()
        for p in paths:
            fn = os.path.basename(p)
            rel = os.path.join(out_subdir, fn)
            if rel not in kb:
                kb[rel] = {"label": "", "tags": ["candidate"], "notes": ""}
        save_ui_kb(kb)

        # 也可以选择把候选自动注册进 image_templates.json（最小字段）
        # 这里默认不自动注册，避免污染已有模板库；等你标注后再入库�?

        return result
    except Exception:
        traceback.print_exc()
        return result
    finally:
        if dev:
            try:
                dev.stop_app(PACKAGE_NAME)
            except Exception:
                pass


def extract_ui_candidates_with_boxes_from_image(image_path: str, out_dir: str, *,
                                                min_size: int = 36, max_size: int = 220,
                                                min_area: int = 36 * 36,
                                                max_area: int = 260 * 260,
                                                aspect_min: float = 0.55,
                                                aspect_max: float = 1.80,
                                                padding: int = 6,
                                                max_items: int = 80) -> list:
    """�?extract_ui_candidates_from_image 类似，但会返�?bbox/center 信息�?

    返回：[{"filename": "ui_candidates/xxx.png", "bbox": [x,y,w,h], "center": [cx,cy]}]
    """
    try:
        import cv2
    except Exception:
        return []

    if not image_path or (not os.path.exists(image_path)):
        return []
    os.makedirs(out_dir, exist_ok=True)

    img = cv2.imread(image_path)
    if img is None:
        return []

    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(gray, 40, 120)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for cnt in contours:
        x, y, bw, bh = cv2.boundingRect(cnt)
        area = bw * bh
        if area < min_area or area > max_area:
            continue
        if bw < min_size or bh < min_size or bw > max_size or bh > max_size:
            continue
        ar = bw / float(bh + 1e-6)
        if ar < aspect_min or ar > aspect_max:
            continue
        if x <= 1 or y <= 1 or (x + bw) >= (w - 1) or (y + bh) >= (h - 1):
            continue
        boxes.append((x, y, bw, bh))

    boxes.sort(key=lambda b: b[2] * b[3], reverse=True)
    kept = []
    def _iou(a, b):
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        x1 = max(ax, bx); y1 = max(ay, by)
        x2 = min(ax + aw, bx + bw); y2 = min(ay + ah, by + bh)
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        if inter <= 0:
            return 0.0
        ua = aw * ah + bw * bh - inter
        return inter / float(ua + 1e-6)

    for b in boxes:
        if len(kept) >= max_items:
            break
        if any(_iou(b, k) > 0.35 for k in kept):
            continue
        kept.append(b)

    ts = int(time.time() * 1000)
    out = []
    for idx, (x, y, bw, bh) in enumerate(kept):
        px1 = max(0, x - padding)
        py1 = max(0, y - padding)
        px2 = min(w, x + bw + padding)
        py2 = min(h, y + bh + padding)
        crop = img[py1:py2, px1:px2]
        if crop is None or crop.size == 0:
            continue
        fn = f"tpl_ui_{ts}_{idx:03d}.png"
        out_path = os.path.join(out_dir, fn)
        cv2.imwrite(out_path, crop)
        cx = int(x + bw / 2)
        cy = int(y + bh / 2)
        out.append({
            "filename": fn,
            "bbox": [int(x), int(y), int(bw), int(bh)],
            "center": [cx, cy],
        })
    return out


def detect_ui_state(dev, coords=None, *, min_hits: int = 2, per_anchor_timeout: float = 0.2) -> dict:
    """根据 ui_states.json �?anchors 判断当前界面状态�?

    返回：{"state": str|None, "hits": int, "detail": {state: hits}}
    """
    db = load_ui_states()
    if not db:
        return {"state": None, "hits": 0, "detail": {}}

    detail = {}
    best_state = None
    best_hits = 0
    for state, meta in db.items():
        anchors = meta.get("anchors", []) if isinstance(meta, dict) else []
        optional = meta.get("optional_anchors", []) if isinstance(meta, dict) else []
        hits = 0

        def _hit_one(a: str) -> bool:
            if not a:
                return False
            # anchor 既支�?template_key，也支持 label（会解析�?template_keys�?
            keys = [a] if a in IMAGE_TEMPLATES else resolve_template_keys(a)
            if not keys:
                keys = [a]
            for k in keys:
                tpl = IMAGE_TEMPLATES.get(k)
                if not tpl:
                    continue
                try:
                    if exists(tpl, timeout=per_anchor_timeout):
                        return True
                except Exception:
                    continue
            return False

        for a in anchors:
            if _hit_one(str(a).strip()):
                hits += 1
        for a in optional:
            if _hit_one(str(a).strip()):
                hits += 1

        detail[state] = hits
        if hits > best_hits:
            best_hits = hits
            best_state = state

    if best_hits < max(1, int(min_hits)):
        return {"state": None, "hits": best_hits, "detail": detail}
    return {"state": best_state, "hits": best_hits, "detail": detail}


def explore_step(serial: str, *,
                 from_state: str = None,
                 candidate_subdir: str = "ui_candidates",
                 max_candidates: int = 60,
                 settle_wait: float = 1.0,
                 diff_threshold: float = 2.2,
                 rollback: dict = None) -> dict:
    """自动探索一步：截图 -> 候选提�?-> 随机点一�?-> 截图 -> 生成 pending 记录�?

    返回：pending 记录（已写入 data/ui_pending.json）�?
    """
    if not serial:
        return {"ok": False, "msg": "serial required"}

    dev = None
    try:
        dev = connect_device(f"android:///{serial}?cap_method=ADBCAP&touch_method=ADBTOUCH")
        model = get_device_model_name(serial)
        coords = get_device_coords_by_model(model)

        # 若调用方没给 from_state，则自动识别
        if not from_state:
            st = detect_ui_state(dev, coords)
            from_state = st.get("state")

        day = datetime.datetime.now().strftime("%Y-%m-%d")
        exp_dir = os.path.join(OUTPUT_DIR, "explore", day, serial.replace(':', '_'))
        os.makedirs(exp_dir, exist_ok=True)

        rid = f"exp_{int(time.time()*1000)}"
        before_rel = os.path.join("explore", day, serial.replace(':', '_'), f"{rid}_before.png")
        after_rel = os.path.join("explore", day, serial.replace(':', '_'), f"{rid}_after.png")
        before_path = os.path.join(OUTPUT_DIR, before_rel)
        after_path = os.path.join(OUTPUT_DIR, after_rel)

        dev.snapshot(filename=before_path)

        out_dir = os.path.join(TEMPLATE_DIR, candidate_subdir)
        cand = extract_ui_candidates_with_boxes_from_image(before_path, out_dir, max_items=max_candidates)
        if not cand:
            rec = {
                "id": rid,
                "ts": int(time.time()),
                "serial": serial,
                "from_state": from_state,
                "status": "no_candidates",
                "before": before_rel,
            }
            return _append_pending(rec)

        # 选一个“没探索过”的候选：�?filename 去重
        pending = load_ui_pending()
        tried = set()
        for r in pending:
            if isinstance(r, dict) and r.get("from_state") == from_state and r.get("candidate", {}).get("filename"):
                tried.add(r["candidate"]["filename"])

        pick = None
        for c in cand:
            if c.get("filename") not in tried:
                pick = c
                break
        if pick is None:
            pick = cand[0]

        cx, cy = pick.get("center", [None, None])
        if cx is None or cy is None:
            cx, cy = cand[0].get("center", [0, 0])

        dev.touch((cx, cy))
        time.sleep(max(0.2, float(settle_wait)))
        dev.snapshot(filename=after_path)

        diff = _img_mean_abs_diff(before_path, after_path)
        changed = diff >= float(diff_threshold)

        # 尝试识别 next_state
        st2 = detect_ui_state(dev, coords)
        to_state_guess = st2.get("state")

        rec = {
            "id": rid,
            "ts": int(time.time()),
            "serial": serial,
            "from_state": from_state,
            "to_state_guess": to_state_guess,
            "before": before_rel,
            "after": after_rel,
            "diff": diff,
            "changed": bool(changed),
            "candidate": {
                "dir": candidate_subdir,
                "filename": pick.get("filename"),
                "bbox": pick.get("bbox"),
                "center": pick.get("center"),
            },
            "commit": {
                "ui_label": "",
                "to_state": "",
                "anchors": [],
                "register": False,
                "rollback": rollback or {"type": "keyevent", "value": "BACK", "timeout": 3},
            },
            "status": "pending",
        }
        return _append_pending(rec)
    except Exception as e:
        return {"ok": False, "msg": str(e)}
    finally:
        if dev:
            try:
                dev.stop_app(PACKAGE_NAME)
            except Exception:
                pass



def explore_all(serial: str, *,
                from_state: str = None,
                candidate_subdir: str = "ui_candidates",
                max_candidates: int = 180,
                settle_wait: float = 1.0,
                diff_threshold: float = 2.2,
                rollback: dict = None,
                max_steps: int = 120) -> dict:
    """自动探索全部：对当前界面提取的候选点逐个点击并生�?pending�?

    - from_state 为空时会自动识别
    - 每次点击后会截图 before/after 并写�?data/ui_pending.json
    - 每轮点击后会尝试回退�?from_state：优先清弹窗 -> 点通用“返回”模�?-> 尝试�?X/关闭 -> KEYEVENT BACK

    返回：summary dict（created/total/reason/...�?
    """
    if not serial:
        return {"ok": False, "msg": "serial required"}

    dev = None
    try:
        dev = connect_device(serial)
        model = get_device_model_name(serial)
        coords = get_device_coords_by_model(model)

        svc = {
            "OUTPUT_DIR": OUTPUT_DIR,
            "TEMPLATE_DIR": TEMPLATE_DIR,
            "load_ui_pending": load_ui_pending,
            "_append_pending": _append_pending,
            "_img_mean_abs_diff": _img_mean_abs_diff,
            "detect_ui_state": detect_ui_state,
            "extract_ui_candidates_with_boxes_from_image": extract_ui_candidates_with_boxes_from_image,
            "ui_click": ui_click,
            "clear_all_popups": clear_all_popups,
            "simple_wait": simple_wait,
        }

        return ui_explore.explore_all_on_device(
            dev=dev,
            serial=serial,
            coords=coords,
            from_state=from_state,
            svc=svc,
            candidate_subdir=candidate_subdir,
            max_candidates=max_candidates,
            settle_wait=settle_wait,
            diff_threshold=diff_threshold,
            rollback=rollback,
            max_steps=max_steps,
            device_alias=model or "",
        )
    except Exception as e:
        return {"ok": False, "msg": f"explore_all failed: {e}"}
    finally:
        # explore 不要 stop app；只做连接清理（�?connect_device 内部策略决定�?
        try:
            pass
        except Exception:
            pass

def commit_pending_transition(pending_id: str, *,
                             ui_label: str,
                             to_state: str,
                             anchors: list = None,
                             register: bool = False,
                             rollback: dict = None) -> dict:
    """�?pending 记录固化为“图谱边”（ui_graph.json�? 状态定义（ui_states.json�? 字典（ui_kb）�?""
    if not pending_id:
        return {"ok": False, "msg": "pending_id required"}

    ui_label = str(ui_label or "").strip()
    to_state = str(to_state or "").strip()
    anchors = anchors if isinstance(anchors, list) else []
    rollback = rollback if isinstance(rollback, dict) else {"type": "keyevent", "value": "BACK", "timeout": 3}

    pending = load_ui_pending()
    idx = None
    rec = None
    for i, r in enumerate(pending):
        if isinstance(r, dict) and str(r.get("id")) == str(pending_id):
            idx = i
            rec = r
            break
    if rec is None:
        return {"ok": False, "msg": "pending not found"}

    from_state = str(rec.get("from_state") or "").strip()
    cand = rec.get("candidate", {}) if isinstance(rec.get("candidate", {}), dict) else {}
    cand_fn = str(cand.get("filename") or "").strip()  # 形如 ui_candidates/xxx.png

    if not ui_label:
        return {"ok": False, "msg": "ui_label required"}
    if not to_state:
        return {"ok": False, "msg": "to_state required"}
    if not from_state:
        from_state = "__unknown__"

    # 1) 写入 ui_kb：candidate filename -> label
    kb = load_ui_kb()
    kb.setdefault(cand_fn, {"label": "", "tags": [], "notes": ""})
    kb[cand_fn]["label"] = ui_label
    if "candidate" not in kb[cand_fn].get("tags", []):
        kb[cand_fn].setdefault("tags", [])
        if isinstance(kb[cand_fn]["tags"], list):
            kb[cand_fn]["tags"].append("candidate")
    save_ui_kb(kb)

    # 2) 可选：入库 image_templates.json（让执行层可以用 label->template_key 点击�?
    if register:
        try:
            ensure_template_registered(cand_fn, template_key=ui_label)
        except Exception:
            pass

    # 3) 写入 ui_states.json：to_state �?anchors
    states = load_ui_states()
    states.setdefault(to_state, {"anchors": [], "optional_anchors": [], "notes": ""})
    if anchors:
        # 简单策略：直接覆盖 anchors（你也可以在前端做“追�?覆盖”开关）
        states[to_state]["anchors"] = [str(a).strip() for a in anchors if str(a).strip()]
    save_ui_states(states)

    # 4) 写入 ui_graph.json：from_state -> action -> to_state
    g = load_ui_graph()
    g.setdefault(from_state, {"actions": []})
    if not isinstance(g[from_state], dict):
        g[from_state] = {"actions": []}
    g[from_state].setdefault("actions", [])

    action = {
        "action_id": pending_id,
        "ui_label": ui_label,
        "candidate_filename": cand_fn,
        "to_state": to_state,
        "verify": {"type": "state", "value": to_state, "timeout": 6},
        "rollback": rollback,
    }
    # 去重：同一�?from_state + ui_label + to_state 只留一�?
    new_actions = []
    for a in g[from_state].get("actions", []):
        if not isinstance(a, dict):
            continue
        if str(a.get("ui_label")) == ui_label and str(a.get("to_state")) == to_state:
            continue
        new_actions.append(a)
    new_actions.append(action)
    g[from_state]["actions"] = new_actions
    save_ui_graph(g)

    # 5) 更新 pending 状�?
    rec["status"] = "committed"
    rec.setdefault("commit", {})
    rec["commit"].update({
        "ui_label": ui_label,
        "to_state": to_state,
        "anchors": anchors,
        "register": bool(register),
        "rollback": rollback,
    })
    pending[idx] = rec
    save_ui_pending(pending)
    return {"ok": True, "record": rec}

def run_single_device_logic(serial, mode, custom_tasks, save_root, apk_path=None):
    prefix = f"[{serial}]"
    dev = None
    try:
        check_stop()
        log_proxy(f"{prefix} 🔌 连接设备...")
        if mode == "install":
            # 装包不需�?minicap/截图能力，避免触�?minicap 安装链路
            dev = connect_device(f"android:///{serial}?cap_method=JAVACAP&touch_method=ADBTOUCH")
        else:
            dev = connect_device(f"android:///{serial}?cap_method=ADBCAP&touch_method=ADBTOUCH")
        model = get_device_model_name(serial)
        coords = get_device_coords_by_model(model)

        if mode == "init":
            _flow_new_package_init(dev, coords, model, serial)
        elif mode == "capture":
            _flow_screenshot_task(dev, coords, model, custom_tasks, save_root, serial)
        elif mode == "install":
            _flow_install_task(dev, model, apk_path)

    except TaskStoppedError:
        log_proxy(f"{prefix} 🛑 任务已停止，清理资源...")
    except Exception as e:
        log_proxy(f"{prefix} �?致命异常: {e}")
        traceback.print_exc()
    finally:
        if dev:
            try:
                dev.stop_app(PACKAGE_NAME)
            except:
                pass

def run_airtest_task(logger_callback, task_mode, custom_tasks=None, target_devices=None, apk_path=None):
    global STOP_FLAG, CURRENT_LOGGER
    STOP_FLAG = False
    CURRENT_LOGGER = logger_callback
    if task_mode == "capture":
        account_manager.reset()

    try:
        dev_list = subprocess.check_output(["adb", "devices"]).decode().strip().split("\n")[1:]
        serials = [s.split("\t")[0] for s in dev_list if "device" in s]
    except:
        serials = []
    if target_devices:
        serials = [s for s in serials if s in target_devices]
    if not serials:
        return log_proxy("�?无可用设�?)

    save_root = ""
    if task_mode == "capture":
        save_root = os.path.join(OUTPUT_DIR, datetime.datetime.now().strftime("%Y-%m-%d"))
        os.makedirs(save_root, exist_ok=True)

    log_proxy(f"🔥 启动 {len(serials)} 个任�?)
    with ThreadPoolExecutor(max_workers=len(serials)) as executor:
        futures = [executor.submit(run_single_device_logic, s, task_mode, custom_tasks, save_root, apk_path) for s in serials]
        for f in futures:
            try:
                f.result()
            except:
                pass
    # �?任务结束后：自动归档超过保留期的历史截图（默认保�?14 天）
    if task_mode == "capture":
        try:
            summary = archive_old_screenshots(keep_days=14, logger=CURRENT_LOGGER)
            if summary.get("success") and summary.get("zip"):
                log_proxy(f"🗜�?已归�?{summary.get('count')} 天截�?-> {summary.get('zip')}")
        except Exception:
            pass

    log_proxy(f"🎉 任务结束")

def stop_task():
    global STOP_FLAG
    STOP_FLAG = True
    try:
        dev_list = subprocess.check_output(["adb", "devices"]).decode().strip().split("\n")[1:]
        for line in dev_list:
            if "device" in line:
                s = line.split("\t")[0]
                subprocess.run(["adb", "-s", s, "shell", "am", "force-stop", PACKAGE_NAME], timeout=2, stdout=subprocess.DEVNULL)
                print(f"[{s}] 🛑 ADB 强停游戏")
    except:
        pass
