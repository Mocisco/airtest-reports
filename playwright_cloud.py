# playwright_cloud.py
# -*- coding: utf-8 -*-

"""
云真机自动占�?释放工具（Playwright + Edge CDP�?

功能�?
- used_devices 同时保存在内存与磁盘（cloud_used_devices.json），用于 Flask/服务重启后的释放兜底
- 释放两条路径�?
  1) release_all_connected(): 精准释放 used_devices 里记录的设备
  2) release_all_mine_from_favorites(): 兜底释放“我的收藏”里所有“自己在使用”的设备
"""

from playwright.sync_api import sync_playwright
import re
import json
import time
import os
import subprocess
from collections import OrderedDict
from typing import Any, Dict, List


# ================= 全局配置 =================
BASE_DIR = os.path.dirname(__file__)
CACHE_FILE = os.path.join(BASE_DIR, "cloud_connected_cache.json")
USED_DEVICES_FILE = os.path.join(BASE_DIR, "cloud_used_devices.json")

COOLDOWN_SECONDS = 40 * 60        # 40 分钟内不重复占用
MAX_WAIT_AFTER_CLICK = 3          # 每次占用后的等待时间

CLOUD_DOMAIN = "cloud.example.com"
CDP_ENDPOINT = "http://127.0.0.1:9222"
CLOUD_HOME_URL = f"http://{CLOUD_DOMAIN}/#/"


class CloudAutoConnector:
    def __init__(self, username: str, password: str, headless: bool = False, progress_cb=None):
        """云真机自动占�?释放

        progress_cb(step:int, total:int, name:str)
        """
        self.username = username
        self.password = password
        self.headless = headless
        self.progress_cb = progress_cb

        # 本轮占用的设备信息（用于释放�?
        # item: {"url": str, "device_id": str|None, "adb_ip": str|None, "title": str|None}
        self.used_devices: List[Dict[str, Any]] = []

    # ================= ADB 工具 =================
    def _adb_connect_and_verify(self, adb_ip: str, retries: int = 3, wait_s: int = 2) -> bool:
        for i in range(retries):
            try:
                subprocess.run(
                    ["adb", "connect", adb_ip],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=10
                )
                out = subprocess.check_output(["adb", "devices"]).decode(errors="ignore")
                if adb_ip.split(":")[0] in out:
                    print(f"🔗 ADB 已确认连�? {adb_ip}")
                    return True
            except Exception:
                pass

            print(f"�?ADB 连接未就绪，重试 {i+1}/{retries}...")
            time.sleep(wait_s)

        print(f"�?ADB 连接失败，跳过设�? {adb_ip}")
        return False

    # ================= 缓存工具 =================
    def _load_connected_cache(self) -> Dict[str, float]:
        if not os.path.exists(CACHE_FILE):
            return {}
        try:
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def _save_connected_cache(self, cache: Dict[str, float]) -> None:
        try:
            with open(CACHE_FILE, "w", encoding="utf-8") as f:
                json.dump(cache, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def _load_used_devices_cache(self) -> List[Dict[str, Any]]:
        if not os.path.exists(USED_DEVICES_FILE):
            return []
        try:
            with open(USED_DEVICES_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, list) else []
        except Exception:
            return []

    def _save_used_devices_cache(self, devices: List[Dict[str, Any]]) -> None:
        try:
            with open(USED_DEVICES_FILE, "w", encoding="utf-8") as f:
                json.dump(devices, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def _clear_used_devices_cache(self) -> None:
        try:
            if os.path.exists(USED_DEVICES_FILE):
                os.remove(USED_DEVICES_FILE)
        except Exception:
            pass

    def recover_used_devices_from_disk(self) -> int:
        """把磁盘里�?used_devices 合并�?self.used_devices（去重后保序）�?""
        disk = self._load_used_devices_cache()
        if not disk:
            return 0

        uniq: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()

        for item in (self.used_devices or []):
            url = (item or {}).get("url")
            if not url:
                continue
            key = ((item or {}).get("device_id") or url).strip()
            uniq[key] = item

        for item in disk:
            url = (item or {}).get("url")
            if not url:
                continue
            key = ((item or {}).get("device_id") or url).strip()
            uniq[key] = item

        self.used_devices = list(uniq.values())
        return len(self.used_devices)

    # ================= 页面辅助 =================
    def _get_existing_cloudphone_page(self, context):
        for p in context.pages:
            try:
                if p.url and CLOUD_DOMAIN in p.url:
                    print(f"�?复用已打开的页�? {p.url}")
                    p.bring_to_front()
                    return p
            except Exception:
                continue
        return None

    def _get_cloudphone_page(self, context):
        page = self._get_existing_cloudphone_page(context)
        if page:
            return page
        raise Exception("未找到已打开�?cloudphone 页面（请先在 Edge 打开云真机站点）")

    def _goto_my_favorites(self, page):
        print("📂 进入【我的收藏�?..")
        try:
            if CLOUD_DOMAIN not in (page.url or ""):
                page.goto(CLOUD_HOME_URL, wait_until="domcontentloaded")
                page.wait_for_timeout(800)
        except Exception:
            pass

        try:
            page.wait_for_selector("text=我的收藏", timeout=15000)
            page.click("text=我的收藏")
        except Exception:
            pass

        page.wait_for_timeout(1200)

    def _return_home(self, page):
        print("↩️ 返回首页...")
        try:
            if page.locator("text=返回首页").count() > 0:
                page.click("text=返回首页")
            else:
                page.goto(CLOUD_HOME_URL, wait_until="domcontentloaded")
        except Exception:
            try:
                page.goto(CLOUD_HOME_URL, wait_until="domcontentloaded")
            except Exception:
                pass
        page.wait_for_timeout(1200)

    def _scan_available_cards(self, page):
        candidates = page.locator("div:has(.list_bottom .el-button:has-text('立即使用'))")
        available = []
        for i in range(candidates.count()):
            card = candidates.nth(i)
            try:
                txt = card.inner_text()
            except Exception:
                continue

            if ("空闲�? not in txt) and ("自己在使�? not in txt):
                continue

            btns = card.locator(".list_bottom .el-button:has-text('立即使用')")
            if btns.count() != 1:
                continue

            available.append(card)

        return available

    def _enter_device_and_get_adb(self, page) -> str:
        print("📲 进入设备并获�?ADB...")
        page.wait_for_timeout(2500)

        try:
            if page.locator("text=远程调试").count() > 0:
                page.click("text=远程调试", timeout=5000)
        except Exception:
            pass

        page.wait_for_timeout(1500)

        content = page.inner_text("body")
        match = re.search(r"adb connect ([\d\.]+:\d+)", content)
        if not match:
            match = re.search(r"([\d\.]+:\d+)", content)

        if not match:
            try:
                page.screenshot(path=os.path.join(BASE_DIR, "debug_error_adb.png"))
            except Exception:
                pass
            raise Exception("未找�?ADB connect 地址")

        return match.group(1)

    def _click_release_on_page(self, page, scope=None) -> bool:
        base = scope if scope is not None else page

        def _confirm_if_any():
            try:
                confirm_btn = page.locator(
                    ".el-message-box__btns button:has-text('确认'), "
                    ".el-message-box__btns button:has-text('确定'), "
                    "button:has-text('确认'), button:has-text('确定')"
                ).first
                if confirm_btn.count() > 0 and confirm_btn.is_visible():
                    confirm_btn.click()
                    page.wait_for_timeout(800)
            except Exception:
                pass

        try:
            btn = base.locator(
                "button:has-text('释放'), a:has-text('释放'), .el-button:has-text('释放')"
            ).first
            if btn.count() > 0 and btn.is_visible():
                btn.click()
                page.wait_for_timeout(400)
                _confirm_if_any()
                return True
        except Exception:
            pass

        try:
            loc = base.locator("text=释放")
            handles = loc.element_handles()
            if not handles:
                return False

            try:
                vh = page.evaluate("() => window.innerHeight") or 900
            except Exception:
                vh = 900

            best = None
            best_score = None
            for h in handles:
                try:
                    box = h.bounding_box()
                    if not box:
                        continue
                    x, y = box.get("x", 9999), box.get("y", 0)

                    if x > 260:
                        continue
                    if y < vh * 0.55:
                        continue

                    score = (x * 2) - y
                    if best is None or score < best_score:
                        best = h
                        best_score = score
                except Exception:
                    continue

            if best is None:
                return False

            best.click()
            page.wait_for_timeout(400)
            _confirm_if_any()
            return True
        except Exception:
            return False

    # ================= 自动占用 =================
    def auto_from_favorites(self, max_devices: int = 5) -> Dict[str, Any]:
        connected_adbs: List[str] = []
        cache = self._load_connected_cache()
        now = time.time()

        try:
            with sync_playwright() as p:
                print("🔗 [CDP] 正在接管已运行的 Edge...")
                browser = p.chromium.connect_over_cdp(CDP_ENDPOINT)

                if not browser.contexts:
                    return {"success": False, "msg": "未发现浏览器 Context"}

                context = browser.contexts[0]
                page = self._get_cloudphone_page(context)

                page.wait_for_selector("text=我的收藏", timeout=20000)

                while len(connected_adbs) < max_devices:
                    self._goto_my_favorites(page)

                    cards = self._scan_available_cards(page)
                    if not cards:
                        print("⚠️ 没有可用设备，结�?)
                        break

                    picked_card = None
                    picked_key = None

                    for card in cards:
                        title = card.inner_text().splitlines()[0].strip()
                        last_time = cache.get(title)

                        if last_time and now - last_time < COOLDOWN_SECONDS:
                            continue

                        picked_card = card
                        picked_key = title
                        break

                    if not picked_card:
                        print("⚠️ 本轮无新的可用设�?)
                        break

                    print(f"�?占用设备: {picked_key}")

                    current_step = len(connected_adbs) + 1
                    if self.progress_cb:
                        self.progress_cb(current_step, max_devices, picked_key)

                    btn = picked_card.locator(".list_bottom .el-button:has-text('立即使用')")
                    btn.wait_for(state="visible", timeout=8000)
                    btn.click()

                    adb_ip = self._enter_device_and_get_adb(page)

                    device_id = None
                    m = re.search(r"/androidscreen/([^/]+)/", page.url or "")
                    if m:
                        device_id = m.group(1)

                    self.used_devices.append({
                        "url": page.url,
                        "device_id": device_id,
                        "adb_ip": adb_ip,
                        "title": picked_key
                    })

                    self._save_used_devices_cache(self.used_devices)

                    if self._adb_connect_and_verify(adb_ip):
                        connected_adbs.append(adb_ip)
                    else:
                        print(f"⚠️ 跳过无法连接的设�? {adb_ip}")

                    cache[picked_key] = time.time()
                    self._save_connected_cache(cache)

                    self._return_home(page)
                    time.sleep(MAX_WAIT_AFTER_CLICK)

                return {"success": True, "adbs": connected_adbs}

        except Exception as e:
            print(f"�?[CDP Error] {e}")
            return {"success": False, "msg": str(e)}

    # ================= 自动释放（精准） =================
    def release_all_connected(self) -> None:
        """
        精准释放：按 used_devices（去重后保序）逐台释放�?
        关键点：
        - 所�?Playwright Page 操作必须�?`with sync_playwright()` 作用域内完成，否则会出现
          "Event loop is closed! Is Playwright already stopped?"
        - 释放开始前先回到首页，避免“当前打开的设备页”影响释放顺�?
        """
        if not self.used_devices:
            self.recover_used_devices_from_disk()
        if not self.used_devices:
            print("ℹ️ 本轮没有需要释放的设备")
            return

        # 去重后保序（确保释放顺序稳定�?
        uniq: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
        for item in self.used_devices:
            url = (item or {}).get("url")
            if not url:
                continue
            key = ((item or {}).get("device_id") or url).strip()
            uniq[key] = item

        devices = list(uniq.values())
        if not devices:
            print("ℹ️ 本轮没有需要释放的设备")
            return

        with sync_playwright() as p:
            print("♻️ 开始释放本轮占用的云真�?..")
            browser = p.chromium.connect_over_cdp(CDP_ENDPOINT)
            if not browser.contexts:
                raise Exception("未发现浏览器 Context")

            context = browser.contexts[0]
            page = self._get_cloudphone_page(context)

            # �?先回首页，避免当前正停在某个设备页导致“先释放当前页设备�?
            try:
                page.goto(CLOUD_HOME_URL, wait_until="domcontentloaded")
                page.wait_for_timeout(1200)
            except Exception:
                try:
                    self._return_home(page)
                except Exception:
                    pass

            for item in devices:
                url = item.get("url")
                device_id = item.get("device_id")
                adb_ip = item.get("adb_ip")

                print(f"♻️ 释放设备: {url}")
                released = False

                # 1) 优先在设备页释放
                try:
                    page.goto(url, wait_until="domcontentloaded")
                    page.wait_for_timeout(1200)

                    # �?二次确认：确保确实切到目标设备页
                    if url and url not in (page.url or ""):
                        print(f"⚠️ 页面未正确切换到目标设备，重试一�? {url}")
                        page.goto(url, wait_until="domcontentloaded")
                        page.wait_for_timeout(1200)

                    released = self._click_release_on_page(page)
                except Exception as e:
                    print(f"⚠️ 设备页释放异�? {e}")

                # 2) 兜底：收藏页卡片释放
                if not released:
                    try:
                        self._goto_my_favorites(page)

                        candidates = []
                        if device_id:
                            candidates.append(page.locator(f"div:has-text('{device_id}')"))
                        if adb_ip:
                            candidates.append(page.locator(f"div:has-text('{adb_ip}')"))
                            candidates.append(page.locator(f"div:has-text('{adb_ip.split(':')[0]}')"))

                        clicked = False
                        for loc in candidates:
                            if loc.count() <= 0:
                                continue
                            for i in range(min(loc.count(), 30)):
                                card = loc.nth(i)
                                if self._click_release_on_page(page, scope=card):
                                    clicked = True
                                    break
                            if clicked:
                                break

                        released = clicked
                    except Exception as e:
                        print(f"⚠️ 收藏页释放异�? {e}")

                if released:
                    print("�?已触发释�?)
                else:
                    print(f"⚠️ 未找到可点击的释放入�?(device_id={device_id}, adb={adb_ip})（页面结构可能变更）")

                # 回首页再继续下一台，降低页面状态干�?
                try:
                    page.goto(CLOUD_HOME_URL, wait_until="domcontentloaded")
                    page.wait_for_timeout(1200)
                except Exception:
                    try:
                        self._return_home(page)
                    except Exception:
                        pass

                time.sleep(1)

        self.used_devices.clear()
        self._clear_used_devices_cache()
        print("�?本轮云真机已全部释放")
    # ================= 自动释放（兜底） =================
    def release_all_mine_from_favorites(self, max_release: int = 20) -> int:
        released_count = 0

        with sync_playwright() as p:
            print("♻️ 兜底释放：开始在【我的收藏】释放“自己在使用”的云真�?..")
            browser = p.chromium.connect_over_cdp(CDP_ENDPOINT)
            if not browser.contexts:
                raise Exception("未发现浏览器 Context")

            context = browser.contexts[0]
            page = self._get_cloudphone_page(context)

            for _ in range(max_release):
                self._goto_my_favorites(page)

                mine_cards = page.locator("div:has-text('自己在使�?)").filter(
                    has=page.locator(".list_bottom .el-button:has-text('立即使用')")
                )
                if mine_cards.count() <= 0:
                    break

                card = mine_cards.first

                if self._click_release_on_page(page, scope=card):
                    released_count += 1
                    print(f"�?已触发释放（卡片页）[{released_count}]")
                    page.wait_for_timeout(1200)
                    continue

                try:
                    use_btn = card.locator(
                        ".list_bottom .el-button:has-text('立即使用'), button:has-text('立即使用')"
                    ).first
                    if use_btn.count() > 0 and use_btn.is_visible():
                        use_btn.click()
                        try:
                            _ = self._enter_device_and_get_adb(page)
                        except Exception:
                            pass

                        ok = self._click_release_on_page(page)
                        if ok:
                            released_count += 1
                            print(f"�?已触发释放（设备页）[{released_count}]")
                        else:
                            print("⚠️ 进入设备页但未找到可点击的释放入口（页面结构可能变更�?)

                        try:
                            page.goto(CLOUD_HOME_URL, wait_until="domcontentloaded")
                            page.wait_for_timeout(1200)
                        except Exception:
                            pass

                        time.sleep(0.8)
                        continue
                except Exception as e:
                    print(f"⚠️ 兜底释放：卡片进入设备页异常: {e}")

                print("⚠️ 兜底释放：该卡片无法释放/进入，停止继续扫描以避免循环")
                break

        return released_count
