import os
import time
import subprocess
from DrissionPage import ChromiumPage

# ================= ⚙️ 配置区域 =================
TARGETS = ["HelloGameBox_CN", "HelloGameBox"]
LOOP_DELAY = 5 
DEBUG_PORT = 9222

# ================= 🛠️ 核心类 =================

class XMonitor:
    def __init__(self):
        self.chrome_path = self._find_chrome()
        self.profile_path = self._get_clean_profile()
        self.memory = {user: None for user in TARGETS}
        self.tabs = {} 

    def _find_chrome(self):
        paths = [
            r"C:\Program Files\Google\Chrome\Application\chrome.exe",
            r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
            os.path.join(os.path.expanduser("~"), r"AppData\Local\Google\Chrome\Application\chrome.exe")
        ]
        for p in paths:
            if os.path.exists(p): return p
        return None

    def _get_clean_profile(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(current_dir, "Chrome_Binance_Profile")
        if not os.path.exists(path): os.makedirs(path)
        return path

    def launch(self):
        """启动浏览器"""
        if not self.chrome_path:
            print("❌ 找不到 Chrome")
            return False
        
        print(f"🚀 1. 启动浏览器...")
        # 先打开空白页，防止干扰
        cmd = f'"{self.chrome_path}" --remote-debugging-port={DEBUG_PORT} --user-data-dir="{self.profile_path}" "about:blank"'
        subprocess.Popen(cmd, shell=True)
        
        print("⏳ 等待初始化 (5秒)...")
        time.sleep(5)
        return True

    def init_tabs(self, page):
        """初始化标签页"""
        print("📑 2. 建立监控标签...")
        
        # 拿到第一个标签页，监控第一个博主
        tab1 = page.latest_tab
        tab1.get(f"https://x.com/{TARGETS[0]}")
        self.tabs[TARGETS[0]] = tab1
        print(f"   -> Tab 1: {TARGETS[0]}")
        
        # 新建标签页，监控第二个博主
        tab2 = page.new_tab(f"https://x.com/{TARGETS[1]}")
        self.tabs[TARGETS[1]] = tab2
        print(f"   -> Tab 2: {TARGETS[1]}")
        
        print("✅ 标签页就绪，准备开始循环...")
        time.sleep(3)

    def start_loop(self):
        try:
            page = ChromiumPage(addr_or_opts=DEBUG_PORT)
        except:
            print("❌ 连接失败，请确认浏览器已启动")
            return

        self.init_tabs(page)
        
        print(f"🔥 3. 持续监控中 (双线程轮询)...")
        
        while True:
            for user in TARGETS:
                try:
                    # 获取该博主的 Tab 对象
                    tab = self.tabs[user]
                    
                    # === 核心修正：直接操作 Tab，不调用 activate ===
                    self._check_user_in_tab(tab, user)
                    
                except Exception as e:
                    # 捕获错误但不退出，保证持续监控
                    print(f"❌ [{user}] 轮询跳过: {e}")

            if LOOP_DELAY > 0:
                print(f"⏳ 休息 {LOOP_DELAY} 秒...")
                time.sleep(LOOP_DELAY)

    def _check_user_in_tab(self, tab, user):
        # 刷新该标签页
        tab.refresh()
        
        # 等待推文加载 (10秒超时)
        if not tab.wait.ele_displayed('xpath://article[@data-testid="tweet"]', timeout=10):
            print(f"⚠️ [{user}] 加载超时，网络慢？")
            return

        articles = tab.eles('xpath://article[@data-testid="tweet"]')
        if not articles: return

        # === 排除置顶逻辑 ===
        target_tweet = articles[0]
        raw_text = target_tweet.text
        
        if "Pinned" in raw_text or "置顶" in raw_text:
            if len(articles) > 1:
                target_tweet = articles[1]
            else:
                return 

        # === 获取唯一指纹 (优先用时间戳) ===
        try:
            time_ele = target_tweet.ele('tag:time', timeout=2)
            if time_ele:
                fingerprint = time_ele.attr('datetime')
            else:
                fingerprint = raw_text[:30].replace('\n', '')
        except:
            fingerprint = raw_text[:30].replace('\n', '')

        # === 对比逻辑 ===
        last_seen = self.memory[user]

        if last_seen is None:
            self.memory[user] = fingerprint
            print(f"🔒 [{user}] 基准已记录")
        
        elif fingerprint != last_seen:
            print(f"\n🚨🚨🚨 [{user}] 发新推文了！")
            print(f"📄 ID: {fingerprint}")
            
            # 确保元素在视野内 (DrissionPage 会自动滚动)
            target_tweet.scroll.to_see()
            
            # 检查是否已赞
            if target_tweet.ele('xpath:.//*[@data-testid="unlike"]', timeout=1):
                print(f"⚠️ 已赞过")
            else:
                # 寻找点赞按钮 (万能匹配)
                like_btn = target_tweet.ele('xpath:.//*[@data-testid="like"]', timeout=2)
                if like_btn:
                    like_btn.click()
                    print(f"👍 {user} -> 秒赞成功！")
            
            # 更新内存状态
            self.memory[user] = fingerprint
        else:
            print(f"💤 [{user}] 无更新")

if __name__ == "__main__":
    bot = XMonitor()
    print("⚠️  请先关闭所有旧的 Chrome 窗口！")
    time.sleep(2)
    
    if bot.launch():
        bot.start_loop()
