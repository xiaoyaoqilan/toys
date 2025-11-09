import os
import time
from telegram import Update
from telegram.ext import Application, ContextTypes, CommandHandler, JobQueue
from google import genai
from google.genai.errors import APIError

# --- 1. 配置 ---
# 使用您的第一个 Bot Token (Bot @Aawud1Bot 的 Token)
BOT_TOKEN = "7925102538:AAF4hBmaKYcxPgWimF2I-HiYmGFMlltIZQ0" 

# !!! 必填：替换为您 Bot 所在的 Telegram 群组 ID（必须是负数）
TARGET_GROUP_ID = -1001234567890 
DIALOGUE_INTERVAL_SECONDS = 300  # 每 5 分钟（300秒）进行一轮对话

# 确保在环境变量中设置了 GEMINI_API_KEY
try:
    client = genai.Client()
except Exception as e:
    print(f"Error initializing Gemini client: {e}")
    client = None

# --- 2. 角色定义和会话初始化 ---
# 两个 AI 角色由 Gemini API 独立管理，不需要第二个 Bot Token。
SYSTEM_PROMPT_A = "你是一位乐观、富有远见的人工智能专家（Agent A）。你的目标是向 Agent B 介绍 AI 在医疗领域的前景和挑战。回复保持简短，不超过 80 字，并以一个问题结束，将对话权交给 Agent B。"
SYSTEM_PROMPT_B = "你是一位谨慎、注重伦理和安全的伦理学家（Agent B）。你的目标是针对 Agent A 提出的每一个观点，提出相关的伦理、隐私和安全性质疑。回复保持简短，不超过 80 字，并以一个问题结束，将对话权交给 Agent A。"

if client:
    AGENT_A_CHAT = client.chats.create(model='gemini-2.5-flash', system_instruction=SYSTEM_PROMPT_A)
    AGENT_B_CHAT = client.chats.create(model='gemini-2.5-flash', system_instruction=SYSTEM_PROMPT_B)
else:
    AGENT_A_CHAT = None
    AGENT_B_CHAT = None

DIALOGUE_STATE = {
    'last_message': "我们来讨论一下 AI 在诊断领域的最新突破，你对它的准确性和速度有什么看法？",
    'next_speaker': 'A' 
}

# --- 3. 定时执行任务函数 (run_dialogue_turn) ---
# ... (此函数内容保持不变) ...
async def run_dialogue_turn(context: ContextTypes.DEFAULT_TYPE):
    # 此处省略函数细节，与之前提供的一致
    global DIALOGUE_STATE
    
    if not AGENT_A_CHAT or not AGENT_B_CHAT:
        await context.bot.send_message(TARGET_GROUP_ID, "【系统错误】：AI 客户端未初始化。请检查 GEMINI_API_KEY。")
        return

    current_speaker = DIALOGUE_STATE['next_speaker']
    last_message = DIALOGUE_STATE['last_message']
    
    # 确定当前发言的 AI 代理
    if current_speaker == 'A':
        chat = AGENT_A_CHAT
        speaker_name = "🟢 Agent A (专家)"
        DIALOGUE_STATE['next_speaker'] = 'B'
    else:
        chat = AGENT_B_CHAT
        speaker_name = "🔴 Agent B (伦理学家)"
        DIALOGUE_STATE['next_speaker'] = 'A'

    # 调用 AI API 获取回复
    try:
        response = chat.send_message(last_message)
        new_message_text = response.text
        
        DIALOGUE_STATE['last_message'] = new_message_text
        
        formatted_message = (
            f"**{speaker_name} 说:**\n"
            f"{new_message_text}"
        )
        
        await context.bot.send_message(
            chat_id=TARGET_GROUP_ID,
            text=formatted_message,
            parse_mode='Markdown'
        )

    except APIError as e:
        error_msg = f"对话中断，API 错误: {e}"
        print(error_msg)
        await context.bot.send_message(chat_id=TARGET_GROUP_ID, text=f"【系统错误】：{error_msg}")
    except Exception as e:
        error_msg = f"发生未预期的错误: {e}"
        print(error_msg)
        await context.bot.send_message(chat_id=TARGET_GROUP_ID, text=f"【系统错误】：{error_msg}")


# --- 4. Bot 启动和控制命令 (start, stop, main 函数保持不变) ---
# ... (此处省略 start, stop, main 函数细节，与之前提供的一致) ...
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_chat.id != TARGET_GROUP_ID:
        await update.message.reply_text("此命令只能在指定的对话群组中使用。")
        return
    if 'dialogue_job' in context.job_queue.jobs():
        await update.message.reply_text("AI 对话任务已在运行中。")
        return
    context.job_queue.run_repeating(
        run_dialogue_turn, 
        interval=DIALOGUE_INTERVAL_SECONDS, 
        first=5, 
        chat_id=TARGET_GROUP_ID, 
        name='dialogue_job'
    )
    await update.message.reply_text(f"已成功启动 AI 自动对话，每 {DIALOGUE_INTERVAL_SECONDS} 秒发言一次。")

async def stop(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_chat.id != TARGET_GROUP_ID:
        await update.message.reply_text("此命令只能在指定的对话群组中使用。")
        return
    current_jobs = context.job_queue.get_jobs_by_name('dialogue_job')
    if not current_jobs:
        await update.message.reply_text("当前没有正在运行的 AI 对话任务。")
        return
    for job in current_jobs:
        job.schedule_removal()
    await update.message.reply_text("已成功停止 AI 自动对话任务。")

def main():
    application = Application.builder().token(BOT_TOKEN).build()
    application.add_handler(CommandHandler("start_dialogue", start))
    application.add_handler(CommandHandler("stop_dialogue", stop))
    print("Bot 正在启动...")
    application.run_polling(poll_interval=3)

if __name__ == '__main__':
    # 最终检查
    if TARGET_GROUP_ID == -1001234567890:
        print("FATAL: 请务必替换代码中的 TARGET_GROUP_ID 为您的实际群组 ID！")
    elif not os.getenv("GEMINI_API_KEY"):
         print("FATAL: GEMINI_API_KEY 环境变量未设置。请先设置该环境变量。")
    else:
        main()