import streamlit as st
import time
import os 
import base64 
from io import BytesIO 
from typing import List, Dict, Any

# 尝试导入 OpenAI
try:
    from openai import OpenAI
except ImportError:
    st.error("请先安装 openai 库: pip install openai")
    OpenAI = None

# ==========================================
# 1. 页面配置与样式
# ==========================================
st.set_page_config(
    page_title="聊天助手 (Chat Coach Pro)",
    page_icon="🤖",
    layout="centered"
)

st.markdown("""
<style>
    /* 统一解析器的样式 */
    .coach-note {
        background-color: #f7f9fc;
        border-left: 4px solid #4b6cb7;
        padding: 12px;
        border-radius: 4px;
        font-size: 0.9em;
        color: #333;
        margin-top: 8px;
    }
    /* 恋爱模式的颜色 (模式一) */
    .dating-tag { background-color: #e84393; color: white; padding: 2px 6px; border-radius: 4px; font-weight: bold; }
    .dating-reply { border: 1px solid #e84393; padding: 10px; border-radius: 8px; margin-bottom: 5px; }
    /* 销售模式的颜色 (模式二) */
    .sales-tag { background-color: #2d3436; color: white; padding: 2px 6px; border-radius: 4px; font-weight: bold; }
    .sales-reply { border: 1px solid #2d3436; padding: 10px; border-radius: 8px; margin-bottom: 5px; }
    /* 普通模式的颜色 (模式三) */
    .normal-tag { background-color: #008c9e; color: white; padding: 2px 6px; border-radius: 4px; font-weight: bold; }
    .normal-reply { border: 1px solid #008c9e; padding: 10px; border-radius: 8px; margin-bottom: 5px; }
    /* 回帖模式的颜色 (模式四) */
    .reply_post-tag { background-color: #f7931e; color: white; padding: 2px 6px; border-radius: 4px; font-weight: bold; }
    .reply_post-reply { border: 1px solid #f7931e; padding: 10px; border-radius: 8px; margin-bottom: 5px; }
    
    .reply-text {
        font-size: 1.1em;
        font-family: "Microsoft YaHei", sans-serif;
    }
</style>
""", unsafe_allow_html=True)

# 定义模型列表和多模态模型名称
# FIX: 使用用户确认可用的 Qwen 和 DeepSeek 模型
TEXT_MODELS = {
    "推荐 (Qwen2.5)": "Qwen/Qwen2.5-7B-Instruct",
    "DeepSeek-Qwen3": "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
    "Qwen2-7B": "Qwen/Qwen2-7B-Instruct",
}
# FIX: 回退到 Qwen-VL-Chat，因为 Qwen 文本模型可用，它最有可能被修复或找到正确名称
VISION_MODEL = "Qwen/Qwen-VL-Chat"

# ==========================================
# 2. 侧边栏：设置 (API) - Key 持久化与环境变量加载
# ==========================================

# 预定义回调函数 (用于 session_state 保持 Key)
def save_api_key():
    """回调函数：将输入框的值保存到 session_state 的持久化变量中"""
    st.session_state['persisted_api_key'] = st.session_state['api_key_input_key']

# 在侧边栏中添加模型选择逻辑
with st.sidebar:
    st.header("⚙️ 核心设置")

    # 1. 优先尝试从环境变量中读取 Key
    API_KEY_ENV = os.environ.get("SILICONFLOW_API_KEY")

    if 'persisted_api_key' not in st.session_state:
        st.session_state['persisted_api_key'] = ''
    
    if API_KEY_ENV:
        st.success("🔑 API Key 已从环境变量加载。")
        current_api_key = API_KEY_ENV
    else:
        st.warning("🚨 建议通过环境变量设置 Key。当前使用输入框。")
        
        st.text_input(
            "🔑 硅基流动 API Key", 
            type="password",
            value=st.session_state['persisted_api_key'], 
            key="api_key_input_key", 
            on_change=save_api_key
        )
        current_api_key = st.session_state['persisted_api_key']

    st.markdown("---")
    st.subheader("🧠 AI 模型选择")
    
    # 模型选择器
    selected_model_key = st.selectbox(
        "选择文本模型",
        list(TEXT_MODELS.keys()),
        index=0,
        key="model_selector"
    )
    current_text_model = TEXT_MODELS[selected_model_key]

    st.caption(f"当前文本模型: **{current_text_model}**")
    st.caption(f"图像分析模型: **{VISION_MODEL}** (用于图片分析)")
    
    st.markdown("---")
    st.markdown("### 🎯 产品目标")
    st.info("模式一: 社交匹配。\n\n模式二: 客户转化。\n\n模式三: 日常沟通与内容生成。\n\n模式四: 论坛与社媒互动 (支持图片分析)。") 

# ==========================================
# 3. 核心逻辑函数 (多模式大脑)
# ==========================================

def get_system_prompt(main_mode: str, sub_mode: str, extra_detail: str = "") -> str:
    """
    根据模式生成 LLM 的系统指令。
    :param extra_detail: 普通模式中传入 '关系'，回帖模式中传入 '内容类型'。
    """
    if main_mode == "Dating":
        base_prompt = "你是顶级情感顾问，精通推拉和暧昧。回复必须简短、有趣、带有调侃。"
        if sub_mode == "开场白生成":
            return f"{base_prompt} 请根据对方的简介，生成3条能立刻抓住对方兴趣的开场白。"
        elif sub_mode == "起死回生术":
            return f"{base_prompt} 对方已冷落你很久，请生成3条低压力、高情绪价值的挽回话术。"
    
    elif main_mode == "Sales":
        base_prompt = "你是资深销售专家，回复必须专业、冷静、以客户价值为导向。"
        if sub_mode == "客户异议处理":
            return f"{base_prompt} 客户提出了异议，请生成3种不同策略的回复：1. 价值锚定 2. 情感共鸣 3. 限时优惠。"
        elif sub_mode == "产品介绍文案":
            return f"{base_prompt} 请根据信息，生成3段针对性极强的文案：1. 痛点切入型 2. 权威背书型 3. 使用场景描绘型。"
            
    # ====== 模式三：普通模式 (基于关系) ======
    elif main_mode == "Normal":
        # extra_detail 此时是关系 (朋友, 同事, 陌生人)
        base_prompt = f"你是高情商的万能聊天助手，请以一个面向'{extra_detail}'的身份，根据用户的输入生成回复。"
        return f"{base_prompt} 请生成3条不同侧重点的回复：1. 情感共鸣式 2. 理性分析式 3. 轻松幽默式。请在回复前标明风格。"
            
    # ====== 模式四：回帖模式 (基于内容类型，支持图像描述) ======
    elif main_mode == "Reply_Post":
        # extra_detail 此时是内容类型 (网址, 截图, 视频)
        base_prompt = f"你是一位资深的社媒评论专家，正在针对一个'{extra_detail}'类型的内容进行高质量回复。如果提供了图片，请优先分析图片内容。"
        return f"{base_prompt} 请生成3条高质量的评论或回帖：1. 观点补充与升华型 2. 质疑与引发讨论型 3. 深度总结与归纳型。请在回复前标明风格。"
            
    return "你是高情商助手，生成3条回复和解析。"

def parse_ai_response(text, mode):
    """解析器：根据模式给Tag上色"""
    results: List[Dict[str, str]] = []
    # 尝试将 DeepSeek 模型可能返回的格式转换为我们期望的格式
    text = text.replace("选项一", "\n选项1").replace("选项二", "\n选项2").replace("选项三", "\n选项3")
    parts = text.split("选项") if "选项" in text else [text] 
    
    if mode == "Dating":
        css_class = "dating"
    elif mode == "Sales":
        css_class = "sales"
    elif mode == "Normal":
        css_class = "normal"
    elif mode == "Reply_Post":
        css_class = "reply_post" 
    else:
        css_class = "normal"
    
    for part in parts:
        style = "通用回复"
        reply = part.strip()
        analysis = "暂无分析"
        
        # 优化解析逻辑以适应不同模型的输出风格
        if "回复：" in part and ("点评：" in part or "解析：" in part):
            try:
                # 尝试提取风格
                style_line = [l for l in part.split("\n") if "风格" in l]
                if style_line:
                    style = style_line[0].split("：")[-1].strip()
                else:
                    style = "N/A"
                
                # 提取回复和点评
                if "点评：" in part:
                    reply = part.split("回复：")[1].split("点评：")[0].strip()
                    analysis = part.split("点评：")[1].strip()
                elif "解析：" in part: # DeepSeek 模型可能会使用“解析”
                    reply = part.split("回复：")[1].split("解析：")[0].strip()
                    analysis = part.split("解析：")[1].strip()
            except:
                pass 
        elif "回复：" in part: 
             try:
                style_line = [l for l in part.split("\n") if "风格" in l]
                if style_line:
                    style = style_line[0].split("：")[-1].strip()
                else:
                    style = "N/A"
                reply = part.split("回复：")[1].strip()
                analysis = "请自行分析"
             except:
                pass
        
        if reply: 
            results.append({"style": style, "reply": reply, "analysis": analysis, "css": css_class})
    
    if not results and text.strip():
        results.append({"style": "原始回复", "reply": text.strip(), "analysis": "格式未能识别，显示原始回复。", "css": css_class})

    return results


def get_ai_response(main_mode, sub_mode, input_text, api_key, use_mock=False, extra_detail: str = "", uploaded_image: BytesIO = None, text_model: str = TEXT_MODELS["推荐 (Qwen2.5)"]):
    """
    更新后的 AI 响应函数，支持图片上传和模型选择。
    :param text_model: 用户选择的纯文本模型。
    """
    
    system_prompt_instruction = get_system_prompt(main_mode, sub_mode, extra_detail)

    # --- B. 演示模式 (Mock Data) ---
    if use_mock:
        time.sleep(1)
        if main_mode == "Dating":
            return [{"style": "💔 起死回生术", "reply": "你最近是不是在忙着拯救世界，都不理我了？", "analysis": "幽默且不追问的试探，巧妙地给了台阶。", "css": "dating"}]
        elif main_mode == "Sales":
            return [{"style": "💰 价值锚定法", "reply": "是的，价格确实不低，但您得到的是5年的稳定服务和专属售后，这能为您节省未来数万元的隐性成本。", "analysis": "承认价格，但将客户关注点从价格转到长期收益。", "css": "sales"}]
        elif main_mode == "Normal":
            return [{"style": "😀 轻松幽默式", "reply": f"（面向 {extra_detail}）这事儿就跟等外卖一样，急也没用，不如先去打局游戏。", "analysis": "用日常梗缓解压力。", "css": "normal"}]
        elif main_mode == "Reply_Post":
            img_desc = "(含图片内容)" if uploaded_image else ""
            return [{"style": "🚀 观点升华型", "reply": f"（针对 {extra_detail}{img_desc}）这个观点非常具有启发性，但我们还可以从另一个维度思考，即其长期的社会影响。", "analysis": "在肯定原帖的基础上，提出了更高的思考层次。", "css": "reply_post"}]
        return []

    # --- C. 真实 AI 调用 ---
    else:
        # 确保 OpenAI 库已导入
        if OpenAI is None:
            st.error("OpenAI 库未安装或导入失败。")
            return []
            
        client = OpenAI(api_key=api_key, base_url="https://api.siliconflow.cn/v1")
        
        model_to_use = text_model
        final_messages = []

        # 1. 如果有上传图片，则尝试使用多模态模型 (Vision Model)
        if uploaded_image:
            model_to_use = VISION_MODEL # 覆盖文本模型，切换到视觉模型
            
            # Base64 编码图片
            base64_image = base64.b64encode(uploaded_image.getvalue()).decode("utf-8")
            image_url_data = f"data:image/{uploaded_image.type.split('/')[1]};base64,{base64_image}"
            
            # 构建多模态 prompt
            full_prompt = f"{system_prompt_instruction}\n\n【用户额外提供的文本信息】：\"{input_text}\"\n\n请结合图片和文本信息生成3个选项，并严格按照以下格式输出（纯文本）：\n\n选项1风格：[简短风格名]\n回复：[内容]\n点评：[解析]\n\n(重复3次)"
            
            messages_content = [
                {"type": "text", "text": full_prompt},
                {"type": "image_url", "image_url": {"url": image_url_data}}
            ]
            
            final_messages = [{"role": "user", "content": messages_content}]

        # 2. 没有图片，使用用户选择的纯文本模型
        else:
            full_prompt = f"""
            {system_prompt_instruction}
            
            【待处理内容】："{input_text}"
            
            请生成 3 个选项，并严格按照以下格式输出（纯文本）：
            
            选项1风格：[简短风格名]
            回复：[内容]
            点评：[解析]
            
            (重复3次)
            """
            final_messages = [{"role": "user", "content": full_prompt}]


        try:
            response = client.chat.completions.create(
                model=model_to_use, 
                messages=final_messages,
                temperature=0.7,
                max_tokens=1500 
            )

            raw_text = response.choices[0].message.content
            return parse_ai_response(raw_text, main_mode)
        except Exception as e:
            # 捕获 API 调用错误，并友好提示用户
            error_message = f"API 调用出错 (模型: {model_to_use}): {e}"
            st.error(error_message)
            
            # 如果是多模态模型调用失败，给出明确指导
            if uploaded_image:
                st.warning(f"⚠️ **图片分析模型调用失败**。请注意，我们当前使用的是 `{VISION_MODEL}`。如果该模型持续失败，您需要**联系硅基流动平台**获取一个**确定可用**的多模态模型的**确切名称**，并替换代码中的 `VISION_MODEL` 变量。")
            else:
                st.warning("⚠️ **文本模型调用失败**。请检查侧边栏的 API Key 是否输入正确，或尝试在侧边栏切换到其他 Qwen/DeepSeek 模型。")
                
            return []

# ==========================================
# 4. 主界面 UI
# ==========================================

st.title("🤖 聊天助手")
st.caption("多模式智能对话与内容生成，一键搞定") 

tab_dating, tab_sales, tab_normal, tab_reply_post = st.tabs([
    "模式一 (恋爱辅助 AI)", 
    "模式二 (销售客服 AI)",
    "模式三 (普通模式)",    
    "模式四 (回帖模式)"    
])


# --- Tab 1: 模式一 (恋爱辅助 AI) ---
with tab_dating:
    st.subheader("🎣 提升你的社交匹配率")
    dating_mode = st.radio(
        "选择功能",
        ["开场白生成", "起死回生术"],
        captions=["根据对方资料生成开场白", "对话已死，挽回话术"],
        key='dating_mode_radio'
    )
    dating_input_placeholder = "粘贴对方的个人简介文字，或粘贴你们的冷场对话记录..."
    dating_input = st.text_area("输入待处理内容", height=150, placeholder=dating_input_placeholder, key='dating_input_text')

    if st.button("💘 生成恋爱话术", type="primary", key='dating_button'):
        if not dating_input:
            st.warning("请先输入内容！")
        else:
            use_mock = not current_api_key
            
            with st.spinner(f"正在以【恋爱模式】生成：{dating_mode}..."):
                # 传入选中的文本模型
                results = get_ai_response("Dating", dating_mode, dating_input, current_api_key, use_mock, text_model=current_text_model)
            
            st.markdown("---")
            if use_mock: 
                st.warning("⚠️ 当前 Key 未设置，切换到演示模式。")
            
            for item in results:
                st.markdown(f"""
                <div class="{item['css']}-reply">
                    <span class="dating-tag">🎯 {item['style']}</span>
                    <div class='reply-text'>{item['reply']}</div>
                    <div class='coach-note'><b>💡 技巧解析：</b>{item['analysis']}</div>
                </div>
                """, unsafe_allow_html=True)


# --- Tab 2: 模式二 (销售客服 AI) ---
with tab_sales:
    st.subheader("📈 提高你的客户转化率")
    sales_mode = st.radio(
        "选择功能",
        ["客户异议处理", "产品介绍文案"],
        captions=["针对性解决客户嫌贵/犹豫等问题", "根据信息生成专业推广文案"],
        key='sales_mode_radio'
    )
    sales_input_placeholder = "粘贴客户的异议（比如：太贵了，再考虑），或粘贴产品核心信息..."
    sales_input = st.text_area("输入待处理内容", height=150, placeholder=sales_input_placeholder, key='sales_input_text')

    if st.button("💵 生成销售话术", type="primary", key='sales_button'):
        if not sales_input:
            st.warning("请先输入内容！")
        else:
            use_mock = not current_api_key

            with st.spinner(f"正在以【销售模式】生成：{sales_mode}..."):
                # 传入选中的文本模型
                results = get_ai_response("Sales", sales_mode, sales_input, current_api_key, use_mock, text_model=current_text_model)
            
            st.markdown("---")
            if use_mock: st.warning("⚠️ 当前 Key 未设置，切换到演示模式。")
            
            for item in results:
                st.markdown(f"""
                <div class="{item['css']}-reply">
                    <span class="sales-tag">🎯 {item['style']}</span>
                    <div class='reply-text'>{item['reply']}</div>
                    <div class='coach-note'><b>💡 策略分析：</b>{item['analysis']}</div>
                </div>
                """, unsafe_allow_html=True)

# --- Tab 3: 模式三 (普通模式) ---
with tab_normal:
    st.subheader("🗣️ 日常沟通与内容生成")
    
    normal_relationship = st.radio(
        "选择沟通对象关系",
        ["朋友", "同事", "陌生人", "家人"],
        key='normal_relationship_radio'
    )
    
    normal_input_placeholder = "输入你需要 AI 回复、建议或表达的内容..."
    normal_input = st.text_area("输入待处理内容", height=150, placeholder=normal_input_placeholder, key='normal_input_text')

    if st.button("💡 生成普通回复", type="primary", key='normal_button'):
        if not normal_input:
            st.warning("请先输入内容！")
        else:
            use_mock = not current_api_key

            with st.spinner(f"正在以【普通模式】生成：{normal_relationship} 关系回复..."):
                # 传入选中的文本模型
                results = get_ai_response("Normal", "普通模式", normal_input, current_api_key, use_mock, normal_relationship, text_model=current_text_model)
            
            st.markdown("---")
            if use_mock: st.warning("⚠️ 当前 Key 未设置，切换到演示模式。")
            
            for item in results:
                st.markdown(f"""
                <div class="{item['css']}-reply">
                    <span class="normal-tag">🎯 {item['style']}</span>
                    <div class='reply-text'>{item['reply']}</div>
                    <div class='coach-note'><b>💡 思路总结：</b>{item['analysis']}</div>
                </div>
                """, unsafe_allow_html=True)


# --- Tab 4: 模式四 (回帖模式) ---
with tab_reply_post:
    st.subheader("💬 论坛与社媒互动专家 (支持图片内容)")
    
    reply_post_content_type = st.radio(
        "选择回帖内容类型",
        ["文字/帖子", "网址链接", "截图内容", "视频/音频内容"],
        key='reply_post_content_type_radio'
    )
    
    uploaded_image = None
    if reply_post_content_type == "截图内容":
        # 更新提示，使用当前的 VISION_MODEL
        st.info(f"请上传图片文件进行分析。AI 将尝试调用多模态模型 **{VISION_MODEL}**。")
        uploaded_image = st.file_uploader(
            "上传截图文件 (JPG/PNG)", 
            type=["jpg", "jpeg", "png"], 
            key='reply_post_image_uploader'
        )
        if uploaded_image:
            st.image(uploaded_image, caption="已上传图片", use_column_width=True, width=150)
            st.success("图片已上传，AI 将尝试分析图片内容。")
            reply_post_input_placeholder = "可以输入额外文字说明图片内容..."
        else:
            reply_post_input_placeholder = "请上传截图，并输入额外文字说明图片内容..."
    else:
        reply_post_input_placeholder = f"输入待回复的帖子或对 {reply_post_content_type} 的描述/摘要..."

    reply_post_input = st.text_area("输入待回复内容", height=150, placeholder=reply_post_input_placeholder, key='reply_post_input_text')

    if st.button("✍️ 生成回帖/评论", type="primary", key='reply_post_button'):
        if not reply_post_input and not uploaded_image:
            st.warning("请至少输入内容或上传图片！")
        else:
            use_mock = not current_api_key

            with st.spinner(f"正在以【回帖模式】生成：{reply_post_content_type} 回复..."):
                # 传入 uploaded_image 和 text_model
                results = get_ai_response("Reply_Post", "回帖模式", reply_post_input, current_api_key, use_mock, reply_post_content_type, uploaded_image, current_text_model)
            
            st.markdown("---")
            if use_mock: st.warning("⚠️ 当前 Key 未设置，切换到演示模式。")
            
            for item in results:
                st.markdown(f"""
                <div class="{item['css']}-reply">
                    <span class="reply_post-tag">🎯 {item['style']}</span>
                    <div class='reply-text'>{item['reply']}</div>
                    <div class='coach-note'><b>💡 策略分析：</b>{item['analysis']}</div>
                </div>
                """, unsafe_allow_html=True)