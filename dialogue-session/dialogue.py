import streamlit as st
from openai import OpenAI
import json
import time

# カウンセラーエージェントの発話生成関数
def generate_counselor_message(counselor_scenario_message, dialogue_history, openai, model, turn, scenario_data):
    counselor_message_prompt = f"""
# 命令書：
あなたは優秀なカウンセラーエージェントです。
以下の制約条件と発話シナリオ、対話履歴をもとに発話を生成してください。

# 制約条件：
- 基本的に発話シナリオに沿って、自然な発話を生成する。
- 患者が困り事、状況、気分、考えを述べた場合は、発話の冒頭で患者の返答に対する繰り返し（言い換え）や共感的な声かけを1文で簡潔に行う。
  - 例：「〇〇ということですね。」「それはつらかったですね。」
- 各ターンの発話シナリオの内容は生成する発話に必ず含める。
- 発話シナリオに含まれる説明や具体例は省略しない。
- 発話シナリオに含まれない説明や質問、提案はしない。
- 患者からの質問には簡潔に回答したうえで、対話の流れが自然になるよう、発話には必ず発話シナリオの内容も含める。
- 指示的な発話や断定的な発話はしない。
  - 例：「まずは〇〇することが大切です。」「〇〇できることが重要です。」

# 今回のターン{turn+1}の発話シナリオ：
{counselor_scenario_message}

# 発話シナリオ一覧：
{json.dumps(scenario_data, ensure_ascii=False, indent=2)}
"""
    # カウンセラーのメッセージリストを更新（対話履歴を更新）
    messages_for_counselor = [{"role": "system", "content": counselor_message_prompt}] + dialogue_history

    counselor_response = openai.chat.completions.create(
        model=model,
        messages=messages_for_counselor,
    )
    counselor_reply = counselor_response.choices[0].message.content.strip()
    return counselor_reply

# 生成された発話を評価する関数
def check_generated_message(previous_user_message, counselor_reply, counselor_scenario_message):
    check_prompt = f"""
# 命令書：
あなたはカウンセラーエージェントが生成した発話を管理するエージェントです。
カウンセラーエージェントは、発話シナリオに沿った発話を行わなければなりません。
制約条件をもとにカウンセラーエージェントが生成した発話が、発話シナリオの内容を全て含んでいるかを評価してください。

# 制約条件：
- 生成された発話に発話シナリオの内容が全て含まれていることを確認する。
- 発話シナリオに含まれる説明や具体例が省略されていないか確認する。
  - 例：認知行動療法の進め方の説明、自動思考の説明と具体例、認知再構成の説明、ホームワークの説明と具体例が省略されていないか確認する。
- 発話シナリオに含まれない説明や質問、提案をしていないか確認する。
- 発話の冒頭に、直前の患者の返答に対する繰り返し（言い換え）や共感的な声かけ、質問に対する回答が追加されていることは問題ない。
"""
    # 評価結果はboolで返す
    check_counselor_reply = openai.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": check_prompt
            },
            {
                "role": "user",
                "content": f"""
以下は直前の患者の発話、カウンセラーエージェントが生成した発話、発話シナリオです。
制約条件をもとにカウンセラーエージェントが生成した発話が、発話シナリオの内容を全て含んでいるかを評価してください。

# 直前の患者の発話：
{previous_user_message}

# カウンセラーエージェントの発話：
{counselor_reply}

# 発話シナリオ：
{counselor_scenario_message}
"""
            }
        ],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "check_generated_message",
                    "description": "カウンセラーエージェントの発話が発話シナリオの内容を含んでいるかを評価する",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "result": {"type": "boolean", "description": "カウンセラーエージェントの発話が発話シナリオの内容を含んでいるかを評価する"},
                        }
                    },
                    "required": [
                        "result",
                    ],
                    "additionalProperties": False
                },
                "strict": True
            }
        ],
        tool_choice="required"
    )
        
    result = check_counselor_reply.choices[0].message.tool_calls[0].function.arguments
    data = json.loads(result)
    return data["result"]

# ストリーム表示を行う関数
def stream_counselor_reply(counselor_reply):
    for chunk in counselor_reply:
        yield chunk
        time.sleep(0.02)

# 対話セッション
if st.session_state.current_page == "dialogue":
    st.title("対話セッション")

    openai = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
    model = "gpt-4o-mini"
    scenario_file = "dialogue-session/counselor_scenario.json"

    if "counselor_turn" not in st.session_state:
        st.session_state.counselor_turn = 0

    if "messages_for_counselor" not in st.session_state:
        st.session_state.messages_for_counselor = []

    with open(scenario_file, "r") as f:
        scenario_data = json.load(f)["counselor_scenario"]

    # サイドバーにターン進捗を表示
    with st.sidebar:
        st.markdown(f"### 対話セッションの進度")
        st.progress((st.session_state.counselor_turn + 1) / len(scenario_data))
        st.markdown(f"**{st.session_state.counselor_turn + 1} / {len(scenario_data)} ターン**")
    
    # 対話履歴を表示し続ける
    for dialogue_history in st.session_state.dialogue_history:
        with st.chat_message(dialogue_history["role"]):
            st.markdown(dialogue_history["content"])
    
    # 現在のターンのカウンセラーエージェントの発話を生成・表示
    if st.session_state.counselor_turn < len(scenario_data):
        # まだ表示されていない発話のみをストリーミング表示する
        if len(st.session_state.messages_for_counselor) == st.session_state.counselor_turn:
            counselor_scenario_message = scenario_data[st.session_state.counselor_turn]["counselor_message"]

            # 1ターン目はシナリオ通りの発話を使用
            if st.session_state.counselor_turn == 0:
                # 表示を遅らせる
                time.sleep(2)
                counselor_reply = counselor_scenario_message
            # 2ターン目以降はカウンセラーエージェントの発話を生成
            else:
                # 3回までは生成する
                retry_count = 0
                max_retries = 3
                while retry_count < max_retries:
                    # 直前の患者の発話
                    previous_user_message = st.session_state.dialogue_history[-1]["content"]

                    counselor_reply = generate_counselor_message(counselor_scenario_message, st.session_state.dialogue_history, openai, model, st.session_state.counselor_turn, scenario_data)
                    # チェックはboolが返ってくるまで何回でも行う
                    check_result = None
                    while not isinstance(check_result, bool):
                        try:
                            check_result = check_generated_message(previous_user_message, counselor_reply, counselor_scenario_message)
                        except Exception as e:
                            print(f"チェックエラーが発生しました。再試行します: {e}")
                    if check_result:
                        break
                    else:
                        retry_count += 1
                        if retry_count < max_retries:
                            print(f"ターン{st.session_state.counselor_turn+1}: 発話がシナリオから逸脱しています。再生成します。（{retry_count}/{max_retries}）")
                            st.session_state.deviation_history.append(f"ターン{st.session_state.counselor_turn+1}: 発話がシナリオから逸脱しています。再生成します。（{retry_count}/{max_retries}）")
                            st.session_state.deviation_history.append(f"逸脱と判断された発話：{counselor_reply}")
                            print(f"逸脱と判断された発話：{counselor_reply}")
                        else:
                            # 2回生成しても発話シナリオから逸脱していた場合は、シナリオ通りの発話を使用
                            print(f"❌ ターン{st.session_state.counselor_turn+1}: 最大再生成回数に達しました。シナリオ通りの発話を使用します。")
                            st.session_state.deviation_history.append(f"❌ ターン{st.session_state.counselor_turn+1}: 最大再生成回数に達しました。シナリオ通りの発話を使用します。")
                            st.session_state.deviation_history.append(f"逸脱と判断された発話：{counselor_reply}")
                            print(f"逸脱と判断された発話：{counselor_reply}")
                            counselor_reply = counselor_scenario_message
                
            # カウンセラーエージェントの発話をストリーム表示
            with st.chat_message("assistant"):
                st.write_stream(stream_counselor_reply(counselor_reply))

            # 対話履歴に追加
            st.session_state.dialogue_history.append({"role": "assistant", "content": counselor_reply})
            st.session_state.messages_for_counselor.append({"role": "assistant", "content": counselor_reply})
    
    # 被験者の入力（23ターン目は入力を求めない）
    if st.session_state.counselor_turn < len(scenario_data) - 1:
        with st.form("chat_form", clear_on_submit=True):
            user_input = st.text_input("あなたの返答を入力してください", key="chat_input")
            submitted = st.form_submit_button("送信")

        if submitted and user_input:
            st.session_state.dialogue_history.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.markdown(user_input)
            st.session_state.counselor_turn += 1
            st.rerun()
    
    # 23ターン終了
    else:
        time.sleep(1)
        st.success("これで対話セッションは終了です。")
        if st.button("説明に戻る"):
            st.session_state.current_page = "description"
            st.session_state.counselor_turn = 0
            st.session_state.messages_for_counselor = []
            st.session_state.dialogue_history = []
            st.session_state.deviation_history = []
            st.rerun()

else:
    st.session_state.current_page = "description"
    st.rerun()