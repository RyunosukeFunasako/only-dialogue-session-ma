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
- 患者が症状に関する返答をした場合は、発話のはじめに繰り返し（言い換え）や共感的な声かけを1文で簡潔に行う。
  - 例：「〇〇ということですね。」「それは〇〇ですね。」
- 各ターンの発話シナリオの内容は生成する発話に必ず含める。
- 発話シナリオに含まれない質問や提案はしない。
- 指示をするような断定的な発話はしない。
  - 例：「まずは〇〇することが大切です。」などの指示的な発話はしない。
- 患者からの質問には回答しながらも、発話シナリオからは逸脱しない。

# 今回のターン{turn}の発話シナリオ：
{counselor_scenario_message}

# 発話シナリオ一覧：
{json.dumps(scenario_data, ensure_ascii=False, indent=2)}
"""
    # カウンセラーのメッセージリストを更新（対話履歴を更新）
    messages_for_counselor = [{"role": "system", "content": counselor_message_prompt}] + dialogue_history[1:]

    counselor_response = openai.chat.completions.create(
        model=model,
        messages=messages_for_counselor,
    )
    counselor_reply = counselor_response.choices[0].message.content.strip()
    return counselor_reply

# 生成された発話を評価する関数
def check_generated_message(counselor_reply, counselor_scenario_message):
    check_prompt = f"""
# 命令書：
あなたはカウンセラーエージェントの発話を管理するエージェントです。
制約条件をもとにカウンセラーが生成した発話が、発話シナリオの内容を含んでいるかを評価してください。

# 制約条件：
- 生成された発話に発話シナリオの内容が含まれていることを確認する。
- 発話シナリオに含まれる説明が省略されていないか確認する。
  - 例：アジェンダや自動思考、認知再構成の説明が省略されていないか確認する。
- 発話シナリオに含まれない質問や提案をしていないか確認する。
- 完全一致している必要はなく、相槌や表現の違いは気にしない。
- 直前の患者の返答に対する繰り返し（言い換え）や共感的な声かけが追加されていることは問題ない。
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
以下はカウンセラーが生成した発話と発話シナリオです。

# カウンセラーの発話：
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
                    "description": "カウンセラーの発話が発話シナリオの内容を含んでいるかを評価する",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "result": {"type": "boolean", "description": "カウンセラーの発話が発話シナリオの内容を含んでいるかを評価する"},
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
    scenario_file = "dialogue-session/counselor_scenario_ver2.json"

    if "counselor_turn" not in st.session_state:
        st.session_state.counselor_turn = 0

    if "messages_for_counselor" not in st.session_state:
        st.session_state.messages_for_counselor = []

    with open(scenario_file, "r") as f:
        scenario_data = json.load(f)["counselor_scenario"]

    # サイドバーにターン進捗を表示
    with st.sidebar:
        st.markdown(f"### 実験の進度")
        st.progress(2 / 5)
        st.markdown(f"### 対話セッションの進捗")
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
                # 3回までは再生成する
                retry_count = 0
                max_retries = 3
                while retry_count < max_retries:
                    counselor_reply = generate_counselor_message(counselor_scenario_message, st.session_state.dialogue_history, openai, model, st.session_state.counselor_turn, scenario_data)
                    # チェックはboolが返ってくるまで何回でも行う
                    check_result = None
                    while not isinstance(check_result, bool):
                        try:
                            check_result = check_generated_message(counselor_reply, counselor_scenario_message)
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
                        else:
                            # 3回目はシナリオ通りの発話を使用
                            print(f"❌ ターン{st.session_state.counselor_turn+1}: 最大再生成回数に達しました。シナリオ通りの発話を使用します。")
                            st.session_state.deviation_history.append(f"❌ ターン{st.session_state.counselor_turn+1}: 最大再生成回数に達しました。シナリオ通りの発話を使用します。")
                            st.session_state.deviation_history.append(f"逸脱と判断された発話：{counselor_reply}")
                            counselor_reply = counselor_scenario_message
                
            # カウンセラーエージェントの発話をストリーム表示
            with st.chat_message("assistant"):
                st.write_stream(stream_counselor_reply(counselor_reply))

            # 対話履歴に追加
            st.session_state.dialogue_history.append({"role": "assistant", "content": counselor_reply})
            st.session_state.messages_for_counselor.append({"role": "assistant", "content": counselor_reply})
    
    # 被験者の入力（23ターン目は入力を求めない）
    if st.session_state.counselor_turn < len(scenario_data) - 1:
        if prompt := st.chat_input("あなたの返答を入力してください。", key=f"chat_input_{len(st.session_state.dialogue_history)}"):
            st.session_state.dialogue_history.append({"role": "user", "content": prompt})
            # 入力を表示
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # 次のターンに進む
            st.session_state.counselor_turn += 1
            st.rerun()
    
    # 23ターン終了
    else:
        st.success("これで対話セッションは終了です。")
        if st.button("説明に戻る"):
            st.session_state.current_page = "description"
            st.rerun()

else:
    st.session_state.current_page = "description"
    st.rerun()