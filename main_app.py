import streamlit as st
import pandas as pd
import requests
import time
from datetime import datetime, timedelta
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

# --- 1. 访问密码保护 ---
def check_password():
    def password_entered():
        if st.session_state["password"] == "666888":
            st.session_state["password_correct"] = True
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        st.text_input("请输入实战授权码", type="password", on_change=password_entered, key="password")
        st.warning("🔒 此系统受保护，仅限内部使用")
        return False
    elif not st.session_state["password_correct"]:
        st.text_input("请输入实战授权码", type="password", on_change=password_entered, key="password")
        st.error("❌ 密码错误，请重新输入")
        return False
    else:
        return True

if not check_password():
    st.stop()

# --- 2. 深度进化 AI 引擎 ---
class ProNeuralEngine:
    def __init__(self):
        self.scaler = StandardScaler()

    def train_model(self, df):
        if len(df) < 150: return None
        data = df[['冠军', '亚军', '冠亚和']].iloc[::-1].values
        window = 30 
        X, y = [], []
        for i in range(len(data) - window):
            X.append(data[i:i+window].flatten())
            y.append(data[i+window, 2])
        
        X_scaled = self.scaler.fit_transform(X)
        model = MLPRegressor(hidden_layer_sizes=(128, 64, 32), max_iter=1000, random_state=42, tol=1e-4)
        model.fit(X_scaled, y)
        return model

    def predict_next(self, df, model):
        try:
            last_issue_val = int(df['期号'].iloc[0])
            next_issue = str(last_issue_val + 1)
            latest_feat = df[['冠军', '亚军', '冠亚和']].head(30).iloc[::-1].values.flatten()
            latest_scaled = self.scaler.transform([latest_feat])
            pred_sum = model.predict(latest_scaled)[0]
            dist = abs(pred_sum - round(pred_sum))
            confidence = round(0.55 + (0.35 * (1 - dist * 2)), 2)
            sum_target = "单" if int(round(pred_sum)) % 2 != 0 else "双"
            c1_target = "大" if pred_sum > 11 else "小"
            return confidence, f"冠军-{c1_target}", f"和值-{sum_target}", next_issue
        except:
            return 0.50, "无法预测", "无法预测", "等待数据..."
            # --- 3. 实时数据抓取 ---
@st.cache_data(ttl=5)
def fetch_live_data():
    all_rows = []
    for i in range(3):
        t_date = (datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d')
        url = "https://api.apiose122.com/pks/getPksHistoryList.do"
        params = {"lotCode": "10037", "date": t_date, "pageSize": "1000"}
        try:
            resp = requests.get(url, params=params, timeout=10)
            if resp.status_code == 200:
                data_list = resp.json().get('result', {}).get('data', [])
                for item in data_list:
                    codes = item.get('preDrawCode', "").split(',')
                    if len(codes) < 2: continue
                    c1, c2 = int(codes[0]), int(codes[1])
                    all_rows.append({
                        "期号": str(item.get('preDrawIssue')),
                        "冠军": c1, "亚军": c2, "冠亚和": c1 + c2,
                        "单双": "双" if (c1 + c2) % 2 == 0 else "单",
                        "大小": "大" if c1 > 5 else "小"
                    })
        except: continue
    if not all_rows: return pd.DataFrame()
    return pd.DataFrame(all_rows).drop_duplicates(subset=['期号']).sort_values(by='期号', ascending=False)

# --- 4. 界面展现与结算 ---
st.set_page_config(page_title="AI 神经网络-最终优化版", layout="wide")

if 'history_log' not in st.session_state: st.session_state.history_log = []
if 'profit' not in st.session_state: st.session_state.profit = 0.0
if 'pending_bet' not in st.session_state: st.session_state.pending_bet = None

# 侧边栏控制区 (优化电脑端显示)
with st.sidebar:
    st.header("📊 实战统计中心")
    init_bal = st.number_input("设置起始总分", value=1000.0)
    curr_bal = init_bal + st.session_state.profit
    logs = st.session_state.history_log
    total_r = len(logs)
    wins = len([l for l in logs if "🟢" in l['状态']])
    win_rate = (wins / total_r * 100) if total_r > 0 else 0
    st.metric("累计实战盈亏", f"{st.session_state.profit:.2f}", delta=f"{st.session_state.profit:.2f}")
    st.metric("实战总胜率", f"{win_rate:.1f}%")
    if st.button("🗑️ 清空所有记录"):
        st.session_state.history_log = []
        st.session_state.profit = 0.0
        st.rerun()

# 主界面展示区
st.title("🧠 神经网络 - 高胜率实战决策模型")
df = fetch_live_data()
if not df.empty:
    if st.session_state.pending_bet:
        bet = st.session_state.pending_bet
        match_row = df[df['期号'] == bet['target']]
        if not match_row.empty:
            res = match_row.iloc[0]
            gain = 0.0
            c1_win = res['大小'] == bet['c1'].split('-')[1]
            gain += bet['amt'] * 0.989 if c1_win else -bet['amt']
            ds_target = bet['sum'].split('-')[1]
            ds_win = res['单双'] == ds_target
            ds_odds = 1.2 if ds_target == "双" else 0.79 
            gain += bet['amt'] * ds_odds if ds_win else -bet['amt']
            st.session_state.history_log.insert(0, {
                "期号": bet['target'], "预测内容": f"{bet['c1']} | {bet['sum']}",
                "分值": bet['amt'], "实际结果": f"{res['大小']} | {res['单双']}",
                "盈亏": round(gain, 2), "状态": "🟢 获利" if gain > 0 else "🔴 亏损",
                "时间": datetime.now().strftime("%H:%M:%S")
            })
            st.session_state.profit += gain
            st.session_state.pending_bet = None
            st.rerun()

    engine = ProNeuralEngine()
    model = engine.train_model(df)
    conf, c1_adv, sum_adv, next_iss = engine.predict_next(df, model)
    bet_amt = 0
    if conf >= 0.72: bet_amt = int(curr_bal * 0.1)
    elif conf >= 0.65: bet_amt = int(curr_bal * 0.05)
    if bet_amt > 0 and not st.session_state.pending_bet:
        if not logs or logs[0]['期号'] != next_iss:
            st.session_state.pending_bet = {"target": next_iss, "amt": bet_amt, "c1": c1_adv, "sum": sum_adv}

    col1, col2, col3 = st.columns(3)
    col1.metric("AI 置信度", f"{conf*100:.1f}%")
    col2.metric("🎯 预测目标", f"{next_iss}期")
    col3.success(f"已加载有效样本: {len(df)}期")

    if bet_amt > 0:
        st.error(f"### 🚀 AI 实战指令：针对【{next_iss}】期")
        st.subheader(f"建议方向：{c1_adv} & {sum_adv}")
    else:
        st.info(f"### 📋 【{next_iss}】期观望：模型判定当前规律性较弱")

    st.divider()
    st.subheader("📝 历史下注实战日志")
    if st.session_state.history_log:
        st.table(pd.DataFrame(st.session_state.history_log).head(15))
    st.divider()
    st.write("### 📜 最新数据走势")
    st.table(df.head(10))
    time.sleep(10)
    st.rerun()
