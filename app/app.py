from __future__ import annotations

import os
from datetime import datetime
from typing import Any, Dict, Tuple

import requests
import streamlit as st

BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")


def _init_state() -> None:
    if "history" not in st.session_state:
        st.session_state.history = []
    if "feature_defaults" not in st.session_state:
        st.session_state.feature_defaults = {
            "lag_1": 0.0,
            "lag_2": 0.0,
            "lag_3": 0.0,
            "MA7": 0.0,
            "MA30": 0.0,
            "volatility": 0.0,
        }


def _add_history(kind: str, request_payload: dict, response_payload: dict | str) -> None:
    st.session_state.history.insert(
        0,
        {
            "time": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "type": kind,
            "request": request_payload,
            "response": response_payload,
        },
    )


def _post_json(path: str, payload: Dict[str, Any]) -> Tuple[bool, dict | None, str | None]:
    url = f"{BASE_URL}{path}"
    try:
        response = requests.post(url, json=payload, timeout=10)
    except requests.RequestException as exc:
        return False, None, f"Connection error: {exc}"

    if not response.ok:
        try:
            data = response.json()
            detail = data.get("detail") or data.get("error") or data
        except Exception:
            detail = response.text
        return False, None, f"API error ({response.status_code}): {detail}"

    try:
        return True, response.json(), None
    except Exception:
        return False, None, "Invalid JSON response from API"


def _feature_inputs(section_key: str) -> Dict[str, float]:
    defaults = st.session_state.feature_defaults
    col1, col2, col3 = st.columns(3)
    with col1:
        lag_1 = st.number_input(
            "lag_1",
            key=f"{section_key}_lag_1",
            value=float(defaults["lag_1"]),
            format="%.6f",
        )
        lag_2 = st.number_input(
            "lag_2",
            key=f"{section_key}_lag_2",
            value=float(defaults["lag_2"]),
            format="%.6f",
        )
    with col2:
        lag_3 = st.number_input(
            "lag_3",
            key=f"{section_key}_lag_3",
            value=float(defaults["lag_3"]),
            format="%.6f",
        )
        ma7 = st.number_input(
            "MA7",
            key=f"{section_key}_MA7",
            value=float(defaults["MA7"]),
            format="%.6f",
        )
    with col3:
        ma30 = st.number_input(
            "MA30",
            key=f"{section_key}_MA30",
            value=float(defaults["MA30"]),
            format="%.6f",
        )
        volatility = st.number_input(
            "volatility",
            key=f"{section_key}_volatility",
            value=float(defaults["volatility"]),
            format="%.6f",
        )

    return {
        "lag_1": float(lag_1),
        "lag_2": float(lag_2),
        "lag_3": float(lag_3),
        "MA7": float(ma7),
        "MA30": float(ma30),
        "volatility": float(volatility),
    }


def _example_inputs() -> Dict[str, float]:
    return {
        "lag_1": 0.012,
        "lag_2": 0.008,
        "lag_3": -0.004,
        "MA7": 0.015,
        "MA30": 0.010,
        "volatility": 0.013,
    }


def _render_signal(signal: str) -> None:
    color = {"BUY": "#2e7d32", "SELL": "#c62828", "HOLD": "#616161"}.get(signal, "#616161")
    st.markdown(
        f"<div style='font-size:20px;font-weight:600;color:{color}'>Signal: {signal}</div>",
        unsafe_allow_html=True,
    )


def _render_history() -> None:
    if not st.session_state.history:
        st.caption("No history yet.")
        return
    for item in st.session_state.history[:6]:
        st.markdown(
            f"**{item['type']}** at `{item['time']}`",
        )
        st.json({"request": item["request"], "response": item["response"]})


def _sidebar() -> str:
    st.sidebar.title("Navigation")
    return st.sidebar.radio(
        "Go to",
        ["Prediction", "Decision", "Ask AI"],
        index=0,
    )


def main() -> None:
    st.set_page_config(page_title="AI Financial Forecasting & Assistant", layout="wide")
    _init_state()

    st.title("AI Financial Forecasting & Assistant")
    section = _sidebar()

    st.sidebar.markdown("**API**")
    st.sidebar.write(BASE_URL)

    if st.sidebar.button("Load Example Inputs"):
        st.session_state.feature_defaults = _example_inputs()
        st.sidebar.success("Example inputs loaded")

    if section == "Prediction":
        st.subheader("Prediction")
        features = _feature_inputs("pred")
        if st.button("Predict"):
            payload = {"features": features}
            with st.spinner("Calling prediction API..."):
                ok, data, error = _post_json("/predict", payload)
            if not ok:
                st.error(error)
            else:
                prediction = data.get("prediction")
                st.metric("Predicted value", f"{prediction:.6f}")
                _add_history("predict", payload, data)

    elif section == "Decision":
        st.subheader("Decision")
        features = _feature_inputs("decision")
        if st.button("Get Decision"):
            payload = {"features": features}
            with st.spinner("Calling decision API..."):
                ok, data, error = _post_json("/decision", payload)
            if not ok:
                st.error(error)
            else:
                prediction = float(data.get("prediction", 0.0))
                signal = data.get("signal", "HOLD")
                risk = data.get("risk", "MEDIUM")
                explanation = data.get("explanation", "")

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Prediction", f"{prediction:.6f}")
                with col2:
                    _render_signal(signal)
                with col3:
                    st.metric("Risk", risk)

                st.info(explanation)
                _add_history("decision", payload, data)

    else:
        st.subheader("Ask AI")
        question = st.text_area("Your question", placeholder="Ask about the market or model outputs...")
        if st.button("Ask"):
            if not question.strip():
                st.warning("Please enter a question.")
            else:
                payload = {"question": question.strip()}
                with st.spinner("Calling RAG API..."):
                    ok, data, error = _post_json("/ask", payload)
                if not ok:
                    st.error(error)
                else:
                    st.success("Answer received")
                    st.write(data.get("answer", ""))
                    _add_history("ask", payload, data)

    st.divider()
    st.subheader("Recent Activity")
    _render_history()


if __name__ == "__main__":
    main()
