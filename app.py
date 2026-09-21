import streamlit as st

from rag import HospitalRAG

st.set_page_config(page_title="CityCare Hospital Assistant", page_icon="🏥")
st.title("🏥 CityCare Hospital Assistant")
st.caption("Ask about departments, doctors, visiting hours, admission, billing and facilities.")


@st.cache_resource
def get_rag():
    return HospitalRAG()


rag = get_rag()
# Each visitor supplies their own key; it lives only in their session, never on disk or in the repo.
api_key = st.sidebar.text_input("Gemini API key", type="password",
                                help="Free key: https://aistudio.google.com/apikey") or None
if not (api_key or HospitalRAG.make_client()):
    st.info("👈 Enter your Gemini API key in the sidebar to start chatting. "
            "Get a free one at https://aistudio.google.com/apikey")
    st.stop()
st.sidebar.success("API key set")
st.sidebar.markdown("**Try asking:**\n- When can I visit ICU patients?\n- Who is the neurologist and what's the fee?\n"
                    "- What documents are needed for admission?\n- How much is the executive health checkup?")

if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

if q := st.chat_input("Ask a question…"):
    with st.chat_message("user"):
        st.markdown(q)
    with st.chat_message("assistant"):
        res = rag.answer(q, st.session_state.messages, api_key)
        st.markdown(res["answer"])
        if res["sources"]:
            with st.expander("Sources"):
                for s in res["sources"]:
                    st.write(f"{s['source']} › {s['heading']} (similarity {s['score']})")
    st.session_state.messages += [{"role": "user", "content": q}, {"role": "assistant", "content": res["answer"]}]
